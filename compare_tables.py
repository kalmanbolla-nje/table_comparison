#!/usr/bin/env python3
"""
Table comparison: ground truth vs. predicted Markdown/HTML files.

Metrics (all of them work for both Markdown and HTML inputs):
  - EM (exact match):       normalized cell-level agreement — precision, recall, F1
  - NumEM (numeric-aware):  agreement after financial number normalization — precision, recall, F1
  - TEDS:                   tree-edit-distance-based similarity (0.0–1.0),
                            computed on a canonical HTML serialization
  - GriTS_Con:              grid-similarity content metric (0.0–1.0),
                            factored 2D-MSS heuristic (following Smock et al. 2022),
                            with normalized LCS cell similarity

Key properties:
  - Span-aware HTML parsing: rowspan/colspan expanded into the cell grid
  - Tables without thead/tbody handled correctly; nested tables excluded
  - The cell grid and the TEDS HTML are built in ONE pass → no index drift
  - Similarity-based table matching; unmatched GT/predicted tables are reported
    explicitly with a score of 0 and a count (nothing is silently skipped)
  - TEDS failures are reported as a separate counter, never averaged in as 0.0
  - Aggregation is kept separate per format (md/html) — no double counting
  - Several prediction sources (directories) can be compared in a single run
  - Machine-readable output: results.json + per_table.csv in the output directory
"""

import argparse
import csv
import html as html_mod
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite
from pathlib import Path

from bs4 import BeautifulSoup
from table_recognition_metric import TEDS

from normalize_markdown_tables import extract_tables_from_markdown, normalize_cell


# ─────────────────────────────────────────────────────────
# TEDS instance (module level, for performance)
# ─────────────────────────────────────────────────────────

_teds = TEDS(structure_only=False)


# ─────────────────────────────────────────────────────────
# Table representation
# ─────────────────────────────────────────────────────────

@dataclass
class TableData:
    """Two views of one extracted table, guaranteed to come from the same source.

    grid:      span-expanded cell grid (header rows included), every row
               padded to the same length
    teds_html: canonical HTML serialization for TEDS (<td> cells, escaped text,
               rowspan/colspan attributes preserved) — both sides are
               serialized with the same convention, so TEDS does not penalize
               differences in formatting style
    n_header_rows: number of header rows at the top of the grid
    """
    grid: list[list[str]]
    teds_html: str
    n_header_rows: int = 1

    @property
    def n_rows(self) -> int:
        return len(self.grid)

    @property
    def n_cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    @property
    def n_cells(self) -> int:
        return self.n_rows * self.n_cols


def _canonical_teds_html(rows_raw: list[list[tuple[str, int, int]]]) -> str:
    """Build the canonical TEDS HTML: <td> cells, escaped text, span attributes."""
    parts = ["<html><body><table>"]
    for cells in rows_raw:
        parts.append("<tr>")
        for text, rs, cs in cells:
            attrs = ""
            if rs > 1:
                attrs += f' rowspan="{rs}"'
            if cs > 1:
                attrs += f' colspan="{cs}"'
            parts.append(f"<td{attrs}>{html_mod.escape(text)}</td>")
        parts.append("</tr>")
    parts.append("</table></body></html>")
    return "".join(parts)


def _expand_spans(rows_raw: list[list[tuple[str, int, int]]],
                  n_header_rows: int = 0) -> list[list[str]]:
    """Expand rowspan/colspan into the cell grid.

    The value of a merged cell is written into every grid position it covers.
    Full-width single-cell body rows are canonicalized after the expansion
    (see _canonicalize_full_width_rows).
    """
    grid: list[list[str]] = []
    carry: dict[int, tuple[str, int]] = {}  # column index -> (text, remaining rows)

    for cells in rows_raw:
        row: list[str] = []
        ci = 0
        idx = 0
        while idx < len(cells) or ci in carry:
            if ci in carry:
                text, rem = carry[ci]
                row.append(text)
                rem -= 1
                if rem > 0:
                    carry[ci] = (text, rem)
                else:
                    del carry[ci]
                ci += 1
                continue
            text, rs, cs = cells[idx]
            idx += 1
            for _ in range(max(cs, 1)):
                while ci in carry:
                    t2, rem2 = carry[ci]
                    row.append(t2)
                    rem2 -= 1
                    if rem2 > 0:
                        carry[ci] = (t2, rem2)
                    else:
                        del carry[ci]
                    ci += 1
                row.append(text)
                if rs > 1:
                    carry[ci] = (text, rs - 1)
                ci += 1
        grid.append(row)

    if not grid:
        return grid
    max_cols = max(len(r) for r in grid)
    grid = [r + [""] * (max_cols - len(r)) for r in grid]
    return _canonicalize_full_width_rows(grid, rows_raw, max_cols, n_header_rows)


def _canonicalize_full_width_rows(grid: list[list[str]],
                                  rows_raw: list[list[tuple[str, int, int]]],
                                  max_cols: int,
                                  n_header_rows: int) -> list[list[str]]:
    """Canonicalize full-width single-cell rows in the table BODY.

    A `<td colspan="N">Section</td>` row (where N is the full grid width) and
    the `<td>Section</td><td></td>…` spelling denote the SAME visual row: an
    overarching section label that is not a column-specific value. Span
    expansion, however, maps the first to `['Section', 'Section', …]` and the
    second to `['Section', '', …]`, which causes tool-dependent cell mismatches
    — and row misalignment during header fusion. Such rows are therefore
    brought to the canonical form (text in the first cell, the rest empty).

    Body only: in the HEADER a colspan marks a genuine column-group label
    ('Year Ended' above two date columns), where expansion is the correct
    behavior.

    The criterion is exact rather than heuristic: it relies solely on the
    `colspan` attribute, so genuine data rows (e.g. `['—', '—', '—']`) are not
    affected. TEDS is unchanged, as it is computed on the canonical HTML
    serialization.
    """
    for i, cells in enumerate(rows_raw):
        if i < n_header_rows or i >= len(grid) or len(cells) != 1:
            continue
        text, _rs, cs = cells[0]
        if cs >= max_cols > 1:
            grid[i] = [text] + [""] * (max_cols - 1)
    return grid


def _span_int(cell, attr: str) -> int:
    """Safely read a rowspan/colspan attribute (invalid value -> 1)."""
    try:
        v = int(cell.get(attr, 1) or 1)
    except (TypeError, ValueError):
        return 1
    # rowspan="0" (to the end of the section) is simplified to 1 — known limitation
    return v if v >= 1 else 1


def extract_tables_from_html(html_text: str) -> list[TableData]:
    """Extract the top-level <table> elements from HTML text.

    - Nested tables are skipped (they are not duplicated into the parent rows)
    - Without a thead the first row is the header; multi-row theads are supported
    - Rows beside a thead but outside tbody do NOT duplicate the header
    - rowspan/colspan are expanded into the cell grid and preserved in the TEDS HTML
    - The cell grid and the TEDS HTML are built in the same pass (no index drift)
    """
    soup = BeautifulSoup(html_text, "html.parser")
    tables: list[TableData] = []

    for table in soup.find_all("table"):
        if table.find_parent("table") is not None:
            continue  # nested table: skipped to avoid double extraction

        thead = table.find("thead")
        rows_raw: list[list[tuple[str, int, int]]] = []
        n_header_rows = 0

        trs = [tr for tr in table.find_all("tr")
               if tr.find_parent("table") is table]
        for tr in trs:
            cells = []
            for c in tr.find_all(["th", "td"], recursive=False):
                text = c.get_text(separator=" ", strip=True)
                cells.append((text, _span_int(c, "rowspan"), _span_int(c, "colspan")))
            if not cells:
                continue
            in_thead = thead is not None and tr.find_parent("thead") is thead
            if in_thead:
                # Every thead row is a header row and is added only as a header
                # (at the top of the grid, even if the thead comes later in the source)
                rows_raw.insert(n_header_rows, cells)
                n_header_rows += 1
            else:
                rows_raw.append(cells)

        if not rows_raw:
            continue
        if n_header_rows == 0:
            n_header_rows = 1  # no thead: the first row is the header

        grid = _expand_spans(rows_raw, n_header_rows)
        if len(grid) < 2:
            continue  # header without data rows: not an evaluable table

        tables.append(TableData(
            grid=grid,
            teds_html=_canonical_teds_html(rows_raw),
            n_header_rows=n_header_rows,
        ))

    return tables


def extract_tabledata_from_markdown(md_text: str) -> list[TableData]:
    """Build the TableData list from Markdown text (separator-filtering parser)."""
    tables = []
    for df in extract_tables_from_markdown(md_text):
        grid = [[str(c) for c in df.columns]] + \
               [[str(c) for c in df.iloc[r]] for r in range(len(df))]
        rows_raw = [[(cell, 1, 1) for cell in row] for row in grid]
        tables.append(TableData(
            grid=grid,
            teds_html=_canonical_teds_html(rows_raw),
            n_header_rows=1,
        ))
    return tables


def load_tables(file_path: Path) -> list[TableData]:
    """Load the tables of a .md or .html file into a uniform representation."""
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() in (".html", ".htm"):
        return extract_tables_from_html(text)
    return extract_tabledata_from_markdown(text)


# ─────────────────────────────────────────────────────────
# Financial normalization
# ─────────────────────────────────────────────────────────

_NULL_TOKENS = {"", "-", "n/a", "n.a.", "na", "none", "null", "nan", "nil"}
_CURRENCY_SYM_RE = re.compile(r"[$€£¥₹＄]")
_CURRENCY_CODE_RE = re.compile(r"\b(huf|eur|usd|gbp|jpy|chf|ft)\b\.?")
_PAREN_NEG_RE = re.compile(r"^\(\s*([0-9\s.,]+)\s*\)$")
_DASH_TRANS = str.maketrans({"–": "-", "—": "-", "−": "-"})
_SPACE_CHARS = (" ", " ", " ")


def _parse_number_locale_aware(v: str) -> str:
    """Resolve thousands and decimal separators in a locale-aware way.

    Rules:
      - dot AND comma present: the one occurring last is the decimal mark
      - comma only: 1-2 digits after it -> decimal ('3,5' -> 3.5);
        groups of three -> thousands ('1,234,567' -> 1234567)
      - dot only: several dots -> European thousands ('1.234.567' -> 1234567);
        a single dot -> Anglo-Saxon decimal (AMBIGUOUS case: '1.500' -> 1.5,
        resolved by assuming the Anglo-Saxon convention)
    """
    if "." in v and "," in v:
        if v.rfind(",") > v.rfind("."):
            return v.replace(".", "").replace(",", ".")   # European
        return v.replace(",", "")                          # Anglo-Saxon
    if "," in v:
        parts = v.split(",")
        if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            return v.replace(",", ".")                     # decimal comma
        if all(len(p) == 3 for p in parts[1:]):
            return v.replace(",", "")                      # thousands groups
        return v.replace(",", "")
    if v.count(".") > 1:
        parts = v.split(".")
        if all(len(p) == 3 for p in parts[1:]):
            return v.replace(".", "")                      # European thousands
    return v


def normalize_cell_financial(value: str) -> str:
    """Normalize a financial value for comparison.

    On top of the base normalize_cell: Unicode NFKC, dash variants,
    unification of NULL tokens (NormTab recommendation), currency symbols
    AND codes, percent sign, parenthesized negatives, locale-aware numbers.
    """
    v = normalize_cell(value)
    v = unicodedata.normalize("NFKC", v)
    v = v.translate(_DASH_TRANS)

    if v in _NULL_TOKENS:
        return ""

    v = _CURRENCY_SYM_RE.sub("", v).strip()
    v = _CURRENCY_CODE_RE.sub("", v).strip()
    v = v.rstrip("%").strip()

    if v in _NULL_TOKENS:
        return ""

    m = _PAREN_NEG_RE.match(v)
    if m:
        v = "-" + m.group(1)

    for ch in _SPACE_CHARS:
        v = v.replace(ch, "")

    v = _parse_number_locale_aware(v)

    try:
        num = float(v)
        if not isfinite(num):
            return v
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        return v


# ─────────────────────────────────────────────────────────
# Optional header fusion (--fuse-headers)
# ─────────────────────────────────────────────────────────

# Heuristic cap on the header block. The value 5 covers the deepest genuine
# header seen in financial reports (period + years + currency + unit row =
# 4 rows); overrun is stopped by the content tests (numeric data row, section
# label row), not by the cap.
MAX_HEADER_ROWS = 5


def _looks_numeric(cell: str) -> bool:
    v = normalize_cell_financial(cell)
    if not v:
        return False
    try:
        float(v)
        return True
    except (ValueError, OverflowError):
        return False


_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def _is_year(cell: str) -> bool:
    """True if the cell is year-shaped (e.g. '2023') — a column label, not data."""
    return bool(_YEAR_RE.match(normalize_cell_financial(cell)))


def _is_header_continuation(row: list[str]) -> bool:
    """True if the row looks like a continuation of a multi-row header.

    A continuation has EVERY cell outside the first column filled and none of
    them numeric (e.g. a date row below a 'Year Ended' header). Year-shaped
    cells ('2023') count as header-like: in financial reports the header
    continuation row often consists of bare years. Section label rows (one
    text cell plus empty cells) and data rows are not continuations.
    """
    # Full-width repeated row (the signature of colspan expansion): NOT a
    # header continuation. A continuation row gives a DIFFERENT label per
    # column (date, currency); if every cell is identical, the row is one
    # label spanning the whole table width — e.g. the 'Fiscal Year:' section
    # label, which the GT writes with empty companion cells. Without this
    # early exclusion the fusion would be asymmetric: absorbing the row on
    # the colspan side but not on the empty-cell side, shifting all data rows.
    if len(row) >= 2 and len(set(row)) == 1:
        return False
    rest = row[1:] if len(row) > 1 else row
    filled = [c for c in rest if c.strip()]
    if not filled:
        return False
    # At least 2 filled cells required (1 when the remainder is one column) —
    # sparse text data rows (e.g. section label rows) are not continuations,
    # but header continuations with empty edge cells are
    if len(rest) >= 2 and len(filled) < 2:
        return False
    return not any(_looks_numeric(c) and not _is_year(c) for c in filled)


def _is_annotation_row(row: list[str]) -> bool:
    """True if the row is a full-width annotation/unit row at the top of the table.

    Example: '(In millions, except per share data)'. Two surface forms:
      - Markdown: only the first cell is filled, the rest are empty;
      - HTML (after colspan expansion): every non-empty cell has the same text.
    A row with numeric content is not an annotation.
    """
    filled = [c.strip() for c in row if c.strip()]
    if not filled or any(_looks_numeric(c) for c in filled):
        return False
    parenthesized = filled[0].startswith("(") and filled[0].endswith(")")
    if len(filled) == 1:
        if row[0].strip():
            # Single-cell row in column 0: only the parenthesized unit text
            # is an annotation ('(In millions)'); section labels (e.g.
            # 'Contracted but not provided for:') remain data rows
            return parenthesized
        # Single-cell row whose text sits in a middle column: annotation
        return True
    # Repeated (colspan-expanded) row: the SAME parenthesis test decides as in
    # the single-cell column-0 case — so the '(In millions)' unit row stays an
    # annotation, while the 'Fiscal Year:' / 'Inventories (1) :' section label
    # stays a data row, no matter whether that side writes it with colspan or
    # with empty cells. This is what makes the fusion tool-independent.
    return len(set(filled)) == 1 and parenthesized


def fuse_multirow_headers(t: TableData) -> TableData:
    """Fuse a multi-row header into a single header row.

    The fusion runs column by column and keeps repeated fragments only once:
    'Year Ended' + 'January 29, 2023' → 'Year Ended January 29, 2023'.

    Size of the header block: the header rows declared by the thead, extended
    with the _is_header_continuation and _is_annotation_row heuristics (up to
    MAX_HEADER_ROWS rows). This brings the DIFFERENT multi-row-header and unit-row
    conventions of the ground truth and the prediction to a common form, so EM
    measures content errors instead of convention differences. TEDS stays on the
    canonical HTML serialization, since its purpose is to measure structure.
    """
    if len(t.grid) < 2:
        return t
    # The header block starts from the declared <thead> depth and is extended by
    # the content heuristics. The declared depth is part of the block because it
    # carries structural information the content heuristics cannot recover on
    # their own; without it tables with deep headers get an under-estimated
    # block. Known limitation: table pairs whose two sides declare different
    # thead depths stay asymmetric.
    h = max(1, t.n_header_rows)
    # The cap follows the declared thead depth, so even with a deep thead there is
    # room for at least one heuristic row (e.g. a unit row)
    limit = min(max(MAX_HEADER_ROWS, t.n_header_rows + 1), len(t.grid) - 1)
    while h < limit and (
            _is_header_continuation(t.grid[h]) or _is_annotation_row(t.grid[h])):
        h += 1
    if h <= 1:
        return t
    fused = []
    for c in range(len(t.grid[0])):
        parts: list[str] = []
        for r in range(h):
            v = t.grid[r][c].strip()
            if v and v not in parts:
                parts.append(v)
        fused.append(" ".join(parts))
    return TableData(
        grid=[fused] + t.grid[h:],
        teds_html=t.teds_html,
        n_header_rows=1,
    )


# ─────────────────────────────────────────────────────────
# GriTS_Con — factored 2D-MSS heuristic (following Smock et al. 2022)
# ─────────────────────────────────────────────────────────

@lru_cache(maxsize=200_000)
def _lcs_sim(a: str, b: str) -> float:
    """Compute the normalized LCS similarity: 2*|LCS| / (|a|+|b|)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur = [0] * (lb + 1)
        ai = a[i - 1]
        for j in range(1, lb + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return 2.0 * prev[lb] / (la + lb)


def _align_sequences(n_a: int, n_b: int, sim_fn) -> list[tuple[int, int]]:
    """Align two sequences order-preservingly (Needleman-Wunsch, gap = 0).

    Returns the monotone pairing that maximizes the total similarity.
    """
    dp = [[0.0] * (n_b + 1) for _ in range(n_a + 1)]
    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            best = dp[i - 1][j]
            if dp[i][j - 1] > best:
                best = dp[i][j - 1]
            diag = dp[i - 1][j - 1] + sim_fn(i - 1, j - 1)
            if diag > best:
                best = diag
            dp[i][j] = best
    pairs = []
    i, j = n_a, n_b
    while i > 0 and j > 0:
        s = sim_fn(i - 1, j - 1)
        if s > 0 and abs(dp[i][j] - (dp[i - 1][j - 1] + s)) < 1e-12:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def grits_con(gt_grid: list[list[str]], pred_grid: list[list[str]]) -> float:
    """Compute GriTS_Con with the factored 2D-MSS heuristic.

    1. Column alignment (column similarity measured on the positional row overlap)
    2. Row alignment over the aligned columns
    3. Credit = sum of normalized LCS over the aligned (row, column) cell pairs
    F1 = harmonic mean(credit/pred_cells, credit/gt_cells).

    The alignment skips inserted/deleted rows and columns, so these do not shift
    every remaining cell (unlike position-based pairing).
    """
    gt = [[normalize_cell(c) for c in row] for row in gt_grid]
    pred = [[normalize_cell(c) for c in row] for row in pred_grid]

    gt_rows, pred_rows = len(gt), len(pred)
    gt_cols = len(gt[0]) if gt else 0
    pred_cols = len(pred[0]) if pred else 0
    gt_total = gt_rows * gt_cols
    pred_total = pred_rows * pred_cols

    if gt_total == 0 and pred_total == 0:
        return 1.0
    if gt_total == 0 or pred_total == 0:
        return 0.0

    overlap_rows = min(gt_rows, pred_rows)

    def col_sim(a: int, b: int) -> float:
        return sum(_lcs_sim(gt[r][a], pred[r][b]) for r in range(overlap_rows)) / overlap_rows

    col_pairs = _align_sequences(gt_cols, pred_cols, col_sim)
    if not col_pairs:
        return 0.0

    def row_sim(ri: int, rj: int) -> float:
        return sum(_lcs_sim(gt[ri][ca], pred[rj][cb]) for ca, cb in col_pairs) / len(col_pairs)

    row_pairs = _align_sequences(gt_rows, pred_rows, row_sim)

    credit = sum(
        _lcs_sim(gt[ri][ca], pred[rj][cb])
        for ri, rj in row_pairs
        for ca, cb in col_pairs
    )
    precision = credit / pred_total
    recall = credit / gt_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────────────────
# Table-level comparison
# ─────────────────────────────────────────────────────────

@dataclass

class CellDiff:
    row: int
    col: int
    expected: str
    actual: str


@dataclass
class TableComparisonResult:
    gt_index: int | None          # None = unmatched predicted table
    pred_index: int | None        # None = unmatched GT table
    matched: bool
    gt_shape: tuple[int, int]
    pred_shape: tuple[int, int]
    structure_match: bool
    gt_cells: int                 # cells of the GT grid (0 if no GT side)
    pred_cells: int               # cells of the predicted grid (0 if no pred side)
    em_cells: int
    num_em_cells: int
    teds_score: float | None      # None = TEDS failure (reported separately, not 0.0)
    teds_error: str | None
    grits_con_score: float
    diffs: list[CellDiff] = field(default_factory=list)
    header_match: bool = False
    header_diffs: list[dict] = field(default_factory=list)

    def _prf(self, hits: int) -> tuple[float, float, float]:
        recall = hits / self.gt_cells if self.gt_cells else 0.0
        precision = hits / self.pred_cells if self.pred_cells else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        return precision, recall, f1

    @property
    def em(self) -> tuple[float, float, float]:
        """(precision, recall, F1) — plain exact match."""
        return self._prf(self.em_cells)

    @property
    def num_em(self) -> tuple[float, float, float]:
        """(precision, recall, F1) — numeric-aware exact match."""
        return self._prf(self.num_em_cells)


def _effective_cells(t: TableData, skip_header_cells: bool) -> int:
    """Count the cells that form the denominator of the cell metrics.

    When the header is skipped, only the data rows count.
    """
    rows = t.n_rows - (1 if skip_header_cells and t.n_rows > 0 else 0)
    return max(rows, 0) * t.n_cols


def _compare_matched_pair(gt: TableData, pred: TableData,
                          gt_index: int, pred_index: int,
                          skip_header_cells: bool = False) -> TableComparisonResult:
    """Compute the metrics for one matched ground truth / prediction table pair.

    skip_header_cells: in header-fusion mode the cell-level EM/NumEM runs on the
    data rows only — header quality is reported separately by the dedicated header
    comparison (header_match, header_diffs), so the header text noise caused by
    differing colspan widths does not distort the content metrics.
    """
    gt_shape = (gt.n_rows, gt.n_cols)
    pred_shape = (pred.n_rows, pred.n_cols)

    gt_header = [normalize_cell(c) for c in gt.grid[0]] if gt.grid else []
    pred_header = [normalize_cell(c) for c in pred.grid[0]] if pred.grid else []
    header_match = gt_header == pred_header
    header_diffs = [
        {"col": ci, "expected": gt.grid[0][ci], "actual": pred.grid[0][ci]}
        for ci in range(min(len(gt_header), len(pred_header)))
        if gt_header[ci] != pred_header[ci]
    ]

    em_cells = 0
    num_em_cells = 0
    diffs: list[CellDiff] = []
    start_row = 1 if skip_header_cells else 0
    for r in range(start_row, min(gt.n_rows, pred.n_rows)):
        for c in range(min(gt.n_cols, pred.n_cols)):
            gt_raw = gt.grid[r][c]
            pred_raw = pred.grid[r][c]
            if normalize_cell(gt_raw) == normalize_cell(pred_raw):
                em_cells += 1
            else:
                diffs.append(CellDiff(row=r, col=c, expected=gt_raw, actual=pred_raw))
            if normalize_cell_financial(gt_raw) == normalize_cell_financial(pred_raw):
                num_em_cells += 1

    teds_score: float | None = None
    teds_error: str | None = None
    try:
        teds_score = _teds(pred.teds_html, gt.teds_html)
    except Exception as exc:  # report the error, never disguise it as 0.0
        teds_error = f"{type(exc).__name__}: {exc}"

    return TableComparisonResult(
        gt_index=gt_index,
        pred_index=pred_index,
        matched=True,
        gt_shape=gt_shape,
        pred_shape=pred_shape,
        structure_match=gt_shape == pred_shape,
        gt_cells=_effective_cells(gt, skip_header_cells),
        pred_cells=_effective_cells(pred, skip_header_cells),
        em_cells=em_cells,
        num_em_cells=num_em_cells,
        teds_score=teds_score,
        teds_error=teds_error,
        grits_con_score=grits_con(gt.grid, pred.grid),
        diffs=diffs,
        header_match=header_match,
        header_diffs=header_diffs,
    )


def _unmatched_gt_result(gt: TableData, gt_index: int,
                         skip_header_cells: bool = False) -> TableComparisonResult:
    """Build the result of an unmatched GT table.

    Every metric is 0, while its cells stay in the denominator.
    """
    return TableComparisonResult(
        gt_index=gt_index, pred_index=None, matched=False,
        gt_shape=(gt.n_rows, gt.n_cols), pred_shape=(0, 0),
        structure_match=False,
        gt_cells=_effective_cells(gt, skip_header_cells), pred_cells=0,
        em_cells=0, num_em_cells=0,
        teds_score=0.0, teds_error=None, grits_con_score=0.0,
    )


def _unmatched_pred_result(pred: TableData, pred_index: int,
                           skip_header_cells: bool = False) -> TableComparisonResult:
    """Build the result of an unmatched (hallucinated/extra) predicted table.

    It penalizes the precision side.
    """
    return TableComparisonResult(
        gt_index=None, pred_index=pred_index, matched=False,
        gt_shape=(0, 0), pred_shape=(pred.n_rows, pred.n_cols),
        structure_match=False,
        gt_cells=0, pred_cells=_effective_cells(pred, skip_header_cells),
        em_cells=0, num_em_cells=0,
        teds_score=0.0, teds_error=None, grits_con_score=0.0,
    )


MATCH_THRESHOLD = 0.1  # a pair below this similarity is not accepted as a match


def compare_tables(gt_tables: list[TableData],
                   pred_tables: list[TableData],
                   skip_header_cells: bool = False) -> list[TableComparisonResult]:
    """Compare the tables of one file pair with similarity-based table matching.

    Tables are matched monotonically, preserving their order in the document,
    with GriTS_Con as the similarity measure; unmatched tables get an explicit
    zero-score result row (GT side: recall penalty; prediction side: precision
    penalty).
    """
    sim_cache: dict[tuple[int, int], float] = {}

    def table_sim(i: int, j: int) -> float:
        key = (i, j)
        if key not in sim_cache:
            sim_cache[key] = grits_con(gt_tables[i].grid, pred_tables[j].grid)
        return sim_cache[key]

    pairs = _align_sequences(len(gt_tables), len(pred_tables), table_sim)
    pairs = [(i, j) for i, j in pairs if table_sim(i, j) >= MATCH_THRESHOLD]

    matched_gt = {i for i, _ in pairs}
    matched_pred = {j for _, j in pairs}

    results = [
        _compare_matched_pair(gt_tables[i], pred_tables[j], i, j,
                              skip_header_cells=skip_header_cells)
        for i, j in pairs
    ]
    results += [
        _unmatched_gt_result(gt_tables[i], i, skip_header_cells=skip_header_cells)
        for i in range(len(gt_tables)) if i not in matched_gt
    ]
    results += [
        _unmatched_pred_result(pred_tables[j], j, skip_header_cells=skip_header_cells)
        for j in range(len(pred_tables)) if j not in matched_pred
    ]
    return results


# ─────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────

def aggregate(results: list[TableComparisonResult]) -> dict:
    """Aggregate the table results over one format.

    EM and NumEM are cell-pooled (precision, recall, F1), TEDS and GriTS_Con
    are macro-averaged. Tables whose TEDS could not be computed (None) are
    left out of the mean — their count is reported in a separate field.
    Unmatched tables contribute a score of 0 to EVERY metric.
    """
    gt_cells = sum(r.gt_cells for r in results)
    pred_cells = sum(r.pred_cells for r in results)
    em_hits = sum(r.em_cells for r in results)
    num_hits = sum(r.num_em_cells for r in results)

    def prf(hits: int) -> dict:
        recall = hits / gt_cells if gt_cells else 0.0
        precision = hits / pred_cells if pred_cells else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        return {"precision": precision, "recall": recall, "f1": f1}

    teds_valid = [r.teds_score for r in results if r.teds_score is not None]
    teds_errors = sum(1 for r in results if r.teds_error is not None)
    grits_scores = [r.grits_con_score for r in results]

    return {
        "table_count": len(results),
        "matched_tables": sum(1 for r in results if r.matched),
        "unmatched_gt_tables": sum(1 for r in results if not r.matched and r.gt_index is not None),
        "unmatched_pred_tables": sum(1 for r in results if not r.matched and r.pred_index is not None),
        "gt_cells": gt_cells,
        "pred_cells": pred_cells,
        "em": prf(em_hits),
        "num_em": prf(num_hits),
        "teds_mean": sum(teds_valid) / len(teds_valid) if teds_valid else None,
        "teds_error_count": teds_errors,
        "grits_con_mean": sum(grits_scores) / len(grits_scores) if grits_scores else None,
    }


# ─────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────

def print_file_results(name: str, results: list[TableComparisonResult]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    for res in results:
        if not res.matched:
            side = "GT" if res.gt_index is not None else "PREDICTED"
            idx = res.gt_index if res.gt_index is not None else res.pred_index
            shape = res.gt_shape if res.gt_index is not None else res.pred_shape
            print(f"\n  --- UNMATCHED {side} table #{idx + 1} "
                  f"({shape[0]}x{shape[1]}) — scored as 0 ---")
            continue

        print(f"\n  --- Table GT#{res.gt_index + 1} <-> Pred#{res.pred_index + 1} ---")
        if res.structure_match:
            print(f"  Structure:    {res.gt_shape[0]} rows x {res.gt_shape[1]} cols  OK")
        else:
            print(f"  Structure:    GT={res.gt_shape[0]}x{res.gt_shape[1]}, "
                  f"got={res.pred_shape[0]}x{res.pred_shape[1]}  MISMATCH!")
        print(f"  Header:       {'match' if res.header_match else 'mismatch!'}")
        for hd in res.header_diffs[:5]:
            print(f"                  [{hd['col']}] expected='{hd['expected']}' -> got='{hd['actual']}'")

        em_p, em_r, em_f = res.em
        nm_p, nm_r, nm_f = res.num_em
        print(f"  EM:           P={em_p * 100:.1f}%  R={em_r * 100:.1f}%  F1={em_f * 100:.1f}%")
        print(f"  NumEM:        P={nm_p * 100:.1f}%  R={nm_r * 100:.1f}%  F1={nm_f * 100:.1f}%")
        if res.teds_score is not None:
            print(f"  TEDS:         {res.teds_score:.3f}")
        else:
            print(f"  TEDS:         ERROR ({res.teds_error})")
        print(f"  GriTS_Con:    {res.grits_con_score:.3f}")

        if res.diffs:
            print(f"  Differences ({len(res.diffs)} total, first 10):")
            for d in res.diffs[:10]:
                print(f"    [row {d.row + 1}, col {d.col + 1}]  "
                      f"expected='{d.expected}'  ->  got='{d.actual}'")


def print_aggregate(title: str, agg: dict) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"  Tables: {agg['table_count']} "
          f"(matched: {agg['matched_tables']}, "
          f"unmatched GT: {agg['unmatched_gt_tables']}, "
          f"unmatched predicted: {agg['unmatched_pred_tables']})")
    print(f"  GT cells: {agg['gt_cells']}, predicted cells: {agg['pred_cells']}")
    em, nm = agg["em"], agg["num_em"]
    print(f"  EM:           P={em['precision'] * 100:.1f}%  R={em['recall'] * 100:.1f}%  F1={em['f1'] * 100:.1f}%")
    print(f"  NumEM:        P={nm['precision'] * 100:.1f}%  R={nm['recall'] * 100:.1f}%  F1={nm['f1'] * 100:.1f}%")
    teds = agg["teds_mean"]
    print(f"  TEDS:         {teds:.3f}" if teds is not None else "  TEDS:         n/a")
    if agg["teds_error_count"]:
        print(f"  TEDS errors:  {agg['teds_error_count']} tables (excluded from the mean)")
    g = agg["grits_con_mean"]
    print(f"  GriTS_Con:    {g:.3f}" if g is not None else "  GriTS_Con:    n/a")
    print(f"{'-' * 60}")


# ─────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────

def _result_to_dict(res: TableComparisonResult, max_diffs: int = 50) -> dict:
    em_p, em_r, em_f = res.em
    nm_p, nm_r, nm_f = res.num_em
    return {
        "gt_index": res.gt_index,
        "pred_index": res.pred_index,
        "matched": res.matched,
        "gt_rows": res.gt_shape[0], "gt_cols": res.gt_shape[1],
        "pred_rows": res.pred_shape[0], "pred_cols": res.pred_shape[1],
        "structure_match": res.structure_match,
        "gt_cells": res.gt_cells, "pred_cells": res.pred_cells,
        "em_cells": res.em_cells, "num_em_cells": res.num_em_cells,
        "em_precision": em_p, "em_recall": em_r, "em_f1": em_f,
        "num_em_precision": nm_p, "num_em_recall": nm_r, "num_em_f1": nm_f,
        "teds": res.teds_score, "teds_error": res.teds_error,
        "grits_con": res.grits_con_score,
        "header_match": res.header_match,
        "n_diffs": len(res.diffs),
        "diffs": [
            {"row": d.row, "col": d.col, "expected": d.expected, "actual": d.actual}
            for d in res.diffs[:max_diffs]
        ],
    }


def export_results(output_dir: Path, run: dict) -> None:
    """Write results.json (full) and per_table.csv (flat)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(run, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    csv_path = output_dir / "per_table.csv"
    fieldnames = [
        "source", "format", "file", "gt_index", "pred_index", "matched",
        "gt_rows", "gt_cols", "pred_rows", "pred_cols", "structure_match",
        "em_precision", "em_recall", "em_f1",
        "num_em_precision", "num_em_recall", "num_em_f1",
        "teds", "teds_error", "grits_con", "header_match", "n_diffs",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for source in run["sources"]:
            for fmt_block in source["formats"]:
                for file_block in fmt_block["files"]:
                    for t in file_block["tables"]:
                        writer.writerow({
                            "source": source["name"],
                            "format": fmt_block["format"],
                            "file": file_block["file"],
                            **{k: t[k] for k in (
                                "gt_index", "pred_index", "matched",
                                "gt_rows", "gt_cols", "pred_rows", "pred_cols",
                                "structure_match",
                                "em_precision", "em_recall", "em_f1",
                                "num_em_precision", "num_em_recall", "num_em_f1",
                                "teds", "teds_error", "grits_con",
                                "header_match", "n_diffs",
                            )},
                        })
    print(f"\nExport: {json_path}")
    print(f"Export: {csv_path}")


# ─────────────────────────────────────────────────────────
# File pairing helpers
# ─────────────────────────────────────────────────────────

_GT_SUFFIX_RE = re.compile(r"_(\d+)(?=\.[^.]+$)")


def canonical_doc_key(name: str) -> str:
    """Build the canonical document key of a GT file name.

    The trailing _N table index is stripped ('X 43_2.html' -> 'X 43.html'), so
    multi-table pages that are split across several GT files can be matched to
    one prediction file.
    """
    return _GT_SUFFIX_RE.sub("", name)


def gt_suffix_num(name: str) -> int:
    """Return the trailing _N table index of a file name (0 if absent)."""
    m = _GT_SUFFIX_RE.search(name)
    return int(m.group(1)) if m else 0


def load_categories(csv_path: Path) -> dict[str, str]:
    """Read a CSV with file,category columns: GT file name -> category."""
    out: dict[str, str] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("file") and row.get("category"):
                out[row["file"].strip()] = row["category"].strip()
    return out


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def _parse_predict_arg(raw: str) -> tuple[str, Path]:
    """Parse a prediction source argument.

    Accepts 'name=directory' or just 'directory' (the name is then the
    directory name).
    """
    if "=" in raw:
        name, _, path = raw.partition("=")
        return name.strip(), Path(path.strip())
    p = Path(raw)
    return p.name, p


def main() -> None:
    # Force UTF-8 output so printing never fails on a legacy Windows code page
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=(
            "Compare ground-truth and predicted Markdown/HTML tables. "
            "Several prediction sources (e.g. docling, marker, llamaparse) can be "
            "compared in a single run: -p name=directory is repeatable."
        )
    )
    parser.add_argument(
        "-g", "--ground-truth", type=Path,
        default=Path("compare") / "ground_truth", metavar="DIR",
        help="Ground-truth directory (default: compare/ground_truth)",
    )
    parser.add_argument(
        "-p", "--predict", action="append", metavar="[NAME=]DIR",
        help="Directory of predicted files; may be given several times "
             "(default: compare/predict)",
    )
    parser.add_argument(
        "--format", choices=["md", "html", "both"], default="both",
        help="Which format(s) to compare (default: both)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("results"), metavar="DIR",
        help="Output directory for results.json / per_table.csv "
             "(default: results)",
    )
    parser.add_argument(
        "--fuse-headers", action="store_true",
        help="Fuse multi-row headers on both sides before comparing, so that "
             "EM measures content errors instead of header-flattening "
             "convention differences (TEDS is unaffected)",
    )
    parser.add_argument(
        "--pred-prefix", default="", metavar="STR",
        help="Prefix to strip from prediction file names when pairing "
             "(e.g. docling_)",
    )
    parser.add_argument(
        "--common-only", action="store_true",
        help="Evaluate only files present on both sides; ground-truth files "
             "without a prediction are NOT scored 0, they are only counted",
    )
    parser.add_argument(
        "--categories", type=Path, metavar="CSV",
        help="CSV with file,category columns: GT file name -> category "
             "(e.g. Simple/Hierarchical); adds per-category sub-totals",
    )
    args = parser.parse_args()

    gt_dir = args.ground_truth.resolve()
    if not gt_dir.is_dir():
        raise SystemExit(f"Error: ground-truth directory not found: {gt_dir}")

    predict_args = args.predict or [str(Path("compare") / "predict")]
    sources = []
    for raw in predict_args:
        name, path = _parse_predict_arg(raw)
        path = path.resolve()
        if not path.is_dir():
            raise SystemExit(f"Error: prediction directory not found: {path}")
        sources.append((name, path))

    extensions = []
    if args.format in ("md", "both"):
        extensions.append(".md")
    if args.format in ("html", "both"):
        extensions.append(".html")

    print(f"Ground truth: {gt_dir}")
    for name, path in sources:
        print(f"Prediction [{name}]: {path}")

    category_map: dict[str, str] = {}
    if args.categories:
        if not args.categories.is_file():
            raise SystemExit(f"Error: category CSV not found: {args.categories}")
        category_map = load_categories(args.categories)
        print(f"Categories: {args.categories} ({len(category_map)} entries)")

    run: dict = {
        "ground_truth": str(gt_dir),
        "formats": [e.lstrip(".") for e in extensions],
        "fuse_headers": args.fuse_headers,
        "common_only": args.common_only,
        "pred_prefix": args.pred_prefix,
        "sources": [],
    }

    for source_name, pred_dir in sources:
        source_block: dict = {"name": source_name, "path": str(pred_dir), "formats": []}

        for ext in extensions:
            # Group ground truth files: 'X 43_1.html' + 'X 43_2.html' -> key
            # 'X 43.html', ordered by the _N table index
            gt_groups: dict[str, list[Path]] = {}
            for f in gt_dir.glob(f"*{ext}"):
                gt_groups.setdefault(canonical_doc_key(f.name), []).append(f)
            for members in gt_groups.values():
                members.sort(key=lambda p: gt_suffix_num(p.name))

            other_prefixes = tuple(f"{n}_" for n, _ in sources if n != source_name)
            pred_files = {}
            for f in pred_dir.glob(f"*{ext}"):
                name = f.name
                if args.pred_prefix and name.startswith(args.pred_prefix):
                    name = name[len(args.pred_prefix):]
                elif name.startswith(f"{source_name}_"):
                    # Source-name prefix convention: docling_X.html belongs to
                    # the 'docling' source, llamaparse_X.html to 'llamaparse' —
                    # so several sources can share one directory
                    name = name[len(source_name) + 1:]
                elif name.startswith(other_prefixes):
                    # File of another source taking part in this run: skip it
                    continue
                pred_files[name] = f

            common = sorted(set(gt_groups) & set(pred_files))
            gt_only = sorted(set(gt_groups) - set(pred_files))

            print(f"\n{'#' * 60}")
            print(f"  Source: {source_name}  Format: {ext.upper()}  —  "
                  f"{len(common)} shared document(s), {len(gt_only)} GT document(s) "
                  f"without a prediction"
                  + ("  [--common-only: these are skipped]" if args.common_only else ""))
            print(f"{'#' * 60}")

            fmt_results: list[TableComparisonResult] = []
            fmt_categories: list[str | None] = []  # parallel to fmt_results
            file_blocks = []

            def _load_gt_group(name: str) -> tuple[list[TableData], list[str]]:
                tables: list[TableData] = []
                table_files: list[str] = []
                for p in gt_groups[name]:
                    ts = load_tables(p)
                    if args.fuse_headers:
                        ts = [fuse_multirow_headers(t) for t in ts]
                    tables.extend(ts)
                    table_files.extend([p.name] * len(ts))
                return tables, table_files

            for name in common:
                gt_tables, gt_table_files = _load_gt_group(name)
                pred_tables = load_tables(pred_files[name])
                if args.fuse_headers:
                    pred_tables = [fuse_multirow_headers(t) for t in pred_tables]

                if not gt_tables:
                    print(f"\nWarning: {name} — no table in the ground truth, skipped.")
                    continue

                results = compare_tables(gt_tables, pred_tables,
                                         skip_header_cells=args.fuse_headers)
                print_file_results(f"{source_name} / {name}", results)
                fmt_results.extend(results)
                table_dicts = []
                for r in results:
                    gt_file = (gt_table_files[r.gt_index]
                               if r.gt_index is not None else None)
                    category = category_map.get(gt_file) if gt_file else None
                    fmt_categories.append(category)
                    table_dicts.append(_result_to_dict(r) | {
                        "gt_file": gt_file,
                        "category": category,
                    })
                file_blocks.append({
                    "file": name,
                    "gt_table_count": len(gt_tables),
                    "pred_table_count": len(pred_tables),
                    "tables": table_dicts,
                })

            # Ground truth documents with NO prediction file: by default every
            # table in them is scored 0; with --common-only they are reported
            # as a count only
            if not args.common_only:
                for name in gt_only:
                    gt_tables, gt_table_files = _load_gt_group(name)
                    if not gt_tables:
                        continue
                    results = [_unmatched_gt_result(t, i, skip_header_cells=args.fuse_headers)
                               for i, t in enumerate(gt_tables)]
                    print(f"\n  MISSING prediction file: {name} — "
                          f"{len(results)} GT table(s) scored as 0")
                    fmt_results.extend(results)
                    table_dicts = []
                    for r in results:
                        gt_file = gt_table_files[r.gt_index]
                        category = category_map.get(gt_file)
                        fmt_categories.append(category)
                        table_dicts.append(_result_to_dict(r) | {
                            "gt_file": gt_file,
                            "category": category,
                        })
                    file_blocks.append({
                        "file": name,
                        "gt_table_count": len(gt_tables),
                        "pred_table_count": 0,
                        "missing_prediction_file": True,
                        "tables": table_dicts,
                    })

            agg = aggregate(fmt_results)
            print_aggregate(f"TOTAL — {source_name} / {ext.upper()} "
                            f"(per format, no combined average)", agg)

            # Per-category breakdown (e.g. simple vs. hierarchical tables)
            cat_aggs: dict[str, dict] = {}
            if category_map:
                for cat in sorted({c for c in fmt_categories if c}):
                    subset = [r for r, c in zip(fmt_results, fmt_categories)
                              if c == cat]
                    cat_aggs[cat] = aggregate(subset)
                    print_aggregate(f"CATEGORY: {cat} — {source_name} / {ext.upper()}",
                                    cat_aggs[cat])
                uncat = sum(1 for c in fmt_categories if not c)
                if uncat:
                    print(f"\n  Uncategorized tables: {uncat} "
                          f"(included only in the overall aggregate)")

            source_block["formats"].append({
                "format": ext.lstrip("."),
                "files": file_blocks,
                "aggregate": agg,
                "categories": cat_aggs,
                "skipped_gt_documents": len(gt_only) if args.common_only else 0,
            })

        run["sources"].append(source_block)

    export_results(args.output.resolve(), run)


if __name__ == "__main__":
    main()
