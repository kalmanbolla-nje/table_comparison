#!/usr/bin/env python3
"""
Normalization of the tables in Markdown files.

Extracts the tables from every .md file in the input directory, normalizes the
cells (whitespace, lowercasing, number format) and writes the normalized
Markdown files to the output directory.
"""

import re
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────
# 1. Extracting Markdown tables (with positions)
# ─────────────────────────────────────────────────────────

def extract_tables_with_positions(md_text: str) -> list[tuple[int, int, pd.DataFrame]]:
    """Extract every table of a Markdown text with its start/end line index.

    Returns: list of (start_line, end_line, df).
    """
    lines = md_text.strip().split("\n")
    result = []
    current_table_lines = []
    table_start_line = -1
    in_table = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_table_row = bool(re.match(r"^\|.*\|$", stripped))

        if is_table_row:
            if not in_table:
                table_start_line = i
            in_table = True
            current_table_lines.append(stripped)
        else:
            if in_table and current_table_lines:
                df = _parse_table_lines(current_table_lines)
                if df is not None:
                    result.append((table_start_line, i - 1, df))
                current_table_lines = []
            in_table = False

    if current_table_lines:
        df = _parse_table_lines(current_table_lines)
        if df is not None:
            result.append((table_start_line, len(lines) - 1, df))

    return result


def extract_tables_from_markdown(md_text: str) -> list[pd.DataFrame]:
    """Extract every table of a Markdown text as a list of DataFrames."""
    return [item[2] for item in extract_tables_with_positions(md_text)]


#: Separator cell: at least one hyphen between optional colons (GFM rule)
_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")

#: Unescaped pipe: cell boundary; a \| stays part of the cell content
_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")


def _is_separator_row(stripped: str) -> bool:
    """True if the line is a GFM separator row (|---|:---:|---:|)."""
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    segments = [s.strip().replace(" ", "") for s in stripped.split("|")][1:-1]
    return bool(segments) and all(_SEPARATOR_CELL_RE.match(s) for s in segments)


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_BOLD_US_RE = re.compile(r"__(.+?)__")
_ITALIC_WHOLE_RE = re.compile(r"^\*([^*].*?)\*$")
_ITALIC_US_WHOLE_RE = re.compile(r"^_([^_].*?)_$")


def _strip_md_emphasis(text: str) -> str:
    """Remove Markdown emphasis markers from a cell.

    **bold** and __bold__ are markup, not cell content; italics are stripped
    only when the markers span the whole cell (a * inside a cell may be a
    footnote marker).
    """
    text = _BOLD_RE.sub(r"\1", text)
    text = _BOLD_US_RE.sub(r"\1", text)
    m = _ITALIC_WHOLE_RE.match(text)
    if m:
        text = m.group(1).strip()
    m = _ITALIC_US_WHOLE_RE.match(text)
    if m:
        text = m.group(1).strip()
    return text


def _split_cells(line: str) -> list[str]:
    """Split a line into cells.

    An escaped \\| stays in the cell content, while Markdown emphasis markers
    (markup) are removed from the cell value.
    """
    cells = [_strip_md_emphasis(cell.strip().replace("\\|", "|"))
             for cell in _CELL_SPLIT_RE.split(line)]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _parse_table_lines(lines: list[str]) -> Optional[pd.DataFrame]:
    """Build a DataFrame from the lines of a Markdown table.

    Filters out the separator row (|---|:---:|---:|); per the GFM rule a
    separator may only appear directly after the header (as the 2nd line), so
    DATA rows consisting only of hyphens are not dropped by mistake.
    """
    parsed_rows = []
    separator_seen = False
    for line in lines:
        stripped = line.strip()
        # The separator row is filtered only once, in the position right after
        # the header (GFM); later all-hyphen rows are data rows (e.g. missing values)
        if len(parsed_rows) == 1 and not separator_seen and _is_separator_row(stripped):
            separator_seen = True
            continue
        cells = _split_cells(line)
        if cells:
            parsed_rows.append(cells)

    if len(parsed_rows) < 2:
        return None

    header = parsed_rows[0]
    data = parsed_rows[1:]
    max_cols = max(len(header), max(len(r) for r in data))
    header = header + [""] * (max_cols - len(header))
    data = [r + [""] * (max_cols - len(r)) for r in data]
    return pd.DataFrame(data, columns=header)


# ─────────────────────────────────────────────────────────
# 2. Cell normalization
# ─────────────────────────────────────────────────────────

def normalize_cell(value: str, numeric_aware: bool = False) -> str:
    """Normalize a cell value for comparison.

    - Trim whitespace, lowercase, collapse repeated spaces
    - Remove runs of three or more = or - characters (===, ---, ...)
    - Optional numeric-aware mode: '1,250' -> '1250'
    """
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    # Drop runs of three or more = or - (spaces may appear between them).
    # The {3,} bound keeps double hyphens, which can be legitimate content
    # (e.g. '2020--2024'); visual separators (===, ---) are at least 3 marks long.
    value = re.sub(r"(?:=\s*){3,}", "", value)
    value = re.sub(r"(?:-\s*){3,}", "", value)

    if numeric_aware:
        cleaned = value.replace(",", "").replace(" ", "")
        try:
            num = float(cleaned)
            if num == int(num) and "." not in value.replace(",", ""):
                value = str(int(num))
            else:
                value = str(num)
        except ValueError:
            pass
    return value


def normalize_dataframe(df: pd.DataFrame, numeric_aware: bool = False) -> pd.DataFrame:
    """Normalize every cell of a DataFrame."""
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].astype(str).apply(
            lambda v: normalize_cell(v, numeric_aware=numeric_aware)
        )
    return out


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a pipe-separated Markdown table."""
    def row_to_line(row: list) -> str:
        return "| " + " | ".join(str(c) for c in row) + " |"

    header = list(df.columns)
    lines = [row_to_line(header)]
    sep = "| " + " | ".join("---" for _ in header) + " |"
    lines.append(sep)
    for _, r in df.iterrows():
        lines.append(row_to_line(list(r)))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 3. Normalizing a full Markdown document
# ─────────────────────────────────────────────────────────

def normalize_markdown_tables(md_text: str, numeric_aware: bool = False) -> str:
    """Normalize the tables found in a Markdown text.

    The rest of the document is left unchanged.
    """
    lines = md_text.split("\n")
    tables = extract_tables_with_positions(md_text)
    if not tables:
        return md_text

    # Replace from the last table backwards so earlier line indices stay valid
    for start, end, df in reversed(tables):
        normalized_df = normalize_dataframe(df, numeric_aware=numeric_aware)
        table_md = dataframe_to_markdown(normalized_df)
        new_lines = table_md.split("\n")
        lines[start : end + 1] = new_lines

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize the tables of Markdown files (cell text, optionally number format)."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path("output"),
        metavar="DIR",
        help="Input directory to read .md files from (default: output)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output_normalized"),
        metavar="DIR",
        help="Output directory (default: output_normalized)",
    )
    parser.add_argument(
        "-n",
        "--numeric",
        action="store_true",
        help="Also normalize number formats (e.g. 1,250 -> 1250)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Error: input directory not found: {input_dir}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"No .md file found in {input_dir}.")
        return

    print(f"Input: {input_dir} ({len(md_files)} Markdown file(s))")
    print(f"Output: {output_dir}")
    if args.numeric:
        print("Number-format normalization: on")
    print()

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        normalized = normalize_markdown_tables(text, numeric_aware=args.numeric)
        out_path = output_dir / md_path.name
        out_path.write_text(normalized, encoding="utf-8")
        print(f"  {md_path.name} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
