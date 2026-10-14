# Table Extraction Comparison

A small, dependency-light evaluator for **table extraction quality**. It compares tables
produced by a document converter (docling, LlamaParse, marker, …) against ground-truth
transcripts of the same tables, and reports cell-level and structure-level agreement.

Both sides may be **Markdown or HTML**; every metric works on either format, and results
for the two formats are aggregated separately so they are never mixed into one average.

## Metrics

| Metric | Level | What it measures |
|---|---|---|
| **EM** (exact match) | cell | Normalized cell text agreement — precision, recall, F1 |
| **NumEM** (numeric-aware) | cell | Same, after financial normalization (currency symbols, thousands separators, dash variants, NULL tokens), so `$ 1,234` and `$1234` count as equal |
| **TEDS** | table | Tree-edit-distance-based similarity on a canonical HTML serialization (Zhong et al., 2020) |
| **GriTS_Con** | table | Grid-similarity content metric, factored 2D-MSS heuristic with normalized LCS cell similarity (following Smock et al., 2022) |

Tables are paired with a document-order-preserving, similarity-based matching. An
unmatched ground-truth table is scored 0 (a recall penalty) and an unmatched predicted
table is scored 0 on the precision side — nothing is silently dropped.

## Install

```bash
pip install -r requirements.txt
```

Three direct dependencies, pinned: `pandas`, `beautifulsoup4`,
`table-recognition-metric`. Tested on Python 3.13.

## Input layout

```
compare/
  ground_truth/      # one table per file: NAME.html (or NAME.md)
  predict/           # converter output, SAME file name
```

Two naming conventions are supported:

- **Multi-table pages.** If a source page holds several tables, split the ground truth
  into `NAME_1.html`, `NAME_2.html`, … — they are grouped and matched against the single
  prediction file `NAME.html`.
- **Several sources in one directory.** Prefix the prediction files
  (`docling_NAME.html`, `llamaparse_NAME.html`) and name each source on the command line.

## Usage

Simplest run — ground truth in `compare/ground_truth`, predictions in `compare/predict`:

```bash
python compare_tables.py
```

Compare two converters that share one prediction directory, HTML only, evaluating just
the files present on both sides:

```bash
python compare_tables.py --format html --common-only -p docling=compare/predict -p llamaparse=compare/predict
```

Run the same comparison again with header fusion, writing to a separate output directory:

```bash
python compare_tables.py --format html --common-only --fuse-headers -o results_fused -p docling=compare/predict -p llamaparse=compare/predict
```

### Why run it twice

Ground truth and converter output often describe the *same* table with different
conventions: a two-row column header may be flattened into one row on one side, a
`(In millions)` unit row may be part of the header on the other. `--fuse-headers` brings
both sides to a common form before comparing, so the cell metrics measure content errors
rather than serialization conventions.

Read the two runs as a pair. The normal run gives the strict, convention-sensitive
picture; the fused run isolates genuine content differences. The gap between them is the
size of the convention noise. TEDS is computed on the original serialization and is
therefore unaffected by fusion — which is a useful check that fusion is not changing the
reconstructed structure.

### Breakdown by table type

Optional. The evaluator runs without it; supplying a CSV with `file,category` columns adds
per-category sub-totals next to the overall aggregate. The file is yours to write — it is
a property of your dataset, not of this tool:

```bash
python compare_tables.py --format html --categories categories.csv
```

```csv
file,category
NASDAQ_AMZN_2024 43.html,Simple
NASDAQ_TSLA_2024 43_1.html,Hierarchical
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `-g, --ground-truth DIR` | `compare/ground_truth` | Ground-truth directory |
| `-p, --predict [NAME=]DIR` | `compare/predict` | Prediction directory; repeatable for several sources |
| `--format {md,html,both}` | `both` | Which format(s) to compare |
| `-o, --output DIR` | `results` | Where to write `results.json` and `per_table.csv` |
| `--fuse-headers` | off | Merge multi-row headers on both sides before comparing |
| `--pred-prefix STR` | — | Prefix to strip from prediction file names when pairing |
| `--common-only` | off | Evaluate only files present on both sides; ground-truth files without a prediction are counted but not scored 0 |
| `--categories CSV` | — | `file,category` map for per-category sub-totals |

## Output

Besides the console report, each run writes two files into the output directory:

- **`results.json`** — the full run: per table, per file, per format and per source,
  including every cell-level difference, the ground-truth file name and the category.
- **`per_table.csv`** — one flat row per compared table, for quick analysis in a
  spreadsheet or with pandas.

## Files

| File | Role |
|---|---|
| `compare_tables.py` | The evaluator: extraction, pairing, metrics, export |
| `normalize_markdown_tables.py` | Markdown table parser and cell normalization |
| `requirements.txt` | Pinned dependencies |

`normalize_markdown_tables.py` also runs standalone to normalize the tables of a directory
of Markdown files:

```bash
python normalize_markdown_tables.py -i input_dir -o output_dir
```

## References

- Zhong, X., ShafieiBavani, E., Jimeno Yepes, A. (2020). *Image-based table recognition:
  data, model, and evaluation.* ECCV 2020, 564–580. [arXiv:1911.10683](https://arxiv.org/abs/1911.10683) — TEDS
- Smock, B., Pesala, R., Abraham, R. (2023). *GriTS: Grid table similarity metric for
  table structure recognition.* ICDAR 2023, 535–549. [arXiv:2203.12555](https://arxiv.org/abs/2203.12555) — GriTS
