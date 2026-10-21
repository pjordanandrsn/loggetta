### Training: the Alpaca demonstration runs past row 284

Training without `--dataset` uses the Alpaca demonstration, and it stopped in data preparation at row 284
("column 'output' must contain non-empty text"). That is the first of the 28 rows in `tatsu-lab/alpaca` with an empty
output, so any demonstration run that needed more than about 23k tokens (some 45 steps at `--seq 512`) failed before
loading weights. The default 20 steps stayed under that, so the quickstart was unaffected. The demonstration now skips
exactly those rows and records how many it skipped on every pass (`rows_skipped_empty_output` in the report and the
data profile). It is pinned to the dataset revision it has had since 2023
(`dce01c9b08f87459cf36a430d809084718273017`), so its token stream is reproducible. A dataset you supply is still
validated row by row, and an empty output there is still refused. A pass that yields no usable row now refuses by
name instead of repeating forever (#86).
