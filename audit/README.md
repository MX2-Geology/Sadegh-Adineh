# Sensitivity runs and synthetic benchmark

These folders hold the runs behind Supplement S8 (seed and completeness sensitivity) and Section 2.9/3.6 with Supplement S9 (synthetic ETAS benchmark). All runs use the package code in `../code`; the small changes are in `patches/`. They are optional: `../run_all.sh` reproduces the main analysis, and `code/19_sensitivity.py` reads the tables stored here.

## Seed and completeness runs (`runs/`)

- `runs/run_variant.sh NAME` (NAME = `seed1`, `seed2`, `seed3`, `Mc25` or `Mc35`) works on a scratch copy of the package and copies the tables and logs back to `runs/NAME/`.
- Seed runs: `SEED` in `code/common.py` is set to 1, 2 or 3, and only `17_constructive.py` is rerun. `patches/17_constructive_unrounded_sd.patch` makes it also write unrounded Jaccard values and the replicate standard deviation of J_boot (`*_UNROUNDED.csv`).
- Mc runs: `patches/12_prepare_Mc25.patch` and `patches/12_prepare_Mc35.patch` set Mc for catalogue A to 2.5 or 3.5 (catalogue B is cut at max(Mc, 3.0)). `12_prepare.py`, `13_etas.py hom30` and `17_constructive.py` are rerun.
- After the runs, rerun `code/19_sensitivity.py`.

## Synthetic benchmark (`bench/`)

- `bench.py SCEN E REP`: simulates and processes one synthetic ETAS catalogue (SCEN = H0 or SRC; E = source enhancement; seed = 1000·E + REP + 7, with E = 0 for H0) and writes `out/<SCEN>_E<E>_r<REP>.json`. In our runs, one catalogue took between 3 min and about 2 h (about 17 h in total, four to six jobs at a time) and up to about 3 GB of memory; six parallel jobs exceeded 15 GB once.
- `jobs.txt`: the 36 jobs (12 H0, and 8 for each of E = 5, 10 and 20), all completed. H0 E1 r10, run last on its own, took about 1 h. Run with `xargs -P 4 -L 1 sh -c 'python bench.py $0 $1 $2' < jobs.txt`; use fewer parallel jobs if memory is short.
- `aggregate.py`: `bench_runs.csv`, `bench_zones.csv` and `bench_summary.json` from `out/*.json`; flags degenerate refits (branching ratio ≤ 0.1).
- `summarise.py`: Table 6, Table S13 (`bench_tables.json`) and the numbers quoted in the text (`bench_numbers.json`).
- `fig_benchmark.py`: Fig. 8 (`Fig08_benchmark.jpg/.pdf`, `fig08_values.json`).
