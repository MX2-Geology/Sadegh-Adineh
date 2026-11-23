#!/usr/bin/env bash
# Full pipeline, run from the repository root. About 1 h on 6 cores (14_fair_stability ~21 min).
set -euo pipefail
cd code
P=${PYTHON:-python}
bash get_external.sh                                       # GEM faults and GCMT NDK files (not redistributed)
$P get_inputs.py ngl isc                                   # NGL MIDAS velocities and ISC Bulletin files (not redistributed)
for f in ../data/w_anatolia/earthquakes/afad_bbox_2016_2026_M2.5.csv ../data/w_anatolia_extra/cat/afad_M1_lowmag.csv; do
  [ -s "$f" ] || { echo "Missing AFAD file $f; see README, 'Obtaining the input data'."; exit 1; }
done
$P 01_gnss.py            > ../logs/01_gnss.log 2>&1      # needs data/raw/ngl_midas.txt
$P 02_gcmt.py            > ../logs/02_gcmt.log 2>&1      # needs data/raw/gcmt/*.ndk (see README)
$P 10_catalogue.py       > ../logs/10_catalogue.log 2>&1
$P 11_mc_spatial.py      > ../logs/11_mc_spatial.log 2>&1
$P 12_prepare.py         > ../logs/12_prepare.log 2>&1
$P 13_etas.py hom30 3.0 0 > ../logs/13_etas_hom30.log 2>&1
$P 13_etas.py ker30 3.0 1 > ../logs/13_etas_ker30.log 2>&1
$P 13_etas.py hom32 3.2 0 > ../logs/13_etas_hom32.log 2>&1
$P 14_fair_stability.py 200 > ../logs/14_fair_stability.log 2>&1
$P 15_kde_persistence.py > ../logs/15_kde_persistence.log 2>&1
$P 16_validate3.py       > ../logs/16_validate3.log 2>&1
$P 17_constructive.py 200 > ../logs/17_constructive.log 2>&1
$P 18_detectability.py  > ../logs/18_detectability.log 2>&1
$P 19_sensitivity.py    > ../logs/19_sensitivity.log 2>&1   # reads audit/runs (seed and Mc runs, Supplement S8)
$P 20_rough_croatia.py   > ../logs/20_rough_croatia.log 2>&1
$P fig01_area.py && $P figs3.py
$P -m pytest -v tests    > ../logs/tests.log 2>&1
# Numbers check against the text of the paper; needs doc/manuscript.md, doc/highlights.md and doc/supplement.md (see README)
if [ -f ../doc/manuscript.md ]; then $P 30_check_numbers.py > ../logs/30_check_numbers.log 2>&1; fi

# Optional (several hours; results are already in audit/):
#   for r in seed1 seed2 seed3 Mc25 Mc35; do bash ../audit/runs/run_variant.sh $r; done && $P 19_sensitivity.py
#   cd ../audit/bench && xargs -P 6 -L 1 sh -c '$0 bench.py $1 $2 $3' $P < jobs.txt && $P aggregate.py && $P summarise.py && $P fig_benchmark.py
