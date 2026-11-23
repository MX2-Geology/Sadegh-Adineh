#!/usr/bin/env bash
# Seed and completeness sensitivity runs (Supplement S8). Run from the repository root after run_all.sh:
#   bash audit/runs/run_variant.sh seed1|seed2|seed3|Mc25|Mc35
# Works on a scratch copy of the package (WORK, default a temporary folder), applies the change in audit/patches
# and copies the resulting tables and logs to audit/runs/<NAME>/. Then rerun code/19_sensitivity.py.
set -euo pipefail
NAME=$1; PKG=$(cd "$(dirname "$0")/../.." && pwd); P=${PYTHON:-python}
W=${WORK:-$(mktemp -d)}; mkdir -p "$W/logs"
cp -r "$PKG/code" "$PKG/configs" "$PKG/out" "$W/"; ln -sfn "$PKG/data" "$W/data"
cd "$W/code"
case $NAME in
  seed[123])
    sed -i "s/^SEED = .*/SEED = ${NAME#seed}/" common.py
    patch -s 17_constructive.py < "$PKG/audit/patches/17_constructive_unrounded_sd.patch"
    $P 17_constructive.py 200 > ../logs/17_constructive.log 2>&1 ;;
  Mc25|Mc35)
    patch -s 12_prepare.py < "$PKG/audit/patches/12_prepare_${NAME}.patch"
    $P 12_prepare.py > ../logs/12_prepare.log 2>&1
    $P -c "import pandas as pd; pd.read_csv('../out/data/cat_A_AFAD_2016_2026.csv').to_csv('../out/data/cat_A_labels.csv', index=False)"
    $P 13_etas.py hom30 "${NAME:2:1}.${NAME:3:1}" 0 > ../logs/13_etas_hom.log 2>&1
    $P 17_constructive.py 200 > ../logs/17_constructive.log 2>&1 ;;
  *) echo "unknown run $NAME"; exit 1 ;;
esac
D="$PKG/audit/runs/$NAME"; mkdir -p "$D/tables" "$D/logs"
cp "$W"/out/tables/T_constructive_* "$D/tables/"; cp "$W"/logs/*.log "$D/logs/"
[[ $NAME == Mc* ]] && cp "$W"/out/tables/{T_Mc_choice.json,T_catalogues.json,T_etas_hom30.json,T_kde_persistence.csv} "$D/tables/"
echo "done: $D"
