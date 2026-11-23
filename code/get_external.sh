#!/usr/bin/env bash
# Downloads the inputs not redistributed in data/ (licence or size). Skips files that already exist.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
[ -s gem_gaf.geojson ] || curl -fsSL -m 600 -o gem_gaf.geojson \
  https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults_harmonized.geojson
mkdir -p gcmt; cd gcmt; U=https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog
[ -s jan76_dec20.ndk ] || curl -fsS -m 600 -O $U/jan76_dec20.ndk
for y in 21 22 23 24; do for m in jan feb mar apr may jun jul aug sep oct nov dec; do
  [ -s $m$y.ndk ] || curl -fsS -m 120 -O $U/NEW_MONTHLY/20$y/$m$y.ndk; done; done
