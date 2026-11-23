# wanat-density-zoning v3.0

Code, configuration, logs and derived data for the paper

> Adineh, S., Zare, M. *Density clusters or earthquake sequences? An uncertainty-aware reassessment of density-based seismogenic zoning in Western Anatolia, Türkiye.* Submitted to *Computers & Geosciences*.

## Purpose

The code tests whether density-based clustering of epicentres (DBSCAN, Rough-DBSCAN, HDBSCAN, OPTICS) yields seismogenic source zones in Western Anatolia (26–31°E, 36–41°N), or only marks earthquake sequences. It

- reproduces the DBSCAN zoning of Scitovski (2018) for Croatia with exact DBSCAN and Rough-DBSCAN;
- scores the stability of DBSCAN, HDBSCAN and OPTICS zones on a common bootstrap grid (30 configurations, B = 200), with and without noise points;
- measures the share of triggered events in each zone with ETAS stochastic declustering, Zaliapin–Ben-Zion nearest-neighbour declustering and Gardner–Knopoff windows;
- zones the declustered catalogue directly over 20 ETAS realisations (ensemble and probability-weighted DBSCAN) with a support rule fixed in advance (bootstrap Jaccard ≥ 0.75 and Jaccard ≥ 0.5 against an independent period, ISC 2000–2015);
- compares zones with mapped faults, GNSS strain rates and GCMT mechanisms;
- estimates how many background events a decade-long catalogue holds within the clustering radius, and tests the support rule on synthetic ETAS catalogues.

**Contact:** Sadegh Adineh, Institute of Rock Structure and Mechanics, Czech Academy of Sciences, Prague (adineh@irsm.cas.cz).

## Requirements

- Python 3.13 (the runs behind the paper used Python 3.13.5 on Linux with 6 cores).
- The Python packages pinned in `requirements.txt`. The ETAS inversion uses the `etas` package (Mizrahi et al., 2021; MIT licence), installed from GitHub at commit 097f08b; `git` must therefore be available during installation.
- `bash` and `curl` for `run_all.sh` and `code/get_external.sh`. On Windows, use WSL (Ubuntu) or Git Bash; the Python scripts themselves are platform independent.
- About 1 GB of free disk space and an internet connection for the first run. The third-party input data are not included and must be obtained first (see Obtaining the input data).

`code/env_system.txt` lists the package versions of the system used for the paper.

## Installation

Download the repository (`git clone` or "Code → Download ZIP" on GitHub), open a terminal in its folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

With conda: `conda env create -f environment.yml && conda activate wanat-density-zoning`.

## Running

Unit tests (12 tests, a few seconds):

```bash
cd code
python -m pytest -v tests
```

Full pipeline, from the repository root (about 1 h on 6 cores; `14_fair_stability.py` takes about 21 min):

```bash
bash run_all.sh                    # or: PYTHON=/path/to/python bash run_all.sh
```

`run_all.sh` first calls `code/get_external.sh` (GEM fault traces and Global CMT files) and `code/get_inputs.py ngl isc` (NGL MIDAS velocities and ISC Bulletin files); downloads are skipped for files that already exist. It stops if the two AFAD files are missing (see Obtaining the input data). It then runs the steps below and writes logs to `logs/`. Every stochastic step uses the seed 20261028 (`code/common.py`).

| Step | Script | What it does |
|---|---|---|
| 1 | `01_gnss.py`, `02_gcmt.py` | GNSS strain-rate grid (NGL MIDAS) and GCMT subset |
| 2 | `10_catalogue.py` | AFAD (catalogue A, 2016–2026) and ISC (catalogue B, 2000–2015) catalogues, magnitude audit, Mc (MAXC + 0.2, MBS) |
| 3 | `11_mc_spatial.py` | Spatial Mc and the analysis Mc |
| 4 | `12_prepare.py` | Projection; Zaliapin–Ben-Zion and Gardner–Knopoff declustering |
| 5 | `13_etas.py hom30 3.0 0`, `ker30 3.0 1`, `hom32 3.2 0` | ETAS inversion and 20 stochastic realisations per run |
| 6 | `14_fair_stability.py 200` | 30-configuration stability grid for DBSCAN, HDBSCAN and OPTICS |
| 7 | `15_kde_persistence.py` | Background-density ratio and persistence |
| 8 | `16_validate3.py` | Zones, fault test, depth, geodesy and T-axes |
| 9 | `17_constructive.py 200` | Constructive zoning (ensemble and weighted DBSCAN) and the support rule |
| 10 | `18_detectability.py` | Expected background events within ε and years to reach MinPts |
| 11 | `19_sensitivity.py` | Seed and Mc sensitivity summary from `audit/runs` |
| 12 | `20_rough_croatia.py` | Scitovski (2018) reproduction with exact and Rough-DBSCAN |
| 13 | `fig01_area.py`, `figs3.py` | Figures 1–7 and S1–S2 (Fig. 8: `audit/bench/fig_benchmark.py`) |
| 14 | `pytest` | Unit tests |
| 15 | `30_check_numbers.py` | Checks the numbers quoted in the paper against `out/tables` and `audit/bench` (runs only if the plain-text paper is placed in `doc/`) |

With the input files listed in `data/CHECKSUMS.sha256`, a full rerun reproduces every file in `out/` exactly; only the run times written to the logs differ.

The seed and completeness sensitivity runs and the synthetic ETAS benchmark take several hours more and are optional; their results are already in `audit/`. How to rerun them is described in `audit/README.md` and at the end of `run_all.sh`.

`30_check_numbers.py` compares the numbers with the text of the paper (`doc/manuscript.md`, `doc/highlights.md`, `doc/supplement.md`). The text is not part of this repository while the paper is under review; the log of the check run for the submitted version is in `logs/30_check_numbers.log` (134 passed, 0 failed).

## Repository map

| Path | Contents |
|---|---|
| `code/` | Pipeline scripts (numbered in run order), shared modules (`common.py`, `clustering.py`, `decluster.py`, `rough_dbscan.py`, `wa3.py`, `figbase.py`), figure scripts, numbers check, download scripts (`get_inputs.py`, `get_external.sh`, `dl.sh`, `dl2.sh`) |
| `code/tests/` | Unit tests (ε rule, Mc (MAXC, MBS) and b estimation, pruning, declustering, DBSCAN/HDBSCAN, Rough-DBSCAN, grid comparison, projection) |
| `configs/run_config.json` | All parameters and command lines of the runs (documentation only; the values are set in the code) |
| `run_all.sh` | Runs the full pipeline |
| `data/` | USGS ComCat files (public domain) and `CHECKSUMS.sha256` for the input files that are not redistributed; the other inputs are placed here by the user (see Obtaining the input data) |
| `out/data/` | ETAS background probabilities per AFAD event ID (`*_Pbg.csv`) and the 20 stochastic realisations (`*_realisations.npy`), ensemble cell memberships, GNSS strain-rate grid. The derived event catalogues are not included (they repeat the third-party records); `run_all.sh` rebuilds them in `out/data/` |
| `out/tables/` | All tables quoted in the paper: `T_*` main text, `S_*` supplement (CSV and JSON) |
| `out/figs/` | Figures 1–7 and S1–S2 as drawn by the code (JPG, 400 dpi). In the paper, Figs 2, 5, S1 and S2 carry small label or colour edits made on the image; the plotted data are the same |
| `logs/` | Logs of the runs behind the paper |
| `audit/runs/` | Seed (1–3) and completeness (Mc 2.5, 3.5) sensitivity runs; `audit/patches/` holds the small code changes they use |
| `audit/bench/` | Synthetic ETAS benchmark (36 catalogues), its tables and Fig. 8 |
| `CITATION.cff`, `LICENSE`, `requirements.txt`, `environment.yml` | Citation metadata, licence and environment |

All tables and figures of the paper are included, so the results can be inspected without rerunning anything. Not included are the third-party input files and the event catalogues derived from them (`out/data/afad_all_region.csv`, `isc_B_all_region.csv`, `cat_A_AFAD_2016_2026.csv`, `cat_A_labels.csv`, `cat_A_labels_constructive.csv`, `cat_B_ISC_2000_2015.csv`, `gcmt_box.csv`, `gnss_midas_eurasia.csv`) and the event-level table `out/tables/S_afad_vs_ref_Mw_matches.csv`. Once the inputs are in place, `bash run_all.sh` regenerates all of them together with everything else in `out/` and `logs/`; `audit/` is regenerated with the commands in `audit/README.md`. No file in the repository exceeds 2 MB.

## Data sources and terms

The code is MIT-licensed; the data are not relicensed and remain under their providers' terms. "Verified" means that the terms were read on the provider's website on 23 November 2026; "needs checking" means that no explicit licence for redistribution was found. Only the USGS files are therefore included.

| Source | Terms | Included |
|---|---|---|
| AFAD event web service (Disaster and Emergency Management Authority, Türkiye) | Needs checking: public service, no published licence found. Credit: AFAD, Deprem Dairesi Başkanlığı (deprem.afad.gov.tr) | No |
| ISC Bulletin (International Seismological Centre) | Needs checking: freely accessible, no licence stated; citation required (International Seismological Centre, On-line Bulletin, https://doi.org/10.31905/D808B830) | No |
| NGL MIDAS GNSS velocities (Nevada Geodetic Laboratory) | Needs checking: free download, no licence stated; NGL asks users to cite Blewitt et al. (2016, 2018) | No |
| Global CMT catalogue | Needs checking: free download, no licence stated; cite Dziewonski et al. (1981) and Ekström et al. (2012) | No |
| GEM Global Active Faults | Verified: CC BY-SA 4.0 (repository licence); cite Styron and Pagani (2020) | No (downloaded) |
| USGS ComCat | Verified: USGS-authored data are in the U.S. public domain; credit the U.S. Geological Survey | Yes: `data/raw/usgs/`, `data/raw/usgs_croatia_1950_2017-05-25.csv` |

## Obtaining the input data

All paths are relative to the repository root. `data/CHECKSUMS.sha256` holds the SHA-256 checksums of the AFAD, ISC and NGL files used for the paper (`sha256sum -c data/CHECKSUMS.sha256`). The services are revised and updated, so a new download can differ from these copies; the AFAD and ISC files used for the paper were downloaded on 14 October 2026.

**ISC Bulletin** (FDSN event web service, https://www.isc.ac.uk/fdsnws/event/1/query, `format=text`, `orderby=time-asc`, `minmagnitude=2.5`). Downloaded by `python code/get_inputs.py isc` (also called by `run_all.sh`):

- `data/raw/isc/isc_YYYY-01-01.txt` and `isc_YYYY-07-01.txt`: half-year files for 2000–2025 (52 files), box 36–41°N, 25–31°E. They give catalogue B (2000–2015). The same files are written by `bash code/dl2.sh 2000 2001 … 2025` (run from the repository root). A test download on 23 November 2026 reproduced the 2000 files byte for byte.
- `data/w_anatolia/earthquakes/isc_fdsn_bbox_2016_2026_M2.5.csv`: 2016-01-01 to 2026-10-14, box 36–41°N, 25.5–31.5°E, converted from the pipe-separated text output to CSV (22,128 events in the copy used). It is used for the cross-match of AFAD magnitudes with independent Mw.

**NGL MIDAS velocities** (IGS14 frame): https://geodesy.unr.edu/velocities/midas.IGS14.txt, saved as `data/raw/ngl_midas.txt` by `python code/get_inputs.py ngl` (also called by `run_all.sh`). The file is updated weekly; on 23 November 2026 the download was identical to the copy used for the paper.

**AFAD catalogue** (catalogue A): AFAD event web service, https://servisnet.afad.gov.tr/apigateway/deprem/apiv2/event/filter, CSV format, box `minlat=36&maxlat=41&minlon=25.5&maxlon=31.5`, 2016-01-01 to 2026-10-13, all magnitudes. Two files are expected, with the columns returned by the service (`rms, eventID, location, latitude, longitude, depth, type, magnitude, country, province, district, neighborhood, date, isEventUpdate, lastUpdateDate`; the case of the first letter does not matter):

- `data/w_anatolia/earthquakes/afad_bbox_2016_2026_M2.5.csv`: events with M ≥ 2.5 (16,083 rows in the copy used);
- `data/w_anatolia_extra/cat/afad_M1_lowmag.csv`: events with M < 2.5 (90,978 rows in the copy used).

`10_catalogue.py` merges the two files and removes duplicate event IDs, so a single complete download split at M 2.5 also works. `python code/get_inputs.py afad` pages through the service and writes both files. On 23 November 2026, however, the service returned only events of the last few weeks, so the 2016–2026 records could not be downloaded again. In that case the catalogue has to be obtained from AFAD (https://deprem.afad.gov.tr) and saved in the two files above.

**GEM Global Active Faults and Global CMT**: downloaded by `bash code/get_external.sh` (called by `run_all.sh`) to `data/raw/gem_gaf.geojson` (from https://github.com/GEMScienceTools/gem-global-active-faults, `geojson/gem_active_faults_harmonized.geojson`) and `data/raw/gcmt/*.ndk` (`jan76_dec20.ndk` and the monthly files for 2021–2024 from https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog).

**USGS ComCat**: included. `data/raw/usgs_croatia_1950_2017-05-25.csv` is used by `20_rough_croatia.py` (Scitovski, 2018, reproduction).

## References

- Blewitt, G., Hammond, W., Kreemer, C., 2018. Harnessing the GPS data explosion for interdisciplinary science. Eos 99. https://doi.org/10.1029/2018EO104623
- Blewitt, G., Kreemer, C., Hammond, W.C., Gazeaux, J., 2016. MIDAS robust trend estimator for accurate GPS station velocities without step detection. Journal of Geophysical Research: Solid Earth 121, 2054–2068. https://doi.org/10.1002/2015JB012552
- Dziewonski, A.M., Chou, T.-A., Woodhouse, J.H., 1981. Determination of earthquake source parameters from waveform data for studies of global and regional seismicity. Journal of Geophysical Research: Solid Earth 86, 2825–2852. https://doi.org/10.1029/JB086iB04p02825
- Ekström, G., Nettles, M., Dziewoński, A.M., 2012. The global CMT project 2004–2010: centroid-moment tensors for 13,017 earthquakes. Physics of the Earth and Planetary Interiors 200–201, 1–9. https://doi.org/10.1016/j.pepi.2012.04.002
- Mizrahi, L., Nandan, S., Wiemer, S., 2021. The effect of declustering on the size distribution of mainshocks. Seismological Research Letters 92, 2333–2342. https://doi.org/10.1785/0220200231
- Scitovski, S., 2018. A density-based clustering algorithm for earthquake zoning. Computers & Geosciences 110, 90–95. https://doi.org/10.1016/j.cageo.2017.08.014
- Styron, R., Pagani, M., 2020. The GEM Global Active Faults Database. Earthquake Spectra 36, 160–180. https://doi.org/10.1177/8755293020944182

## Citation

If you use this code, please cite the paper above (reference to be completed on publication) and this repository; `CITATION.cff` holds the metadata (GitHub shows it under "Cite this repository"). An archived release with a DOI will be added when the paper is accepted.

## Licence

Code: MIT (see `LICENSE`). Data: see Data sources and terms.
