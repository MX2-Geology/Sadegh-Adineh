"""Downloads the third-party input data that are not redistributed in this repository (see README, 'Obtaining the input data').
Run from the repository root:  python code/get_inputs.py [ngl] [isc] [afad]   (no argument = all three). Existing files are kept.
The services are updated and revised over time, so a new download can differ slightly from the copies used for the paper
(14 October 2026); data/CHECKSUMS.sha256 lists the SHA-256 checksums of those copies."""
import io, os, sys, time, urllib.request, urllib.parse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NGL_URL = 'https://geodesy.unr.edu/velocities/midas.IGS14.txt'
ISC_URL = 'https://www.isc.ac.uk/fdsnws/event/1/query'
AFAD_URL = 'https://servisnet.afad.gov.tr/apigateway/deprem/apiv2/event/filter'
BOX_ISC_B = dict(minlatitude=36, maxlatitude=41, minlongitude=25, maxlongitude=31)       # catalogue B files (2000-2025)
BOX_2016 = dict(minlatitude=36, maxlatitude=41, minlongitude=25.5, maxlongitude=31.5)    # AFAD and ISC 2016-2026 files


def get(url, params=None, timeout=900):
    if params: url = url + '?' + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'wanat-density-zoning'}), timeout=timeout) as r:
        return r.read().decode('utf-8')


def isc_text(t0, t1, box, minmag=2.5):
    return get(ISC_URL, dict(starttime=t0, endtime=t1, **box, minmagnitude=minmag, format='text', orderby='time-asc'))


def isc_table(txt):
    L = [l for l in txt.splitlines() if l and (not l.startswith('#') or l.startswith('#EventID'))]
    if len(L) < 2: return pd.DataFrame()
    d = pd.read_csv(io.StringIO('\n'.join(L)), sep='|'); d.columns = [c.strip('#').strip() for c in d.columns]; return d


def ngl(out=f'{ROOT}/data/raw/ngl_midas.txt'):
    if os.path.exists(out): return
    os.makedirs(os.path.dirname(out), exist_ok=True); open(out, 'w').write(get(NGL_URL)); print('NGL MIDAS ->', out)


def isc(years_b=range(2000, 2026), t0='2016-01-01', t1='2026-10-14', outdir=f'{ROOT}/data/raw/isc',
        out2016=f'{ROOT}/data/w_anatolia/earthquakes/isc_fdsn_bbox_2016_2026_M2.5.csv'):
    os.makedirs(outdir, exist_ok=True)
    for y in years_b:                                  # half-year text files, as written by code/dl2.sh
        for a, b in ((f'{y}-01-01', f'{y}-07-01'), (f'{y}-07-01', f'{y + 1}-01-01')):
            f = f'{outdir}/isc_{a}.txt'
            if os.path.exists(f) and os.path.getsize(f): continue
            open(f, 'w').write(isc_text(a, b, BOX_ISC_B)); print('ISC', a); time.sleep(2)
    if not os.path.exists(out2016):
        os.makedirs(os.path.dirname(out2016), exist_ok=True); parts = []
        for y in range(int(t0[:4]), int(t1[:4]) + 1):
            a, b = max(t0, f'{y}-01-01'), min(t1, f'{y + 1}-01-01')
            if a < b: parts.append(isc_table(isc_text(a, b, BOX_2016))); time.sleep(2)
        d = pd.concat(parts, ignore_index=True); d.to_csv(out2016, index=False); print('ISC 2016-2026 ->', out2016, len(d))


def afad(t0='2016-01-01', t1='2026-10-14', out=f'{ROOT}/data/w_anatolia/earthquakes/afad_bbox_2016_2026_M2.5.csv',
         out_low=f'{ROOT}/data/w_anatolia_extra/cat/afad_M1_lowmag.csv'):
    if os.path.exists(out) and os.path.exists(out_low): return
    box = dict(minlat=36, maxlat=41, minlon=25.5, maxlon=31.5); parts = []
    for p in pd.date_range(t0, t1, freq='7D').append(pd.DatetimeIndex([pd.Timestamp(t1)])).unique().to_series().pipe(
            lambda s: list(zip(s[:-1], s[1:]))):
        off = 0
        while True:                                    # the service returns at most 1000 events per request
            txt = get(AFAD_URL, dict(start=p[0].strftime('%Y-%m-%dT%H:%M:%S'), end=p[1].strftime('%Y-%m-%dT%H:%M:%S'),
                                     **box, format='csv', orderby='time', limit=1000, offset=off))
            d = pd.read_csv(io.StringIO(txt))
            if len(d): parts.append(d)
            if len(d) < 1000: break
            off += 1000
        time.sleep(1)
    if not parts:
        sys.exit('AFAD: the event service returned no events for this period; see README (Obtaining the input data).')
    d = pd.concat(parts, ignore_index=True).drop_duplicates('EventID')
    d.columns = [c[0].lower() + c[1:] for c in d.columns]
    for f in (out, out_low): os.makedirs(os.path.dirname(f), exist_ok=True)
    d[d.magnitude >= 2.5].to_csv(out, index=False); d[d.magnitude < 2.5].to_csv(out_low, index=False)
    print('AFAD ->', out, int((d.magnitude >= 2.5).sum()), '|', out_low, int((d.magnitude < 2.5).sum()))


if __name__ == '__main__':
    what = sys.argv[1:] or ['ngl', 'isc', 'afad']
    for w in what: dict(ngl=ngl, isc=isc, afad=afad)[w]()
