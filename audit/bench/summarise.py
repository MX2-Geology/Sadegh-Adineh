"""Benchmark Table 6 (main text) and Table S13 (supplement) plus a numbers JSON, from bench_runs.csv / bench_zones.csv / out/*.json."""
import pandas as pd, numpy as np, json, glob
import os; B = os.path.dirname(os.path.abspath(__file__)) + '/'
R = pd.read_csv(B + 'bench_runs.csv'); Z = pd.read_csv(B + 'bench_zones.csv')
Z['supported'] = Z.supported.astype(str).str.lower().eq('true')
R['status'] = np.where(R.error.notna(), 'fit error', np.where(R.br_fit > 0.1, 'converged', 'degenerate'))
Z = Z.merge(R[['scen', 'E', 'rep', 'status']], on=['scen', 'E', 'rep'])
VAR = ['ens-DBKM', 'wDB10', 'wDB15', 'wDB20']; ORA = ['oracle-wDB10', 'oracle-wDB15', 'oracle-wDB20']
def lab(s, E): return 'H0' if s == 'H0' else f'SRC, E = {E:g}'
rows = []; N = {}
for (s, E), g in R.groupby(['scen', 'E']):
    ok = g[g.status == 'converged']; k = lab(s, E); n = {}
    n['n'] = len(g); n['ok'] = len(ok); n['deg'] = int((g.status == 'degenerate').sum()); n['err'] = int((g.status == 'fit error').sum())
    gg = g[g.status != 'fit error']
    n['nA_med'] = int(gg.n_A.median()) if len(gg) else None; n['nA_min'] = int(gg.n_A.min()) if len(gg) else None; n['nA_max'] = int(gg.n_A.max()) if len(gg) else None
    n['br_ok_med'] = float(ok.br_fit.median()) if len(ok) else None; n['br_ok_min'] = float(ok.br_fit.min()) if len(ok) else None; n['br_ok_max'] = float(ok.br_fit.max()) if len(ok) else None
    n['bgratio_ok_med'] = float((ok.exp_bg_fit / ok.true_bg).median()) if len(ok) else None
    zs = Z[(Z.scen == s) & (Z.E == E) & Z.supported & Z.variant.isin(VAR)]
    for st in ['converged', 'degenerate']:
        f = zs[(zs.status == st) & (zs.src < 0)]
        n[f'false_cats_{st}'] = int(f.rep.nunique()); n[f'false_zones_{st}'] = int(len(f))
        n[f'false_by_var_{st}'] = {v: int((f.variant == v).sum()) for v in VAR}
    if s == 'SRC':
        for v in VAR:
            h = zs[(zs.status == 'converged') & (zs.variant == v) & (zs.src >= 0)]
            n[f'hits_{v}'] = int(len(h[['rep', 'src']].drop_duplicates())); n['src_total_ok'] = 3 * len(ok)
        for v in ORA:
            h = Z[(Z.scen == s) & (Z.E == E) & (Z.variant == v) & (Z.src >= 0) & (Z.status != 'fit error')]
            n[f'hits_{v}'] = int(len(h[['rep', 'src']].drop_duplicates())); n['src_total_all'] = 3 * len(gg)
    if s == 'H0':
        n['kde_med_ok'] = [float(ok.kde_med.min()), float(ok.kde_med.max())]; n['kde_max_ok'] = [float(ok.kde_max.min()), float(ok.kde_max.max())]
        n['kde_med_ok_ge_1_9'] = int((ok.kde_med >= 1.9).sum()); n['kde_max_ok_ge_3_94'] = int((ok.kde_max >= 3.94).sum())
        dg = g[g.status == 'degenerate']; n['kde_med_deg'] = dg.kde_med.round(2).tolist(); n['kde_max_deg'] = dg.kde_max.round(2).tolist()
    N[k] = n
    fc = f"{n['false_cats_converged']} of {n['ok']}" + (f" ({n['false_cats_degenerate']} of {n['deg']})" if n['deg'] else '')
    hit = '—' if s == 'H0' else ' / '.join(f"{n[f'hits_{v}']}" for v in VAR) + f" of {n['src_total_ok']}"
    ora = '—' if s == 'H0' else ' / '.join(f"{n[f'hits_{v}']}" for v in ORA[:2]) + f" of {n['src_total_all']}"
    rows.append([k, f"{n['n']} ({n['ok']} / {n['deg']} / {n['err']})", f"{n['nA_med']:,} ({n['nA_min']:,}–{n['nA_max']:,})" if n['nA_med'] else '—',
                 f"{n['br_ok_med']:.2f}" if n['br_ok_med'] is not None else '—', fc, hit, ora])
hdr = ['Scenario', 'Catalogues (converged / degenerate / fit error)', 'Events in catalogue A, median (range)', 'Branching ratio, converged fits (median)',
       'Catalogues with ≥ 1 false supported zone: converged (degenerate)', 'Sources hit by supported zones, converged fits: ens-DBKM / wDB10 / wDB15 / wDB20', 'Sources hit by oracle zones: wDB10 / wDB15']
md = '| ' + ' | '.join(hdr) + ' |\n|' + '|'.join([':---'] + ['---:'] * (len(hdr) - 1)) + '|\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows)
cap = ('Synthetic ETAS benchmark of the support rule (J_boot ≥ 0.75 and J_B ≥ 0.5). Degenerate: refitted branching ratio ≤ 0.1, so that almost every event is '
       'background. False supported zone: a supported zone whose event centroid lies more than 25 km from every embedded source centre (in H0, every supported zone). Each SRC catalogue has three sources. Oracle: wDB with the true '
       'background labels as weights, any zone counted. The observed catalogue A has 5,406 events.')
# Table S17: per catalogue
S = []
for _, r in R.sort_values(['scen', 'E', 'rep']).iterrows():
    zz = Z[(Z.scen == r.scen) & (Z.E == r.E) & (Z.rep == r.rep)]
    sup = lambda v: zz[(zz.variant == v) & zz.supported]
    cell = lambda v: f"{len(sup(v))} ({int((sup(v).src >= 0).sum())})" if r.status != 'fit error' else '—'
    S.append([lab(r.scen, r.E), str(int(r.rep)), f"{int(r.n_A):,}" if r.n_A == r.n_A else '—', f"{r.br_fit:.3f}" if r.br_fit == r.br_fit else '—', r.status,
              f"{r.kde_med:.2f}" if r.kde_med == r.kde_med else '—', f"{r.kde_max:.2f}" if r.kde_max == r.kde_max else '—'] + [cell(v) for v in VAR])
hS = ['Scenario', 'Replicate', 'Events A', 'Branching ratio (refit)', 'Fit', 'KDE ratio, median', 'KDE ratio, max'] + [f'{v}: supported (on a source)' for v in VAR]
mdS = '| ' + ' | '.join(hS) + ' |\n|' + '|'.join([':---'] * 2 + ['---:'] * 2 + [':---'] + ['---:'] * (len(hS) - 5)) + '|\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in S)
capS = ('Synthetic benchmark, all catalogues. Seed of each catalogue: 1000·E + replicate + 7 (E = 0 for H0). KDE ratio: P_bg-weighted kernel density inside the raw DBKM zones '
        'relative to the rest of the region (observed median 1.90, maximum 3.94). Supported zones per variant, with the number that overlap an embedded source in parentheses.')
errs = [json.load(open(f)) for f in glob.glob(B + 'out/*.json')]; errs = [dict(scen=e['scen'], E=e['E'], rep=e['rep'], error=e['error']) for e in errs if e.get('error')]
N['_errors'] = errs; N['_n_files'] = len(glob.glob(B + 'out/*.json'))
json.dump(dict(table3=dict(cap=cap, md=md), tableS17=dict(cap=capS, md=mdS)), open(B + 'bench_tables.json', 'w'), ensure_ascii=False, indent=1)
json.dump(N, open(B + 'bench_numbers.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(md); print(); print(json.dumps(N, indent=1, default=float)[:3000])
