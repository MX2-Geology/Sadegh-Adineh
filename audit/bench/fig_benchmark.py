"""Fig. 8: synthetic ETAS benchmark of the support rule. Reads bench_runs.csv and bench_zones.csv (from aggregate.py) in this folder;
writes Fig08_benchmark.jpg/.pdf and fig08_values.json."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, json
import os; B = os.path.dirname(os.path.abspath(__file__)) + '/'
R = pd.read_csv(B + 'bench_runs.csv'); Z = pd.read_csv(B + 'bench_zones.csv')
R = R[R.error.isna()].copy(); R['fit_ok'] = R.br_fit > 0.1
Z = Z.merge(R[['scen', 'E', 'rep', 'fit_ok']], on=['scen', 'E', 'rep'])
Z['supported'] = Z.supported.astype(str).str.lower().eq('true')
VAR = ['ens-DBKM', 'wDB10', 'wDB15', 'wDB20']; COL = dict(zip(VAR, ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']))
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.6))
# (a) H0: catalogues with >= 1 supported zone
h = R[R.scen == 'H0']; zh = Z[(Z.scen == 'H0') & Z.supported]
x = np.arange(len(VAR)); out = {}
ok = [zh[(zh.variant == v) & zh.fit_ok][['rep']].drop_duplicates().shape[0] for v in VAR]
bad = [zh[(zh.variant == v) & ~zh.fit_ok][['rep']].drop_duplicates().shape[0] for v in VAR]
a = ax[0, 0]; a.bar(x, ok, color='0.35', label=f'converged fits (n = {int(h.fit_ok.sum())})'); a.bar(x, bad, bottom=ok, color='0.8', edgecolor='0.35', hatch='///', label=f'degenerate fits (n = {int((~h.fit_ok).sum())})')
a.set_xticks(x, VAR); a.set_ylabel('H0 catalogues with ≥ 1 supported zone'); a.set_ylim(0, max(len(h), 1)); a.legend(frameon=False, loc='upper right')
a.set_title('(a) False positives under H0 (homogeneous background)', loc='left', fontsize=8)
out['H0_false'] = dict(ok=dict(zip(VAR, ok)), degenerate=dict(zip(VAR, bad)), n_ok=int(h.fit_ok.sum()), n_deg=int((~h.fit_ok).sum()))
# (b) power vs E
a = ax[0, 1]; Es = sorted(R[R.scen == 'SRC'].E.unique()); out['power'] = {}
def power(v, E, oracle=False):
    # real variants: converged fits only; oracle uses true labels, so all catalogues
    okm = (lambda d: d.fit_ok) if not oracle else (lambda d: d.fit_ok | True)
    rr = R[(R.scen == 'SRC') & (R.E == E)]; rr = rr[okm(rr)]; n = len(rr)
    if n == 0: return np.nan, 0
    zz = Z[(Z.scen == 'SRC') & (Z.E == E) & (Z.variant == v) & (Z.src >= 0)]; zz = zz[okm(zz)]
    if not oracle: zz = zz[zz.supported]
    return zz[['rep', 'src']].drop_duplicates().shape[0] / (3 * n), n
for v in VAR:
    p = [power(v, E)[0] for E in Es]; a.plot(Es, p, 'o-', color=COL[v], label=v); out['power'][v] = dict(zip(map(str, Es), p))
for eps, ls in [(10, '--'), (15, ':')]:
    p = [power(f'oracle-wDB{eps}', E, True)[0] for E in Es]; a.plot(Es, p, 's' + ls, color='k', mfc='w', label=f'oracle wDB{eps} (any zone)'); out['power'][f'oracle-wDB{eps}'] = dict(zip(map(str, Es), p))
out['kde_H0'] = {k: dict(med=g.kde_med.round(3).tolist(), mx=g.kde_max.round(3).tolist()) for k, g in [('ok', h[h.fit_ok]), ('deg', h[~h.fit_ok])]}
out['n_src_cats'] = {str(E): power('wDB10', E)[1] for E in Es}; out['n_src_cats_oracle'] = {str(E): power('oracle-wDB10', E, True)[1] for E in Es}
a.set_xscale('log'); a.set_xlim(4, 25); a.set_xticks([5, 10, 20], ['5', '10', '20']); a.minorticks_off(); a.set_ylim(-0.03, 1.03); a.set_xlabel('Source enhancement E (× regional background rate)'); a.set_ylabel('Fraction of embedded sources recovered')
a.legend(frameon=False, fontsize=6.5, loc='center right'); a.set_title('(b) Power: supported zones that hit a source', loc='left', fontsize=8)
# (c) KDE ratios of raw DBKM zones under H0
a = ax[1, 0]
for okf, mfc, lab in [(True, '0.3', 'converged fit'), (False, 'none', 'degenerate fit')]:
    hk = h[h.fit_ok == okf]
    a.scatter(hk.kde_med, np.full(len(hk), 1.0), facecolors=mfc, edgecolors='0.3', s=16, label=lab); a.scatter(hk.kde_max, np.full(len(hk), 0.0), facecolors=mfc, edgecolors='0.3', marker='^', s=16)
a.legend(frameon=False, fontsize=6.5, loc='upper right')
a.axvline(1.90, color='C3', lw=1); a.axvline(3.94, color='C3', lw=1, ls='--')
a.text(1.90, 1.35, ' observed median 1.90', color='C3', fontsize=6.5); a.text(3.94, 0.35, ' observed max 3.94', color='C3', fontsize=6.5)
a.set_xscale('log'); a.set_xticks([1, 2, 4, 10, 20], ['1', '2', '4', '10', '20']); a.minorticks_off(); a.set_yticks([0, 1], ['max', 'median']); a.set_ylim(-0.6, 1.9); a.set_xlabel('KDE background ratio of raw DBKM zones (H0 catalogues)')
a.set_title('(c) Selection bias of the KDE ratio under H0', loc='left', fontsize=8)
# (d) synthetic catalogue size vs observed
a = ax[1, 1]; s = R.copy()
for (sc, E), g in s.groupby(['scen', 'E']):
    lab = 'H0' if sc == 'H0' else f'SRC E = {E:g}'; a.scatter(g.n_A, g.br_fit, s=14, label=lab, marker='o' if sc == 'H0' else 'D')
a.axvline(5406, color='C3', lw=1); a.text(5406, 0.45, ' observed 5,406', color='C3', fontsize=6.5)
a.set_xlabel('Events in synthetic catalogue A (M ≥ 3.0)'); a.set_ylabel('Fitted branching ratio'); a.set_ylim(-0.05, 1.0); a.legend(frameon=False, fontsize=6.5, loc='lower right')
a.set_title('(d) Synthetic catalogues and refitted ETAS', loc='left', fontsize=8)
fig.tight_layout(); fig.savefig(B + 'Fig08_benchmark.jpg', dpi=300, pil_kwargs=dict(quality=92)); fig.savefig(B + 'Fig08_benchmark.pdf')
json.dump(out, open(B + 'fig08_values.json', 'w'), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
