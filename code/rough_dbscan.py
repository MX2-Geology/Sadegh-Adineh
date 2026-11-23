"""Leaders-based Rough-DBSCAN after Viswanath & Babu (2009, Pattern Recogn. Lett. 30, 1477-1488), as used by Scitovski (2018).
(1) Leaders clustering: points are scanned in the given order; a point within tau of an existing leader becomes its follower,
otherwise it becomes a new leader. (2) Each leader carries its follower count. (3) DBSCAN is run on the leaders with the counts
as sample weights, so a leader is 'core' when the summed counts of leaders within eps reach MinPts. (4) Followers inherit their
leader's label. The published algorithm's exact density estimator and recommended tau could not be checked against the paper
text (paywalled) -> [VERIFY]; we therefore report tau in {eps/4, eps/2} and two scan orders."""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

def leaders(X, tau, order=None):
    order = np.arange(len(X)) if order is None else order
    L = []; lab = np.full(len(X), -1); cnt = []
    for i in order:
        if L:
            P = np.asarray(L); d = np.hypot(*(P - X[i]).T); j = int(np.argmin(d))
            if d[j] <= tau: lab[i] = j; cnt[j] += 1; continue
        L.append(X[i]); cnt.append(1); lab[i] = len(L) - 1
    return np.asarray(L), np.asarray(cnt), lab

def rough_dbscan(X, eps, minpts, tau, order=None):
    L, cnt, fol = leaders(X, tau, order)
    ll = DBSCAN(eps=eps, min_samples=minpts).fit_predict(L, sample_weight=cnt)
    return ll[fol], len(L)
