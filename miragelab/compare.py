"""Lean, no-generation comparison of candidate directions on cached base-prompt states (runs/<run>/acts).

One row per question text (identical texts give identical no-image states). Fit on half the texts, score on the
other half (stratified, --splits repeats):
  auc     held-out AUC of the type-centred projection, mirage vs abstain (no-image states): is it a usable monitor?
  rel     cosine between the two halves' fits: is it signal or noise at this n?
  cos_img cosine with the image direction
Directions: mirage-abstain diff of means at n = 7 / 25 / all mirages (n=7 is the fold vectors' size);
image = with-image minus no-image state (paired per text); nulls: shuffled-label, isotropic random,
activation-spread random. Plus, from prior work: agree = mean pairwise cosine of per-example difference vectors
(Braun et al. 2025, predicts steerability); an LR probe on the full state; its Hewitt & Liang control task (labels
shuffled within type); and non-activation baselines (question type only, question text TF-IDF): a probe that
can't beat the text baseline is reading the question, not a mirage state.

  python -m miragelab.compare --run runs/slake --layers 24,30,34   (CPU, ~1 min)
  python -m miragelab.compare --selftest
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

unit = lambda v: v / np.linalg.norm(v)


def center(X, cat, ref):
    """Subtract each type's mean over `ref` rows (falls back to all rows of that type)."""
    X = X.copy()
    for c in np.unique(cat):
        s = cat == c
        X[s] -= X[s & ref].mean(0) if (s & ref).any() else X[s].mean(0)
    return X


def fit_dirs(X, Ximg, y, half, rng):
    """X: no-image states (type-centred), Ximg: with-image states, y: mirage mask; fit on `half` rows only."""
    m, a = np.where(half & y)[0], np.where(half & ~y)[0]
    dm = lambda mi, ai: unit(X[mi].mean(0) - X[ai].mean(0))
    d = {f"mirage n={n}": dm(rng.choice(m, n, False), rng.choice(a, min(len(a), 2 * n), False)) for n in (7, 25, 100) if n <= len(m)}
    d["mirage all"] = dm(m, a)
    # mirage = the model acting as if an image were there, so the sign points with-image minus no-image
    d["image"] = unit((Ximg[half] - X[half]).mean(0))
    p = rng.permutation(np.r_[m, a])
    d["null shuffled-label"] = dm(p[: len(m)], p[len(m):])
    d["null isotropic"] = unit(rng.normal(size=X.shape[1]))
    Xh = X[half] - X[half].mean(0)
    d["null activation"] = unit(Xh.T @ rng.normal(size=len(Xh)))
    return d


def agreement(D):
    """Braun et al. 2025: mean pairwise cosine of per-example difference vectors (rows of D); low -> steering is unreliable."""
    U = D / np.linalg.norm(D, axis=1, keepdims=True)
    n = len(U)
    return float((np.linalg.norm(U.sum(0)) ** 2 - n) / (n * (n - 1)))


def compare(X, Ximg, y, cat, texts=None, splits=5, seed=0):
    """One row per question text (no-image states depend on the text only, so duplicates would leak across halves).
    Returns {name: (auc mean, auc sd, rel, cos_img, agree)}; nulls' rel is meaningless (fresh draw per half)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder
    rng = np.random.default_rng(seed)
    auc, rel, cimg, agr = {}, {}, {}, {}
    add = lambda dct, k, v: dct.setdefault(k, []).append(v)
    for _ in range(splits):
        h = np.zeros(len(y), bool)
        for c in (True, False):                      # stratified halves
            ii = rng.permutation(np.where(y == c)[0]); h[ii[: len(ii) // 2]] = True
        fits = []
        for tr in (h, ~h):
            Xc = center(X, cat, tr)
            Xi = Ximg - (X - Xc)   # same type means for both conditions, so the image offset survives
            d = fit_dirs(Xc, Xi, y, tr, rng)
            for k, v in d.items():
                add(auc, k, roc_auc_score(y[~tr], Xc[~tr] @ v))
                add(cimg, k, abs(v @ d["image"]))
            add(agr, "mirage all", agreement(Xc[tr & y] - Xc[tr & ~y].mean(0)))
            add(agr, "image", agreement((Xi - Xc)[tr]))
            probe = lambda lab: LogisticRegression(C=0.01, max_iter=2000).fit(Xc[tr], lab[tr]).decision_function(Xc[~tr])
            add(auc, "probe (LR)", roc_auc_score(y[~tr], probe(y)))   # ponytail: fixed C, tune if it matters
            ys = y.copy()                            # Hewitt & Liang control: labels shuffled within type, scored on the shuffle
            for c in np.unique(cat):
                ii = np.where(cat == c)[0]; ys[ii] = ys[rng.permutation(ii)]
            add(auc, "probe control task", roc_auc_score(ys[~tr], probe(ys)) if 0 < ys[~tr].sum() < (~tr).sum() else np.nan)
            oh = OneHotEncoder(handle_unknown="ignore").fit(cat[tr, None])
            add(auc, "baseline: type only", roc_auc_score(y[~tr], LogisticRegression().fit(oh.transform(cat[tr, None]), y[tr])
                                                          .decision_function(oh.transform(cat[~tr, None]))))
            if texts is not None:
                tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(texts[tr])
                add(auc, "baseline: text TF-IDF", roc_auc_score(y[~tr], LogisticRegression(C=1, max_iter=2000)
                                                                .fit(tf.transform(texts[tr]), y[tr]).decision_function(tf.transform(texts[~tr]))))
            fits.append(d)
        for k in fits[0]:
            add(rel, k, fits[0][k] @ fits[1][k])
    m = lambda dct, k: np.mean(dct.get(k, [np.nan]))
    return {k: (np.nanmean(v), np.nanstd(v), m(rel, k), m(cimg, k), m(agr, k)) for k, v in auc.items()}


def report(res):
    print(f"   {'direction':22s} {'held-out AUC':>14s} {'rel':>6s} {'cos_img':>8s} {'agree':>6s}")
    for k, (m, s, r, c, g) in res.items():
        print(f"   {k:22s} {m:8.3f}±{s:.3f} {r:6.2f} {c:8.2f} {g:6.2f}", flush=True)


def selftest():
    """Planted mirage direction + planted image offset: real directions must beat nulls."""
    rng = np.random.default_rng(0)
    n, d = 1200, 64
    y = rng.random(n) < 0.3
    cat = rng.integers(0, 4, n)
    w, g = unit(rng.normal(size=d)), unit(rng.normal(size=d))
    X = rng.normal(size=(n, d)) + 2 * np.outer(y, w) + 3 * cat[:, None]
    res = compare(X, X + 2 * g, y, cat, splits=2)
    assert res["mirage all"][0] > 0.8 and res["mirage all"][2] > 0.8, res["mirage all"]
    assert res["image"][4] > 0.9 and res["mirage all"][4] < res["image"][4]         # constant offset agrees, noisy class doesn't
    assert abs(res["probe control task"][0] - 0.5) < 0.1 and abs(res["baseline: type only"][0] - 0.5) < 0.1
    assert abs(res["null isotropic"][0] - 0.5) < 0.1 and abs(res["null shuffled-label"][0] - 0.5) < 0.1
    assert res["image"][2] > 0.9 and res["image"][0] < 0.6 and res["probe (LR)"][0] > 0.65   # image offset is unrelated to y here
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/slake")
    ap.add_argument("--layers", default="24,30,34")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    from . import analysis as A
    df, acts, meta = A.load(a.run)
    idx, lab, _, _ = A.unique_texts(df)              # one row per text: the no-image state is a function of the text
    keep = np.isin(lab, ["mirage", "abstain"])
    idx, y = idx[keep], lab[keep] == "mirage"
    cat, texts = df.category.values[idx], df.question.map(A.norm_text).values[idx]
    ci, cn = meta["cond"].index("with_image"), meta["cond"].index("without_image")
    print(f"{len(idx)} unique texts: {y.sum()} mirage, {(~y).sum()} abstain (base prompt, no image)")
    for L in map(int, a.layers.split(",")):
        j = meta["layers"].index(L)
        X, Xi = (np.asarray(acts[idx, c, 0, j], np.float32) for c in (cn, ci))
        print(f"L{L}")
        report(compare(X, Xi, y, cat, texts, a.splits))


if __name__ == "__main__":
    main()
