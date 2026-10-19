"""Labels, probes and 2-D embeddings for the Qwen3-VL-4B mirage replication.

Labels follow Ben-Levi et al. (Mirage Probes) on the WITHOUT-image response:
  admit   - response acknowledges the missing image / uncertainty
  mirage  - no acknowledgement AND same answer as with-image (bag-of-words cos >= .7)
  neutral - everything else
`label_fixed` adds admission phrasings their regex misses (smoke: "Since I cannot
view the actual image" was labelled mirage).
"""
import json, os, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Data root holding runs/, raw_data/, data/ (default: the working directory).
ROOT = Path(os.environ.get("MIRAGE_ROOT", "."))
# Upstream Mirage Probes checkout (default: the upstream/ submodule next to this package).
UPSTREAM = Path(os.environ.get("MIRAGE_UPSTREAM", Path(__file__).resolve().parents[1] / "upstream"))
sys.path.insert(0, str(UPSTREAM))
import scripts.data.gen_mutations_get_responses as gen  # noqa: E402

EXTRA_ADMIT = [
    r"\b(cannot|can't|can not|unable to|do not|don't|am not able to)\s+(directly\s+)?(view|see|access|examine|analy[sz]e|inspect|look at)\s+(the\s+)?(actual|specific|provided|attached|radiology|medical)?\s*(image|images|scan|radiograph|picture)\b",
    r"\bwithout (access to|seeing|viewing) (the\s+)?(actual\s+)?(image|scan|radiograph)\b",
    r"\b(no|not) (actual\s+)?(image|scan|radiograph) (is|was|has been) (provided|attached|included|shared|available)\b",
    r"\bimage (provided |attached |itself )?(is|was) not (visible|provided|attached|included|available|shared)\b",
]


def admits(text):
    ni, unc = gen._detect_acknowledgement(text)
    return bool(ni or unc or any(re.search(p, (text or "").lower()) for p in EXTRA_ADMIT))


NON_ANSWER = re.compile(r"^(unknown|not? (sure|known|determinable|applicable|available)|cannot (be )?determine(d)?|can ?not (be )?determined|"
                        r"unable to (determine|answer)|undetermined|indeterminate|insufficient (information|data)|n/?a|"
                        r"no image( (is )?provided| available)?|not provided|image not provided|cannot answer)\b")


def outcome(rec):
    """Open-answer outcome of the WITHOUT-image reply (chance of a right guess is ~0 here):
    abstain = final answer is a non-answer; hedged = admits the image is missing but
    still answers concretely; mirage = concrete answer, no admission."""
    w = rec["without_image"]
    pred = (w.get("pred_normalized") or "").strip()
    if not pred or NON_ANSWER.match(pred):
        return "abstain"
    return "hedged" if admits(w["response"]) else "mirage"


def label(rec, fixed=True):
    w = rec["without_image"]
    if rec["with_image"]["response_missing"] or w["response_missing"]:
        return "neutral"
    if (admits(w["response"]) if fixed else w["acknowledged_missing_or_uncertain"]):
        return "admit"
    return "mirage" if w["response_similarity_with_image"] else "neutral"


def leak_groups(*keys):
    """Connected components over rows that share ANY key value (e.g. base question id
    and normalised question text). Use as CV groups so no input crosses a split."""
    n = len(keys[0])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for key in keys:
        first = {}
        for i, k in enumerate(key):
            if k in first:
                a, b = find(i), find(first[k])
                if a != b:
                    parent[a] = b
            else:
                first[k] = i
    return np.array([find(i) for i in range(n)])


def norm_text(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def unique_texts(df):
    """One row per normalised question text (no-image input = system prompt + text, so
    identical texts give identical states). Returns (row indices into df, majority
    outcome per text, n records per text, whether the text's outcome was unanimous)."""
    t = df.question.map(norm_text).values
    out = []
    for txt in pd.unique(t):
        ii = np.where(t == txt)[0]
        vc = pd.Series(df.outcome.values[ii]).value_counts()
        out.append((ii[0], vc.index[0], len(ii), len(vc) == 1))
    idx, lab, n, stable = map(np.array, zip(*out))
    return idx, lab, n, stable


def text_split(texts, frac_fit=0.7, seed=0, strat=None):
    """Deterministic fit/eval split of unique items, stratified by `strat` if given."""
    rng = np.random.default_rng(seed)
    texts = np.asarray(texts)
    fit = np.zeros(len(texts), bool)
    for g in (np.unique(strat) if strat is not None else [None]):
        ii = np.where(strat == g)[0] if strat is not None else np.arange(len(texts))
        fit[rng.permutation(ii)[: int(round(frac_fit * len(ii)))]] = True
    return fit


def row(r):
    return dict(uid=r["unique_id"], variant=r["variant_id"], question=r["variant_question_text"],
                gt=r["ground_truth_normalized"], category=r["category"],
                with_resp=r["with_image"]["response"], no_resp=r["without_image"]["response"],
                with_correct=r["with_image"]["correct"], no_correct=r["without_image"]["correct"],
                with_pred=r["with_image"].get("pred_normalized"), no_pred=r["without_image"].get("pred_normalized"),
                label_paper=label(r, fixed=False), label=label(r, fixed=True),
                no_commits="[[" in r["without_image"]["response"],
                outcome=outcome(r) if r.get("task_type") == "open" and "image_path" in r else None,
                split=r.get("split"), image_id=r.get("image_id"), modality=r.get("modality"),
                prior_correct=(r.get("prior") or {}).get("correct"))


def records(run_dir):
    """Every generated record (behaviour rates must come from this, not the extracted subset)."""
    return pd.DataFrame([row(r) for r in json.load(open(Path(run_dir) / "responses.json"))])


def with_prior(run_dir):
    """records() + text-only prior (prior.json) + commit = no admission of the missing image."""
    f = records(run_dir)
    p = pd.DataFrame(json.load(open(Path(run_dir) / "prior.json")))
    f = f.merge(p, on=["uid", "variant"], how="left", validate="one_to_one")
    f["commit"] = f.label != "admit"
    return f


def agreement(a, b):
    """Observed vs chance agreement of two answer columns (Cohen's kappa).
    The paper's mirage label requires with/without answers to match; on yes/no
    a large share of matches is expected by chance alone."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    obs = float((a == b).mean())
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    exp = float(sum(pa.get(k, 0) * pb.get(k, 0) for k in set(pa.index) | set(pb.index)))
    return obs, exp, (obs - exp) / (1 - exp) if exp < 1 else float("nan")


def load(run_dir):
    """Extracted records joined with activations. Returns (df, acts, meta); df row i <-> acts[i]."""
    run = Path(run_dir)
    recs = {(r["unique_id"], r["variant_id"]): r for r in json.load(open(run / "responses.json"))}
    meta = json.load(open(run / "acts" / "meta.json"))
    acts = np.load(run / "acts" / "acts.npy", mmap_mode="r")
    return pd.DataFrame([row(recs[tuple(k)]) for k in meta["keys"]]), acts, meta


def subset(run_dir, n_total=1800, seed=0):
    """All mirage + neutral records, admits randomly filled up to n_total. Writes responses_subset.json."""
    run = Path(run_dir)
    recs = [r for r in json.load(open(run / "responses.json"))
            if r["with_image"]["response"] and r["without_image"]["response"]]
    labs = np.array([label(r) for r in recs])
    keep = np.where(labs != "admit")[0]
    adm = np.where(labs == "admit")[0]
    rng = np.random.default_rng(seed)
    keep = np.sort(np.r_[keep, rng.choice(adm, max(0, min(len(adm), n_total - len(keep))), replace=False)])
    json.dump([recs[i] for i in keep], open(run / "responses_subset.json", "w"))
    return {k: int((labs[keep] == k).sum()) for k in ("mirage", "neutral", "admit")}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def probe_cv(X, y, groups, n_splits=5, seed=0, shuffle_labels=False, C=0.1, center=None):
    """Balanced accuracy of a logistic probe, folds grouped (use leak_groups) so the
    same input never sits on both sides of a split. center = per-row category array:
    subtract each category's mean computed on the fold's train rows (removes what the
    category alone predicts, without leaking test rows into the means)."""
    X0 = np.asarray(X, np.float32)
    y = np.asarray(y).astype(int)
    if shuffle_labels:
        y = np.random.default_rng(seed).permutation(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(X0, y, groups):
        if center is not None:
            fit = np.zeros(len(X0), bool); fit[tr] = True
            X = center_by(X0, center, fit)
        else:
            X = X0
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, class_weight="balanced", max_iter=2000, tol=1e-3))
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        scores.append(np.mean([np.mean(p[y[te] == c] == c) for c in (0, 1) if (y[te] == c).any()]))
    return float(np.mean(scores)), float(np.std(scores))


def _folds(y, groups, n_splits, seed):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups)


def _fold_X(X0, center, tr):
    if center is None:
        return X0
    fit = np.zeros(len(X0), bool); fit[tr] = True
    return center_by(X0, center, fit)


def probe_oof(X, y, groups, n_splits=5, seed=0, C=0.1, center=None):
    """Out-of-fold predictions of the logistic probe (for confusion matrices)."""
    X0 = np.asarray(X, np.float32); y = np.asarray(y)
    pred = np.empty_like(y)
    for tr, te in _folds(y, groups, n_splits, seed):
        X = _fold_X(X0, center, tr)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000, tol=1e-3))
        pred[te] = clf.fit(X[tr], y[tr]).predict(X[te])
    return pred


def one_vector_auc_cv(X, y, groups, n_splits=5, seed=0, center=None):
    """Mean out-of-fold ROC-AUC of a single diff-of-means direction (binary y)."""
    from sklearn.metrics import roc_auc_score
    X0 = np.asarray(X, np.float32); y = np.asarray(y).astype(bool)
    aucs = []
    for tr, te in _folds(y, groups, n_splits, seed):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:     # tiny minority class: fold can't fit or score
            continue
        X = _fold_X(X0, center, tr)
        v = diff_of_means(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], X[te] @ v))
    return float(np.mean(aucs)) if aucs else float("nan")


def type_baseline_cv(cat, y, groups, n_splits=5, seed=0):
    """Balanced accuracy of predicting y from question type alone (train-fold majority per type)."""
    cat = np.asarray(cat); y = np.asarray(y)
    pred = np.empty_like(y)
    for tr, te in _folds(y, groups, n_splits, seed):
        maj = pd.Series(y[tr]).groupby(cat[tr]).agg(lambda s: s.value_counts().index[0])
        fallback = pd.Series(y[tr]).value_counts().index[0]
        pred[te] = [maj.get(c, fallback) for c in cat[te]]
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def probe_split(X, y, train, test, shuffle_labels=False, C=0.1, seed=0):
    """Balanced accuracy of a logistic probe fit on `train` rows, scored on `test` rows."""
    X = np.asarray(X, np.float32)
    y = np.asarray(y).astype(int)
    ytr = np.random.default_rng(seed).permutation(y[train]) if shuffle_labels else y[train]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000, tol=1e-3))
    p = clf.fit(X[train], ytr).predict(X[test])
    return float(np.mean([np.mean(p[y[test] == c] == c) for c in np.unique(y[test])]))


def center_by(X, groups, fit):
    """Subtract each group's mean (computed on `fit` rows only) from its rows.
    Removes what a probe could learn from the group label alone (e.g. question type)."""
    X = np.asarray(X, np.float32).copy()
    groups = np.asarray(groups)
    for g in np.unique(groups):
        s = groups == g
        ref = s & fit
        X[s] -= X[ref].mean(0) if ref.any() else X[s].mean(0)
    return X


def paired_direction(Xa, Xb):
    """Unit mean of per-row differences Xa - Xb (same questions, two conditions)."""
    d = (np.asarray(Xa, np.float32) - np.asarray(Xb, np.float32)).mean(0)
    return d / np.linalg.norm(d)


def perm_cos_null(X, lab, fit, p, a="mirage", b="abstain", cat=None, n=200, seed=0):
    """Null for cos(p, direction a-b): shuffle outcome labels among fit rows, refit the
    (optionally type-centred) diff-of-means, take |cos| with p. Returns (observed cos,
    95th percentile of null |cos|, permutation p-value)."""
    X = np.asarray(X, np.float32)
    if cat is not None:
        X = center_by(X, cat, fit)
    m = fit & np.isin(lab, [a, b])
    Xm, ym = X[m], (np.asarray(lab)[m] == a)
    obs = float(p @ diff_of_means(Xm, ym))
    rng = np.random.default_rng(seed)
    null = np.array([abs(float(p @ diff_of_means(Xm, rng.permutation(ym)))) for _ in range(n)])
    return obs, float(np.quantile(null, .95)), float((1 + (null >= abs(obs)).sum()) / (n + 1))


def diff_of_means(X, y):
    """Unit direction mean(X|y=1) - mean(X|y=0)."""
    X = np.asarray(X, np.float32)
    y = np.asarray(y).astype(bool)
    v = X[y].mean(0) - X[~y].mean(0)
    return v / np.linalg.norm(v)


def best_layer(acts, y, fit, score, cond=1, pool=0):
    """Pick the steering layer without touching test: diff-of-means on `fit` rows,
    ROC-AUC of the projection on `score` rows, per saved layer. Returns (j, aucs)."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for j in range(acts.shape[3]):
        X = np.asarray(acts[:, cond, pool, j], np.float32)
        v = diff_of_means(X[fit], y[fit])
        aucs.append(float(roc_auc_score(y[score], X[score] @ v)))
    return int(np.argmax(aucs)), aucs


def axis_position(X, labels, a="abstain", m="mirage", h="hedged"):
    """Where does the h-mean sit relative to the a->m line?
    t = projection of (mu_h - mu_a) on the a->m direction, as a fraction of |mu_m - mu_a|
        (0 = at abstain, 1 = at mirage);
    off = |component of (mu_h - mu_a) orthogonal to that line| / |mu_m - mu_a|.
    t in (0,1) with small off = one graded axis; large off = hedging is its own direction."""
    X = np.asarray(X, np.float32)
    labels = np.asarray(labels)
    mu = {k: X[labels == k].mean(0) for k in (a, m, h)}
    d = mu[m] - mu[a]
    n = np.linalg.norm(d)
    r = mu[h] - mu[a]
    t = float(r @ d / n**2)
    off = float(np.linalg.norm(r - t * d) / n)
    return t, off


def lda2(Xtr, ytr, Xte):
    """Supervised 2-D view: shrinkage LDA on all dims (no PCA first -- PCA can drop a
    low-variance discriminative direction), fit on train, applied to test. Returns
    (Z_test, fitted pipeline). Shows how classes are arranged, not whether they are
    natural clusters (k-means ARI answers that)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2, solver="eigen", shrinkage="auto"))
    pipe.fit(np.asarray(Xtr, np.float32), ytr)
    return pipe.transform(np.asarray(Xte, np.float32)), pipe


def knn_purity(Z, labels, k=10):
    """Mean share of each point's k nearest neighbours (in Z) that share its label,
    divided by the share expected from label frequencies. 1 = no local structure;
    higher = the embedding keeps same-label points together. Comparable across
    embeddings and against the full-dimensional space."""
    from sklearn.neighbors import NearestNeighbors
    Z = np.asarray(Z, np.float32)
    labels = np.asarray(labels)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    same = (labels[idx] == labels[:, None]).mean(1)
    freq = pd.Series(labels).map(pd.Series(labels).value_counts(normalize=True)).values
    return float(same.mean() / np.mean(freq))


def one_vector_auc(X, y, fit, score):
    """Held-out ROC-AUC of a single diff-of-means direction (fit rows) on score rows."""
    from sklearn.metrics import roc_auc_score
    X = np.asarray(X, np.float32)
    v = diff_of_means(X[fit], y[fit])
    return float(roc_auc_score(y[score], X[score] @ v))


def best_layer_centred(df, acts, a="mirage", b="abstain", cond=1, pool=0):
    """Layer index where a-vs-b survives removing question type: diff-of-means fit on
    type-centred train states, ROC-AUC on type-centred validation states. Raw AUC can
    peak early where question type (which predicts outcome) dominates."""
    from sklearn.metrics import roc_auc_score
    tr, va = (df.split == "train").values, (df.split == "validation").values
    m = df.outcome.isin([a, b]).values
    y = (df.outcome == a).values
    aucs = []
    for j in range(acts.shape[3]):
        Xc = center_by(acts[:, cond, pool, j], df.category.values, tr)
        v = diff_of_means(Xc[tr & m], y[tr & m])
        aucs.append(float(roc_auc_score(y[va & m], Xc[va & m] @ v)))
    return int(np.argmax(aucs)), aucs


def embed(X, method, seed=0):
    X = np.asarray(X, np.float32)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(2, random_state=seed).fit_transform(X)
    if method == "tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        return TSNE(2, random_state=seed, init="pca").fit_transform(PCA(50, random_state=seed).fit_transform(X))
    if method == "phate":
        import phate
        return phate.PHATE(n_components=2, random_state=seed, verbose=0).fit_transform(X)
    raise ValueError(method)
