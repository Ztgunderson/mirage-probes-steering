import numpy as np
import pytest

from miragelab import analysis as A


def rec(no_resp, similar=True, missing=False):
    return {"with_image": {"response_missing": missing},
            "without_image": {"response": no_resp, "response_missing": False,
                              "acknowledged_missing_or_uncertain": A.gen._detect_acknowledgement(no_resp) != (False, False),
                              "response_similarity_with_image": similar}}


@pytest.mark.parametrize("text", [
    "Since I cannot view the actual image, I cannot confirm... [[Yes]]",
    "However, the image provided is not visible here, so [[Unknown]]",
])
def test_fixed_label_catches_paper_misses(text):
    r = rec(text)
    assert A.label(r, fixed=False) == "mirage"      # the paper's regex misses it
    assert A.label(r, fixed=True) == "admit"


@pytest.mark.parametrize("text", [
    "The heart is enlarged with a widened mediastinum. [[Yes]]",
    "Consolidation is visible in the left lower lobe. [[Yes]]",
])
def test_confident_findings_stay_mirage(text):
    assert A.label(rec(text), fixed=True) == "mirage"


def test_shifted_unacknowledged_is_neutral():
    assert A.label(rec("Probably a CT scan. [[No]]", similar=False)) == "neutral"


def test_wilson_known_value():
    p, lo, hi = A.wilson(8, 10)
    assert p == 0.8 and abs(lo - 0.490) < 0.005 and abs(hi - 0.943) < 0.005


def test_probe_separable_vs_shuffled():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 100)
    X = rng.normal(size=(200, 20)) + y[:, None] * 1.5
    g = np.arange(200) // 2
    assert A.probe_cv(X, y, g)[0] > 0.9
    assert abs(A.probe_cv(X, y, g, shuffle_labels=True)[0] - 0.5) < 0.12


def test_folds_never_split_a_group():
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.tile([0, 1], 50)
    g = np.arange(100) // 5
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(100), y, g):
        assert not set(g[tr]) & set(g[te])


def test_diff_of_means_direction():
    X = np.array([[0, 0], [0, 0], [2, 0], [2, 0]], float)
    assert np.allclose(A.diff_of_means(X, [0, 0, 1, 1]), [1, 0])


def test_subset_keeps_all_rare_labels(tmp_path):
    import json
    def r(uid, no, sim):
        return {"unique_id": uid, "variant_id": "original", "variant_question_text": "q",
                "ground_truth_normalized": "yes", "category": "",
                "with_image": {"response": "x [[Yes]]", "response_missing": False, "correct": True},
                "without_image": {"response": no, "response_missing": False, "correct": True,
                                  "acknowledged_missing_or_uncertain": False,
                                  "response_similarity_with_image": sim}}
    recs = [r(f"m{i}", "It is enlarged. [[Yes]]", True) for i in range(3)] + \
           [r(f"a{i}", "No image was provided. [[Yes]]", True) for i in range(20)]
    json.dump(recs, open(tmp_path / "responses.json", "w"))
    assert A.subset(tmp_path, n_total=8) == {"mirage": 3, "neutral": 0, "admit": 5}
    assert len(A.records(tmp_path)) == 23


def test_answer_probs_collapses_token_variants():
    import math
    from miragelab.prior import answer_probs, choices
    top = [{"token": "Yes", "logprob": math.log(.5)}, {"token": " yes", "logprob": math.log(.1)},
           {"token": "No", "logprob": math.log(.2)}, {"token": "Maybe", "logprob": math.log(.2)}]
    p, cov = answer_probs(top, ["YES", "NO"])
    assert abs(cov - .8) < 1e-9 and abs(p["YES"] - .75) < 1e-9
    top = [{"token": "B", "logprob": math.log(.6)}, {"token": " C.", "logprob": math.log(.3)}]
    p, _ = answer_probs(top, ["A", "B", "C", "D", "E"])
    assert abs(p["B"] - 2 / 3) < 1e-9 and p["A"] == 0
    assert choices({"task_type": "mcq", "options": "A. x\nB. y\nC. z"}) == ["A", "B", "C"]
    assert choices({"task_type": "open", "options": ""}) == ["YES", "NO"]


def test_agreement_chance_and_perfect():
    obs, exp, k = A.agreement(["yes", "no"] * 50, ["yes", "no"] * 50)
    assert obs == 1 and abs(exp - .5) < 1e-9 and abs(k - 1) < 1e-9
    obs, exp, k = A.agreement(["yes"] * 50 + ["no"] * 50, ["yes", "no"] * 50)
    assert abs(obs - .5) < 1e-9 and abs(k) < 1e-9
    assert A.agreement(["A", None], ["A", "B"])[0] == 1


@pytest.mark.parametrize("pred,gt,ok", [
    ("Left Lung, Right", "Left Lung, Right", True),
    ("the right lung", "Right Lung", True),
    ("lungs", "Lung", True),
    ("Liver", "Lung", False),
    ("CT", "MRI", False),
    (None, "CT", None),
])
def test_slake_open_scoring(pred, gt, ok):
    from miragelab.slake import correct
    assert correct(pred, gt) == ok


@pytest.mark.parametrize("resp,pred,out", [
    ("I cannot see the image. [[Unknown]]", "unknown", "abstain"),
    ("Without a radiology image it is impossible. [[Cannot determine]]", "cannot determine", "abstain"),
    ("No image is provided, so I assume abdominal CT. [[liver]]", "liver", "hedged"),
    ("The liver is the largest organ seen here. [[liver]]", "liver", "mirage"),
    ("", None, "abstain"),
])
def test_open_outcome(resp, pred, out):
    r = {"without_image": {"response": resp, "pred_normalized": pred}}
    assert A.outcome(r) == out


def test_steer_hook_subtracts_and_removes():
    import torch
    from types import SimpleNamespace
    from miragelab.steer import Steer
    layer = torch.nn.Identity()
    m = SimpleNamespace(device=torch.device("cpu"), model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer])))
    x = torch.zeros(1, 3, 2, dtype=torch.bfloat16)
    with Steer(m, 1, np.array([1.0, 0.0]), alpha=2.0):
        y = layer(x)
    assert torch.allclose(y[0, :, 0].float(), torch.full((3,), -2.0)) and torch.all(y[0, :, 1] == 0)
    assert torch.all(layer(x) == 0)                     # hook removed on exit
    with Steer(m, 1, np.array([1.0, 0.0]), alpha=0.0):  # alpha 0 = no hook at all
        assert torch.all(layer(x) == 0)


def test_steer_rotate_keeps_norm_scale_keeps_direction():
    import torch
    from types import SimpleNamespace
    from miragelab.steer import Steer
    layer = torch.nn.Identity()
    m = SimpleNamespace(device=torch.device("cpu"), model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer])))
    x = torch.tensor([[[3.0, 4.0]]], dtype=torch.bfloat16)                # norm 5
    with Steer(m, 1, np.array([1.0, 0.0]), 3.0, "rotate"):
        y = layer(x).float()[0, 0]
    assert abs(y.norm() - 5) < 0.05 and abs(y[0]) < 0.05 and y[1] > 4.9   # (0, 4) rescaled to norm 5
    with Steer(m, 1, None, 1.0, "scale"):
        y = layer(x).float()[0, 0]
    assert torch.allclose(y, torch.tensor([2.4, 3.2]), atol=0.03)         # |dh| = 1 along h


def test_best_layer_finds_informative_layer():
    rng = np.random.default_rng(0)
    y = np.repeat([False, True], 50)
    acts = rng.normal(size=(100, 2, 2, 3, 8)).astype("f4")
    acts[y, 1, 0, 2, 0] += 3                          # only layer index 2 separates
    fit = np.arange(100) % 2 == 0
    j, aucs = A.best_layer(acts, y, fit, ~fit)
    assert j == 2 and aucs[2] > .95


def test_probe_split_separable_vs_shuffled():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 100)
    X = rng.normal(size=(200, 10)) + y[:, None] * 1.5
    tr = np.arange(200) % 2 == 0
    assert A.probe_split(X, y, tr, ~tr) > .9
    assert abs(A.probe_split(X, y, tr, ~tr, shuffle_labels=True) - .5) < .15


def test_center_by_removes_group_offsets_using_fit_rows():
    X = np.array([[10.], [12.], [0.], [2.], [100.]])
    g = np.array(["a", "a", "b", "b", "a"])
    fit = np.array([True, True, True, True, False])
    Xc = A.center_by(X, g, fit)
    assert np.allclose(Xc[:4, 0], [-1, 1, -1, 1]) and Xc[4, 0] == 89   # held-out row uses fit mean


def test_direction_points_from_source_to_abstain_on_fit_only():
    from miragelab.steer import direction
    lab = np.array(["mirage", "abstain", "hedged"] * 4 + ["mirage", "abstain"])
    fit = np.array([True] * 12 + [False] * 2)
    X = np.zeros((14, 3), "f4")
    X[lab == "mirage", 0] = 2.0
    X[lab == "hedged", 1] = 2.0
    X[12, 2] = 50.0                                   # held-out row must not leak into the fit
    v, gap = direction(X, lab, fit, "mirage")
    assert np.allclose(v, [1, 0, 0]) and abs(gap - 2) < 1e-6
    vh, _ = direction(X, lab, fit, "hedged")
    assert np.allclose(vh, [0, 1, 0])

def test_axis_position_on_and_off_line():
    lab = np.array(["abstain"] * 2 + ["mirage"] * 2 + ["hedged"] * 2)
    X = np.array([[0, 0], [0, 0], [2, 0], [2, 0], [1, 0], [1, 0]], float)
    t, off = A.axis_position(X, lab)
    assert abs(t - .5) < 1e-6 and off < 1e-6
    X[4:] = [[0, 2], [0, 2]]                          # hedged off to the side
    t, off = A.axis_position(X, lab)
    assert abs(t) < 1e-6 and abs(off - 1) < 1e-6


def test_probe_split_multiclass():
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1, 2], 60)
    X = rng.normal(size=(180, 6)); X[np.arange(180), y] += 3
    tr = np.arange(180) % 2 == 0
    assert A.probe_split(X, y, tr, ~tr) > .9


def test_lda2_separates_three_classes_held_out():
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1, 2], 80)
    X = rng.normal(size=(240, 120)); X[:, 0] += y * 4
    tr = np.arange(240) % 2 == 0
    Z, pipe = A.lda2(X[tr], y[tr], X[~tr])
    assert Z.shape == ((~tr).sum(), 2) and (pipe.predict(X[~tr]) == y[~tr]).mean() > .9


def test_knn_purity_structure_vs_noise():
    rng = np.random.default_rng(4)
    lab = np.repeat(["a", "b"], 100)
    Z = rng.normal(size=(200, 2)); Z[lab == "b"] += 10
    assert A.knn_purity(Z, lab) > 1.9                   # perfectly grouped -> 1/0.5
    assert abs(A.knn_purity(rng.normal(size=(200, 2)), lab) - 1) < .15


def test_one_vector_auc():
    rng = np.random.default_rng(5)
    y = np.repeat([False, True], 100)
    X = rng.normal(size=(200, 5)); X[y, 0] += 3
    fit = np.arange(200) % 2 == 0
    assert A.one_vector_auc(X, y, fit, ~fit) > .95


def test_neutral_prompt_removes_only_pressure():
    from miragelab.pressure import neutral
    from miragelab.slake import SYSTEM
    n = neutral(SYSTEM)
    assert "MUST always provide a definitive answer" not in n and "best educated guess" not in n
    assert "Base your answer on the visual evidence" in n and "[[your answer]]" in n
    assert len(SYSTEM) - len(n) < 200


def test_paired_direction():
    rng = np.random.default_rng(6)
    base = rng.normal(size=(50, 4))
    assert np.allclose(A.paired_direction(base + [0, 3, 0, 0], base), [0, 1, 0, 0])


def test_best_layer_centred_ignores_type_driven_layer():
    import pandas as pd
    rng = np.random.default_rng(7)
    n = 400
    cat = np.where(np.arange(n) % 2 == 0, "color", "abnorm")
    out = np.where(cat == "color", np.where(rng.random(n) < .8, "mirage", "abstain"),
                   np.where(rng.random(n) < .2, "mirage", "abstain"))
    split = np.where(np.arange(n) % 4 == 0, "validation", "train")
    acts = rng.normal(size=(n, 2, 1, 2, 6)).astype("f4")
    acts[cat == "color", 1, 0, 0, 0] += 5              # layer 0: encodes question type only
    acts[out == "mirage", 1, 0, 1, 1] += 1.5           # layer 1: encodes the behaviour
    df = pd.DataFrame(dict(outcome=out, split=split, category=cat))
    j, aucs = A.best_layer_centred(df, acts)
    assert j == 1 and aucs[1] > .8


def test_leak_groups_chains_shared_keys():
    uid = ["a", "a", "b", "c", "d"]
    txt = ["x", "y", "y", "z", "z"]
    g = A.leak_groups(uid, txt)
    assert g[0] == g[1] == g[2] and g[3] == g[4] and g[0] != g[3]


def test_unique_texts_majority_and_stability():
    import pandas as pd
    df = pd.DataFrame(dict(question=["What?", "what? ", "Where?", "Where?", "Where?"],
                           outcome=["mirage", "mirage", "abstain", "hedged", "abstain"]))
    idx, lab, n, stable = A.unique_texts(df)
    assert list(idx) == [0, 2] and list(lab) == ["mirage", "abstain"] and list(n) == [2, 3] and list(stable) == [True, False]


def test_text_split_stratified_and_deterministic():
    strat = np.repeat(["a", "b"], [10, 20])
    f1 = A.text_split(np.arange(30), .7, 0, strat); f2 = A.text_split(np.arange(30), .7, 0, strat)
    assert (f1 == f2).all() and f1[strat == "a"].sum() == 7 and f1[strat == "b"].sum() == 14


def test_probe_cv_centering_uses_train_rows_only():
    rng = np.random.default_rng(8)
    cat = np.repeat(["p", "q"], 100)
    y = (cat == "p").astype(int)                      # label fully explained by category
    X = rng.normal(size=(200, 5)); X[cat == "p", 0] += 4
    g = np.arange(200)
    assert A.probe_cv(X, y, g)[0] > .95
    assert A.probe_cv(X, y, g, center=cat)[0] < .7   # nothing left once category is removed


def test_cv_helpers_on_type_confounded_data():
    rng = np.random.default_rng(9)
    n = 300
    cat = rng.choice(["a", "b", "c"], n)
    y = np.where(cat == "a", "m", np.where(rng.random(n) < .5, "m", "x"))
    X = rng.normal(size=(n, 6)); X[cat == "a", 0] += 4          # type signal
    X[y == "m", 1] += 2                                          # behaviour signal
    g = np.arange(n)
    assert A.one_vector_auc_cv(X, y == "m", g) > .8
    assert A.one_vector_auc_cv(X, y == "m", g, center=cat) > .75   # behaviour survives centring
    assert A.probe_oof(X, y, g).shape == (n,)
    c2 = np.repeat(["a", "b"], 50); y2 = np.where(c2 == "a", "m", "x")
    assert A.type_baseline_cv(c2, y2, np.arange(100)) == 1.0     # type fully determines y
    assert abs(A.type_baseline_cv(c2, rng.permutation(y2), np.arange(100)) - .5) < .15


def test_perm_cos_null_detects_real_alignment_only():
    rng = np.random.default_rng(10)
    lab = np.repeat(["mirage", "abstain"], 100)
    X = rng.normal(size=(200, 50)); X[lab == "mirage", 0] += 1.0
    fit = np.ones(200, bool)
    p = np.zeros(50); p[0] = 1
    obs, q95, pv = A.perm_cos_null(X, lab, fit, p, n=100)
    assert obs > q95 and pv < .05
    q = np.zeros(50); q[7] = 1                         # unrelated direction
    obs, q95, pv = A.perm_cos_null(X, lab, fit, q, n=100)
    assert pv > .05


def test_judge_rules_sample_and_kappa():
    from miragelab import judge as J
    assert J.from_rules(False, True, True) == "abstain"
    assert J.from_rules(True, True, True) == "hedged"          # admission outranks findings
    assert J.from_rules(True, False, True) == "hard_mirage"
    assert J.from_rules(True, False, False) == "soft_mirage"
    recs = [{"unique_id": i, "variant_question_text": f"q{i % 5}", "without_image": {"response": "r"},
             "o": "mirage" if i % 2 else "abstain"} for i in range(40)]
    s = J.sample(recs, lambda r: r["o"], n_per=10)
    assert len(s) == 10 and len({(x["regex"], x["question"]) for x in s}) == 10   # 5 texts x 2 strata, one each
    assert s == J.sample(recs, lambda r: r["o"], n_per=10)
    assert J.kappa(list("aabb"), list("aabb")) == 1.0 and abs(J.kappa(list("abab"), list("aabb"))) < 1e-9
    c = J.compare({"x": "a", "y": "b"}, {"x": "a", "y": "a", "z": "b"})
    assert c["n"] == 2 and c["agree"] == 0.5


def test_dose_split_disjoint_and_sized():
    from miragelab.dose import split
    texts = np.array([f"q{i}" for i in range(300)])
    lab = np.array(["mirage", "abstain", "hedged"] * 100)
    fit, dose, test = split(texts, lab)
    assert (fit.astype(int) + dose + test == 1).all()           # every text in exactly one part
    assert abs(test.mean() - 0.4) < 0.02 and abs(fit.mean() - 0.4) < 0.02 and abs(dose.mean() - 0.2) < 0.02


def test_looping_flags_degenerate_only():
    from miragelab.dose import looping
    assert looping("to " * 40)
    assert looping("the right kidney is left of the liver — no. " * 4)
    assert not looping("The liver is in the right upper quadrant. [[liver]]")


def test_dose_rule():
    from miragelab.dose import choose
    s = lambda fam, dose, mir, ab, acc, loop=0.0: dict(family=fam, dose=dose, mirage=mir, abstain=ab, acc=acc, loop=loop)
    summ = {"prompt": s("prompt", 0, .09, .87, .45),
            "g1": s("general", 1, .05, .80, .44), "g2": s("general", 2, .02, .70, .35),   # g2 costs 10 pts: out
            "g3": s("general", 3, .01, .60, .45, loop=.05),                               # loops: out
            "l1": s("leftover", 1, .04, .85, .45), "l2": s("leftover", 2, .04, .90, .45)}  # tie -> more abstain
    assert choose(summ, "general") == "g1"
    assert choose(summ, "leftover") == "l2"
    assert choose(summ, "combo") is None


def test_one_vector_auc_cv_survives_tiny_minority():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(11, 8)); y = np.array([1, 1] + [0] * 9)
    auc = A.one_vector_auc_cv(X, y, np.arange(11), n_splits=2)
    assert np.isnan(auc) or 0 <= auc <= 1


@pytest.mark.parametrize("resp,outcome", [
    ("<think>\nNo image is attached, so I cannot see it.\n</think>\n\n<answer>\n[[Unknown]]\n</answer>", "abstain"),
    ("<think>\nThe liver is usually upper right.\n</think>\n\n<answer>\nliver\n</answer>", "mirage"),
    ("<think>\nThe image is not provided.\n</think>\n<answer>No image is provided, but it is probably the liver. [[liver]]</answer>", "hedged"),
    ("<think>\nLet me look at the image carefully and", "truncated"),
])
def test_think_label_reads_answer_part_only(resp, outcome):
    from miragelab.transfer import think_label
    assert think_label(resp)[1] == outcome


def test_split_think_keeps_reasoning():
    from miragelab.transfer import split_think
    th, ans = split_think("<think>no image here</think><answer>[[Unknown]]</answer>")
    assert "no image" in th and ans == "[[Unknown]]"


def test_template_direction_ignores_mirage_labels():
    from miragelab.dose import TEMPLATE, template_direction
    assert TEMPLATE.search("what organ is the gray part on the left of the image?")
    assert not TEMPLATE.search("what is the largest organ in the picture?")
    rng = np.random.default_rng(0)
    texts = np.array(["what organ is the gray part on the top of the image?"] * 5 + ["where is the liver?"] * 15)
    cat = np.array(["Organ"] * 20)
    X = rng.normal(size=(20, 16)); X[:5, 0] += 5                 # template rows differ on axis 0
    v = template_direction(X, texts, cat, np.ones(20, bool))
    assert abs(np.linalg.norm(v) - 1) < 1e-6 and abs(v[0]) > 0.8
