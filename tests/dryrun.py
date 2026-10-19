"""CPU dry runs on the real SLAKE run, model / Steer / generate stubbed (canned replies), outputs in a scratch
copy of the run dir. Exercises arm building, run_arms, labelling, summaries, row writing and reports; only
model.generate / forward passes are left out. Needs no GPU (and must not touch one: a live chain may hold it).
Usage: python -m tests.dryrun [scratch_dir] [case ...]   cases: dose fitdose transfer introspect sweep model (default all)"""
import contextlib, json, os, shutil, sys, tempfile, types
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from miragelab import introspect as I, transfer as T  # noqa: E402
from miragelab import dose as D  # noqa: E402

RUN = D.A.ROOT / "runs/slake"
REPLIES = ["<think>ok</think> The organ is the liver. [[Liver]]", "<think>ok</think> No image is attached, so I cannot tell. [[Unknown]]",
           "<think>ok</think> It may be the lung, but I cannot see it. [[Lung]]", "<think>ok</think> Yes. [[Yes]]"]
NEW = ["prompt + random (combo norm)", "prompt + activation random (combo norm)",
       "prompt + general + random-for-leftover (combo alphas)"]
steer_layers, seen = [], []


class Fake:
    device = None
    tokenizer = type("T", (), {})()
    config = None
    from_pretrained = classmethod(lambda c, *a, **k: c())
    def eval(self): return self


@contextlib.contextmanager
def fake_steer(model, layer, v, alpha, *mode):
    steer_layers.append(layer); yield


def fake_generate(model, proc, recs, with_image, max_new_tokens, bs, **kw):
    return [REPLIES[(i + len(kw)) % len(REPLIES)] for i in range(len(recs))]


real_run_arms = D.run_arms
def spy_run_arms(model, proc, arms, *a, **k):
    seen.append([(x[0], x[1], float(np.linalg.norm(sum((al * v for v, al, *m in x[4] if v is not None), np.zeros(1))))) for x in arms])
    return real_run_arms(model, proc, arms, *a, **k)


def show(title, arms):
    print(title, *(f"  {n:56s} {f:10s} push {p:6.1f}" for n, (f, p) in arms.items()), sep="\n")


def fake_states(model, proc, recs, layer, bs):
    """The extraction's own states for these texts (so the fit stage's state check passes), else noise."""
    from miragelab import analysis as A
    if not hasattr(fake_states, "S"):
        df, acts, meta = A.load(RUN)
        idx = A.unique_texts(df)[0]
        fake_states.S, fake_states.L = np.asarray(acts[idx, 1, 0], np.float32), meta["layers"]
        fake_states.row = {t: k for k, t in enumerate(df.question.map(A.norm_text).values[idx])}
    ii = [fake_states.row.get(A.norm_text(r["variant_question_text"])) for r in recs]
    X = np.stack([[fake_states.S[i, fake_states.L.index(l)] if i is not None and l in fake_states.L else np.zeros(2560, np.float32)
                   for l in np.atleast_1d(layer)] for i in ii]) + np.random.default_rng(len(recs)).normal(size=(len(recs), np.size(layer), 2560)) * 0.01
    return X if np.ndim(layer) else X[:, 0]


WIDTH = {"Qwen/Qwen3-VL-4B-Thinking": 2560, "Qwen/Qwen3-VL-2B-Instruct": 2048, "nvidia/Cosmos-Reason2-2B": 2048}


def stub():
    D.prompt_states = fake_states
    D.AutoConfig = types.SimpleNamespace(from_pretrained=lambda hf: types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=WIDTH[hf])))
    D.AutoProcessor = D.Qwen3VLForConditionalGeneration = I.AutoProcessor = I.Qwen3VLForConditionalGeneration = Fake
    D.Steer, D.generate, D.run_arms, T.run_arms, T.generate = fake_steer, fake_generate, spy_run_arms, spy_run_arms, fake_generate
    T.load_model = lambda tag: (Fake(), Fake())
    T.states_all = lambda model, proc, recs, bs: np.random.default_rng(len(recs)).normal(size=(len(recs), 28, 2048)).astype(np.float32)


def call(mod, *args):
    seen.clear(); steer_layers.clear()
    sys.argv = ["x", *args]; mod.main()
    return {n: (f, p) for n, f, p in seen[0]} if seen else {}


def dose(R):
    run = lambda *a: call(D, "--run", str(R), *a)
    arms = run("--stage", "test", "--seed", "0")
    show("seed 0 arms:", arms)
    combo = arms["prompt + combo g2 l1"][1]
    assert abs(combo - 86.3) < 1, combo
    assert all(abs(arms[n][1] - combo) < 1e-3 * combo for n in NEW[:2]) and abs(arms[NEW[2]][1] - combo) < 0.25 * combo
    assert {"prompt v2", "prompt v2 + combo (chosen)", *NEW} <= set(arms) and set(steer_layers) == {34}
    assert {r["arm"] for r in json.load(open(R / "dose/test_rows.json"))} >= set(arms)

    run("--stage", "test", "--seed", "0", "--samples", "1")          # chain-folds' last seed-0 stage
    print("sampled picks:", [n for n, _, _ in seen[1]])
    assert [n for n, _, _ in seen[1]] == ["prompt [sampled]", "prompt + combo g2 l1 [sampled]"], seen[1]

    (R / "dose_s1").mkdir(exist_ok=True)       # seed 1's own test stage on its live fit (+ s0's choice if its dose isn't done)
    for f in ("vectors.npz", "gate.json", "dose_choice.json"):
        shutil.copy(RUN / ("dose_s1" if (RUN / "dose_s1" / f).exists() else "dose") / f, R / "dose_s1" / f)
    arms = run("--stage", "test", "--seed", "1")
    L1 = json.load(open(R / "dose_s1/gate.json"))["layer"]
    show(f"seed 1 own (layer {L1}) arms:", arms)
    assert set(steer_layers) == {L1} and {"prompt v2", *NEW} <= set(arms) and (R / "dose_s1/test_rows.json").exists()

    arms = run("--stage", "test", "--seed", "1", "--vectors_from", "0")
    show("xfer s1<-s0 arms:", arms)
    assert set(steer_layers) == {34} and (R / "dose_s1/test_rows_from_s0.json").exists()
    assert all(f in D.GRID or "random" in n or "shuffled" in n for n, (f, p) in arms.items())
    assert set(NEW) <= set(arms) and not any("v2" in n or n in ("base", "prompt") for n in arms)
    run("--stage", "pool", "--seeds", "0,1")


def fitdose(R):
    """A fresh seed's fit -> dose -> test (chain-folds' stages for seeds 2-4 import this file)."""
    run = lambda *a: call(D, "--run", str(R), "--seed", "2", *a)
    run("--stage", "fit", "--perms", "20")
    g = json.load(open(R / "dose_s2/gate.json"))
    arms = run("--stage", "dose")
    print("seed 2 gate:", g, "| dose arms:", list(arms))
    assert len(arms) == 10 and (R / "dose_s2/dose_choice.json").exists() and set(steer_layers) == {g["layer"]}
    arms = run("--stage", "test")
    assert {"base", "prompt", "prompt v2", *NEW} <= set(arms), list(arms)


def transfer(R):
    for m in ("instruct2b", "cosmos2b"):
        call(T, "--run", str(R), "--stage", "gen", "--model", m, "--smoke", "8")
    arms = call(T, "--run", str(R), "--stage", "steer", "--smoke", "8")
    print("transfer steer dose arms:", list(arms), "| layers", set(steer_layers))
    out = R.parent / "transfer_smoke/instruct2b_to_cosmos2b"
    assert all((out / f).exists() for f in ("representation.json", "dose_choice.json", "test_rows.json"))


def introspect(R):
    import re
    from transformers import AutoProcessor
    canned = lambda model, proc, convs, n, bs: ["Hmm [[3]] no, [[7]]" if c[-1]["role"] == "user" and len(c) > 2
                                                 else "[[no]]" for c, _ in convs]
    I.chat, real = canned, I.chat
    (R / "dose_think4b").mkdir(exist_ok=True)          # other-model rows must be ignored
    shutil.copy(R / "dose/test_rows.json", R / "dose_think4b/test_rows.json")
    call(I, "--run", str(R), "--smoke", "6")
    rows, rep = json.load(open(R / "introspect_smoke/rows.json")), json.load(open(R / "introspect_smoke/report.json"))
    assert all(r["seen"] == 7 and r["conf"] == 7 for r in rows) and {r["src"] for r in rows} <= {"dose", "dose_s1"}
    assert rep["check"]["with image"]["no"] == 6, rep["check"]
    I.chat = real
    # the real chat template must take the follow-up conversation (assistant turn + image placeholder): CPU only
    proc = AutoProcessor.from_pretrained(D.MODEL)
    r = next(x for x in json.load(open(RUN / "responses.json")) if x["image_path"])
    m, p = I.conv(r, True, [{"role": "assistant", "content": [{"type": "text", "text": "It is the liver. [[Liver]]"}]},
                            {"role": "user", "content": [{"type": "text", "text": I.ASK["seen"]}]}])
    t = proc.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    assert t.count("<|vision_start|>") == 1 and "[[Liver]]" in t and t.rstrip().endswith("assistant") and p, t
    print("introspect template OK:", re.sub(r"\s+", " ", t[-200:]))


def sweep(R):
    run = lambda *a: call(D, "--run", str(R), "--stage", "sweep", *a)
    run("--smoke", "4")
    t = json.load(open(R / "dose_sweep_smoke/dose_sweep.json"))
    print("sweep smoke arms:", list(t["arms"]))
    run()
    t = json.load(open(R / "dose/dose_sweep.json"))
    A, nrm = t["arms"], t["state_norm"]
    print(f"sweep: {len(A)} arms, state norm {nrm:.1f}, combo push {t['combo_push']:.1f}")
    for n, x in A.items():
        if "push" in x and not n.startswith(("swap", "general + random")):
            assert abs(x["push"] - x["share"] / 100 * nrm) < 1e-3 * nrm, (n, x["push"])
    assert abs(A["general 10% v1"]["along"] - 0.1 * nrm) < 1e-2 and A["general 10% v1"]["off"] < 1e-2 * nrm
    assert len(A["combo 20% v1"]["random_mirage"]) == 6 and "beats_random" not in A["combo 20% v2"]
    assert {"rotate template 20%", "scale grow 10%", "swap: general at combo size"} <= set(A) and "acc" in A["leftover 5% v2"]
    assert set(steer_layers) == {34} and abs(A["swap: general at combo size"]["push"] - t["combo_push"]) < 1e-3
    assert len(A) == 2 + 4 * 5 * 2 + 5 * 4 * 2 + 2 + 6 + 4, len(A)


def model(R):
    run = lambda *a: call(D, "--run", str(R), "--stage", "test", *a)
    arms = run("--model", "think4b", "--smoke", "4")
    show("think4b arms (smoke):", arms)
    assert {"base", "prompt", "prompt v2", "prompt v2 + combo (chosen)", *NEW} <= set(arms) and not any("template" in n or "REVERSED" in n for n in arms)
    rows = json.load(open(R / "dose_think4b_smoke/test_rows.json"))
    assert {r["outcome"] for r in rows} <= {"mirage", "abstain", "hedged", "truncated"} and set(steer_layers) == {34}
    for m in ("instruct2b", "cosmos2b"):
        arms = run("--model", m, "--prompts_only", "--smoke", "4")
        assert list(arms) == ["base", "prompt", "prompt v2"] and (R / f"dose_{m}_smoke/test_rows.json").exists(), arms
    try:
        run("--model", "instruct2b", "--smoke", "4")
        raise RuntimeError("width mismatch not caught")
    except AssertionError as e:
        print("width check OK:", e)


def main(scratch=None, cases=()):
    R = Path(scratch or tempfile.mkdtemp()) / "slake"
    for d in (R, R.parent / "transfer_smoke"):
        shutil.rmtree(d, ignore_errors=True)
    R.mkdir(parents=True)
    for f in ("acts", "acts_neutral.npy", "responses.json"):
        (R / f).symlink_to(RUN / f)
    shutil.copytree(RUN / "dose", R / "dose")
    stub()
    for c in cases or ("dose", "fitdose", "transfer", "introspect", "sweep", "model"):
        print(f"\n=== {c}", flush=True)
        globals()[c](R)
    print("DRY RUN OK", R)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2:])
