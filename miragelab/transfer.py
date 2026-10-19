"""Does a mirage vector carry over to a sibling model? Qwen3-VL-2B-Instruct -> Cosmos-Reason2-2B
(NVIDIA's reasoning post-train of that same model: same architecture, d=2048, 28 layers).

Texts, split and records are the 4B SLAKE run's (dose.split, stratified by the 4B labels), so
fit / dose / test are the same question texts as dose.py. No honesty prompt here: the vector alone.

Stages (under runs/transfer/<model>/):
  gen    --model M     no-image greedy replies on all distinct texts + last-prompt states at every
                       layer -> rows.json, states.npy [text, layer, d] fp16
  steer  --source S --target T
         layer: best type-centred CV AUC (mirage vs abstain) on SOURCE fit texts.
         Representation (no generation): on TARGET test texts, AUC of the projection on the source
         vector, the target's own vector (fit texts), and 20 random directions; cos(source, own).
         Behaviour: alpha unit = the target's own fit gap, so own and transferred arms push equally.
         dose texts: own / transferred x1-3 -> dose.choose (with-image cost + looping rule);
         test texts: base, own*, transferred*, random and transferred-REVERSED at the transferred alpha.
Cosmos runs in reasoning mode (its model-card system-prompt suffix); outcome is read from the text
after </think> only, `think_admits` records whether the reasoning noticed the missing image, and a
reply with no </think> within max_new_tokens is `truncated`.
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from . import analysis as A
from .dose import GRID, choose, label, looping, run_arms, split, summarise
from .extract import messages
from .steer import direction, generate

THINK = ("\n\nAnswer the question in the following format: <think>\nyour reasoning\n</think>\n\n"
         "<answer>\nyour answer\n</answer>.")
MODELS = {"instruct2b": dict(hf="Qwen/Qwen3-VL-2B-Instruct", think=False, max_new_tokens=384),
          "cosmos2b": dict(hf="nvidia/Cosmos-Reason2-2B", think=True, max_new_tokens=1536),
          "think4b": dict(hf="Qwen/Qwen3-VL-4B-Thinking", think=True, max_new_tokens=1536, fmt="")}   # template opens <think> itself


def split_think(resp):
    """(reasoning, answer text). No </think> -> (resp, None): truncated mid-reasoning."""
    if "</think>" not in resp:
        return resp, None
    th, ans = resp.split("</think>", 1)
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", ans, re.S)
    return th, (m.group(1) if m else ans).strip()


def think_label(resp):
    th, ans = split_think(resp)
    return (None, "truncated") if ans is None else label(ans)


def prep_for(tag):
    sfx = MODELS[tag].get("fmt", THINK) if MODELS[tag]["think"] else ""
    return (lambda r: dict(r, system_prompt=r["system_prompt"] + sfx)) if sfx else (lambda r: r)


@torch.no_grad()
def states_all(model, proc, recs, bs):
    """Last-prompt residual at every layer, no image, left-padded batches -> [n, layers, d]."""
    out = []
    for i in range(0, len(recs), bs):
        texts = [proc.apply_chat_template(messages(r, []), add_generation_prompt=True, tokenize=False) for r in recs[i:i + bs]]
        hs = model(**proc(text=texts, return_tensors="pt", padding=True).to(model.device), output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1] for h in hs[1:]], 1).float().cpu().numpy())
    return np.concatenate(out)


def load_model(tag):
    proc = AutoProcessor.from_pretrained(MODELS[tag]["hf"])
    proc.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODELS[tag]["hf"], dtype=torch.bfloat16, device_map="cuda").eval()
    return model, proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/slake", help="the 4B SLAKE run: records, texts, split")
    ap.add_argument("--stage", choices=["gen", "steer"], required=True)
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--source", choices=list(MODELS), default="instruct2b")
    ap.add_argument("--target", choices=list(MODELS), default="cosmos2b")
    ap.add_argument("--dose_image", type=int, default=60)
    ap.add_argument("--test_image", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0, help="only this many texts / image records")
    a = ap.parse_args()
    root = Path(a.run).parent / ("transfer_smoke" if a.smoke else "transfer")

    df, _, meta = A.load(a.run)
    idx, lab4, _, _ = A.unique_texts(df)
    texts = df.question.map(A.norm_text).values[idx]
    cat = df.category.values[idx]
    fit, dose, test = split(texts, lab4)
    allrecs = {(r["unique_id"], r["variant_id"]): r for r in json.load(open(Path(a.run) / "responses.json"))}
    recs = [allrecs[tuple(meta["keys"][i])] for i in idx]
    keep = np.arange(len(idx))
    if a.smoke:      # a few texts from each part, half 4B mirages, so every stage has something to fit, tune and test
        keep = np.concatenate([np.r_[np.where(m & (lab4 == "mirage"))[0][: a.smoke // 2],
                                     np.where(m & (lab4 != "mirage"))[0][: a.smoke - a.smoke // 2]] for m in (fit, dose, test)])

    if a.stage == "gen":
        out = root / a.model
        out.mkdir(parents=True, exist_ok=True)
        model, proc = load_model(a.model)
        rs = [prep_for(a.model)(recs[i]) for i in keep]
        S = states_all(model, proc, rs, a.bs)
        resps = generate(model, proc, rs, False, MODELS[a.model]["max_new_tokens"], a.bs)
        lab_fn = think_label if MODELS[a.model]["think"] else label
        rows = []
        for i, r, x in zip(keep, rs, resps):
            pred, o = lab_fn(x)
            th, _ = split_think(x) if MODELS[a.model]["think"] else ("", None)
            rows.append(dict(i=int(i), text=texts[i], category=cat[i], part="fit" if fit[i] else "dose" if dose[i] else "test",
                             response=x, pred=pred, outcome=o, think_admits=A.admits(th) if th else None, loop=looping(x)))
        np.save(out / "states.npy", S.astype(np.float16))
        json.dump(rows, open(out / "rows.json", "w"))
        c = {o: sum(r["outcome"] == o for r in rows) for o in ("abstain", "hedged", "mirage", "truncated")}
        print(f"{a.model}: {len(rows)} texts {c} | loops {sum(r['loop'] for r in rows)}"
              + (f" | think admits the image is missing: {sum(bool(r['think_admits']) for r in rows)}" if MODELS[a.model]["think"] else ""), flush=True)
        return

    # steer: states and labels of both models, over the kept texts in the same order
    def side(tag):
        rows = json.load(open(root / tag / "rows.json"))
        assert [r["i"] for r in rows] == list(keep), f"{tag}: gen ran on different texts"
        return np.load(root / tag / "states.npy").astype(np.float32), np.array([r["outcome"] for r in rows])
    Ss, ls = side(a.source)
    St, lt = side(a.target)
    f, d, t, c = fit[keep], dose[keep], test[keep], cat[keep]
    m = f & np.isin(ls, ["mirage", "abstain"])
    y = ls[m] == "mirage"
    k = min(5, int(y.sum()), int((~y).sum()))           # folds: every train fold must hold both classes (as dose.py)
    aucs = [A.one_vector_auc_cv(Ss[m, j], y, np.arange(m.sum()), n_splits=k, center=c[m]) for j in range(Ss.shape[1])] if k >= 2 else [np.nan]
    if not np.isfinite(aucs).any():
        msg = f"too few mirage/abstain texts on {a.source} fit texts to pick a layer (mirage {y.sum()}, abstain {(~y).sum()})"
        if not a.smoke:
            raise SystemExit(msg)
        print(f"smoke: {msg}; middle layer stands in to exercise the code", flush=True)
        aucs = [np.nan] * Ss.shape[1]; aucs[Ss.shape[1] // 2] = 0.5
    j = int(np.nanargmax(aucs))
    a.layer = j + 1                          # states[:, j] = output of decoder layer j+1 = Steer(layer=j+1)
    vs, _ = direction(Ss[:, j], ls, f, "mirage")
    vt, gt = direction(St[:, j], lt, f, "mirage")
    rng = np.random.default_rng(0)
    rand = [v / np.linalg.norm(v) for v in rng.normal(size=(20, vs.size))]

    from sklearn.metrics import roc_auc_score
    mt = t & np.isin(lt, ["mirage", "abstain"])
    yt = lt[mt] == "mirage"
    proj_auc = lambda v: float(roc_auc_score(yt, St[mt, j] @ v)) if 0 < yt.sum() < len(yt) else float("nan")
    rep = dict(layer=a.layer, source_fit_cv_auc=float(aucs[j]), cos_source_own=float(vs @ vt), gap_own=gt,
               test_auc_transferred=proj_auc(vs), test_auc_own=proj_auc(vt),
               test_auc_random=[proj_auc(v) for v in rand], n_test=dict(mirage=int(yt.sum()), abstain=int((~yt).sum())),
               target_outcomes={o: int((lt == o).sum()) for o in ("abstain", "hedged", "mirage", "truncated")})
    out = root / f"{a.source}_to_{a.target}"
    out.mkdir(exist_ok=True)
    json.dump(rep, open(out / "representation.json", "w"), indent=1)
    print(f"layer {a.layer} | cos(source, own) {vs @ vt:.2f} | target test AUC: transferred {rep['test_auc_transferred']:.3f} "
          f"own {rep['test_auc_own']:.3f} random {np.nanmean(rep['test_auc_random']):.3f}±{np.nanstd(rep['test_auc_random']):.3f} "
          f"(mirage {yt.sum()} / abstain {(~yt).sum()})", flush=True)

    model, proc = load_model(a.target)
    a.max_new_tokens = MODELS[a.target]["max_new_tokens"]
    kw = dict(label=think_label if MODELS[a.target]["think"] else label, prep=prep_for(a.target))
    img_rows = lambda mask: [allrecs[tuple(meta["keys"][r])] for r in np.random.default_rng(0).permutation(
        np.where(df.question.map(A.norm_text).isin(set(texts[keep][mask])).values)[0])]
    arm = lambda fam, v, x: (f"{fam} x{x}", fam, x, False, [(v, x * gt)])
    no_img = lambda mask: [recs[i] for i in keep[mask]]

    dose_arms = [("base", "prompt", 0, False, [])] + [arm(fam, v, x) for fam, v in (("own", vt), ("transferred", vs)) for x in (1, 2, 3)]
    rows = run_arms(model, proc, dose_arms, no_img(d), img_rows(d)[: a.smoke or a.dose_image], out / "dose_rows.json", a, **kw)
    summ = summarise(rows)
    summ["prompt"] = summ["base"]        # dose.choose compares against the 'prompt' arm; here that is base
    chosen = {fam: choose(summ, fam) for fam in ("own", "transferred")}
    json.dump(dict(summary=summ, chosen=chosen), open(out / "dose_choice.json", "w"), indent=1)
    print("CHOSEN:", chosen, flush=True)

    xs = {fam: (summ[chosen[fam]]["dose"] if chosen[fam] else 1) for fam in chosen}
    test_arms = [("base", "base", 0, False, []), arm("own", vt, xs["own"]), arm("transferred", vs, xs["transferred"]),
                 (f"random x{xs['transferred']}", "control", 0, False, [(rand[0], xs["transferred"] * gt)]),
                 (f"transferred REVERSED x{xs['transferred']}", "control", 0, False, [(vs, -xs["transferred"] * gt)])]
    run_arms(model, proc, test_arms, no_img(t), img_rows(t)[: a.smoke or a.test_image], out / "test_rows.json", a, **kw)


if __name__ == "__main__":
    main()
