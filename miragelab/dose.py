"""Honesty prompt + steering against the mirages the prompt leaves, with a
train / dose / test split over distinct question texts.

Split (stratified by base outcome, --seed, default 0): train 60% = vector-fit 40% + dose 20%;
test 40%, not generated on until the test stage. One layer, picked by type-centred
CV AUC (mirage vs abstain) on vector-fit texts.

Vectors (h <- h - alpha * v_hat, alpha in units of the fit projection gap, as steer.py):
  general   mirage - abstain on BASE-prompt states (steer.py's vector, refit on this split)
  leftover  mirage - abstain on HONESTY-prompt states, type-centred: what the prompt misses
Controls, all at the chosen leftover alpha: shuffled-label leftover (labels permuted within
question type: same states, no label), random unit vector, leftover reversed (should ADD
mirages), plus base (no prompt, no vector). Size-matched to the chosen combo's push: random unit
vector, activation-spread random (fit states' spread), real general + random in place of leftover.

Stages (cached under <run>/dose/, each resumes arm by arm):
  fit   honesty-prompt replies + states on vector-fit texts -> vectors + gate
        (leftover CV AUC vs within-type label-permutation null; test stage refuses a failed gate)
  dose  alpha grid per family on dose texts, without and with image -> chosen alpha by RULE
  test  base, prompt, the chosen arm of each family, controls, on test texts
        --vectors_from N: seed N's vectors / layer / alphas on this seed's test texts
        --model TAG (transfer.MODELS): another model on these test texts -> dose_<TAG>[_s<seed>]/; the 4B
          vectors with alphas rescaled by the ratio of median state norms (width asserted), or --prompts_only
  sweep push size as a share of the median layer state norm (5-25%) on dose texts: every direction at
        every size (v1 and v2 prompt), size-matched random draws (isotropic / activation-spread /
        estimation-matched), combo swap, direction-only (rotate) and size-only (scale) pushes, a readout
        (push, along / off the general axis, x natural spread) and a no-generation geometry check
        -> dose_sweep_rows.json, dose_sweep.json
RULE (fixed before any dose data): lowest no-image mirage rate among arms whose with-image
accuracy is within 5 points of prompt-only and whose looping rate is <= 2%;
ties -> higher abstain, then the lower dose.
"""
import argparse, collections, contextlib, json, re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from . import analysis as A
from .slake import correct
from .extract import MODEL, messages
from .steer import HONEST, HONEST_V2, direction, Steer, generate

# the prompt's leftover mirages are mostly this question form (09-30 fit: 5/7; the prompt fixes 1/6 of them)
TEMPLATE = re.compile(r"^what organ is the .*(gray|grey|black|white|dark|bright).*part")
SAMPLE = dict(do_sample=True, temperature=0.7, top_p=0.95)
GRID = {"general": [(1, 0), (2, 0), (3, 0)], "leftover": [(0, 1), (0, 2), (0, 3)],
        "combo": [(1, 1), (1, 2), (2, 1)]}           # (general x, leftover x)
SHARES = (5, 10, 15, 20, 25)                         # sweep push sizes, % of the median layer state norm


def split(texts, lab, seed=0):
    train = A.text_split(texts, 0.6, seed=seed, strat=lab)
    fit = train.copy()
    fit[train] = A.text_split(texts[train], 2 / 3, seed=seed, strat=lab[train])
    return fit, train & ~fit, ~train


def template_direction(X, texts, cat, fit):
    """Unit (template - other Organ) texts on type-centred base states, fit rows only: a vector
    that knows the question FORM and nothing about mirages. Control for the leftover vector."""
    X = A.center_by(X, cat, fit)
    tm = np.array([bool(TEMPLATE.search(t)) for t in texts])
    v = X[fit & tm].mean(0) - X[fit & (cat == "Organ") & ~tm].mean(0)
    return v / np.linalg.norm(v)


def looping(resp, n=8, k=4):
    """Degenerate reply: some n-word run repeats k+ times (n=8,k=4: base 1%, prompt 0%, broken x4 arm 79% in steer_L34) ('to to to...', '— no. Actually')."""
    w = resp.split()
    c = collections.Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return bool(c) and max(c.values()) >= k


def choose(summ, family, tol=0.05, max_loop=0.02):
    """RULE above. summ: {arm: {family, dose, mirage, abstain, acc, loop}}, must hold 'prompt'."""
    ref = summ["prompt"]["acc"]
    ok = [a for a, s in summ.items() if s["family"] == family and s["acc"] >= ref - tol and s["loop"] <= max_loop]
    return min(ok, key=lambda a: (summ[a]["mirage"], -summ[a]["abstain"], summ[a]["dose"])) if ok else None


def summarise(rows):
    out = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        no = [r for r in rows if r["arm"] == arm and r["cond"] == "without_image"]
        im = [r for r in rows if r["arm"] == arm and r["cond"] == "with_image"]
        out[arm] = dict(family=no[0]["family"], dose=no[0]["dose"],
                        **{o: float(np.mean([r["outcome"] == o for r in no])) for o in ("abstain", "hedged", "mirage", "truncated")},
                        acc=float(np.mean([bool(r["correct"]) for r in im])) if im else float("nan"),
                        loop=float(np.mean([r["loop"] for r in no + im])))
    return out


def label(resp):
    pred = A.gen._extract_open_answer(resp)
    return pred, A.outcome({"without_image": {"response": resp, "pred_normalized": A.gen._normalize_open_text(pred)}})


@torch.no_grad()
def prompt_states(model, proc, recs, layer, bs):
    """Residual at the last prompt token without image (= extract.py last_prompt), batched, left-padded.
    layer: int -> [n, d]; list -> [n, len, d]."""
    out = []
    for i in range(0, len(recs), bs):
        texts = [proc.apply_chat_template(messages(r, []), add_generation_prompt=True, tokenize=False) for r in recs[i:i + bs]]
        hs = model(**proc(text=texts, return_tensors="pt", padding=True).to(model.device), output_hidden_states=True).hidden_states
        out.append(np.stack([hs[l][:, -1].float().cpu().numpy() for l in np.atleast_1d(layer)], 1))
    X = np.concatenate(out)
    return X if np.ndim(layer) else X[:, 0]


def with_prompt(recs, hon):
    """hon: False (base) / True (HONEST) / a prompt suffix, appended to the system prompt."""
    return [dict(r, system_prompt=r["system_prompt"] + (HONEST if hon is True else hon)) for r in recs] if hon else list(recs)


def run_arms(model, proc, arms, no_img, with_img, path, a, samples=0, label=label, prep=lambda r: r):
    """arms: [(name, family, dose, honest, [(v, alpha[, Steer mode]), ...])], honest: False / True (HONEST) / a prompt suffix.
    Appends rows to `path`, skips arms already there.
    label: reply -> (pred, outcome); prep: per-model record transform (e.g. a reasoning-format prompt)."""
    rows = json.load(open(path)) if path.exists() else []
    done = {r["arm"] for r in rows}
    for name, family, dose, hon, vecs in arms:
        if name in done:
            continue
        for with_image, recs in ((False, no_img), (True, with_img)):
            rs = [prep(r) for r in with_prompt(recs, hon)]
            passes = [(0, {})] if not samples else [(k + 1, SAMPLE) for k in range(samples * (not with_image))]
            for s, kw in passes:
                with contextlib.ExitStack() as st:
                    for v, alpha, *mode in vecs:
                        st.enter_context(Steer(model, a.layer, v, alpha, *mode))
                    resps = generate(model, proc, rs, with_image, a.max_new_tokens, a.bs, **kw)
                for r, resp in zip(rs, resps):
                    pred, o = label(resp)
                    rows.append(dict(uid=r["unique_id"], text=A.norm_text(r["variant_question_text"]), category=r["category"],
                                     cond="with_image" if with_image else "without_image", arm=name, family=family, dose=dose,
                                     sample=s, response=resp, pred=pred, correct=correct(pred, r["ground_truth_raw"]),
                                     outcome=o, loop=looping(resp)))
        json.dump(rows, open(path, "w"))
        s = summarise(rows)[name]
        print(f"{name:36s} mirage {s['mirage']:.2f} hedged {s['hedged']:.2f} abstain {s['abstain']:.2f} "
              f"| with-image acc {s['acc']:.3f} | loop {s['loop']:.2f}", flush=True)
    return rows


def pool(run, seeds):
    """Test-stage arms across split seeds (T=0 rows only). Vector arms pool by family: each seed picks its own alpha."""
    per = {}
    for s in seeds:
        d = Path(run) / ("dose" + (f"_s{s}" if s else ""))
        if not (d / "test_rows.json").exists():
            print(f"seed {s}: no test rows, skipped"); continue
        g, ch = json.load(open(d / "gate.json")), json.load(open(d / "dose_choice.json"))["chosen"]
        print(f"seed {s}: layer {g['layer']} gate {'PASS' if g['gate'] else 'FAIL'} (AUC {g['auc']:.2f} p={g['p']:.3f}, "
              f"leftover mirages {g['n_mirage']}) | chosen {ch}")
        for arm, x in summarise([r for r in json.load(open(d / "test_rows.json")) if not r["sample"]]).items():
            per.setdefault(x["family"] if x["family"] in GRID else arm, []).append(x)
    for k, xs in per.items():
        ms = lambda f: f"{np.mean([x[f] for x in xs]):.3f}±{np.std([x[f] for x in xs]):.3f}"
        print(f"{k:50s} seeds {len(xs)} | mirage {ms('mirage')} abstain {ms('abstain')} acc {ms('acc')} loop {ms('loop')}")
    xf = [(s, f) for s in seeds if s for f in sorted((Path(run) / f"dose_s{s}").glob("test_rows_from_s*.json"))
          if (f.parent / "test_rows.json").exists()]
    if not xf:
        return
    df, _, _ = A.load(run)
    idx, lab, _, _ = A.unique_texts(df)
    texts = df.question.map(A.norm_text).values[idx]
    for s, f in xf:
        k = int(f.stem.rsplit("_s", 1)[1])
        fit, dose, _ = split(texts, lab, k)
        seen = set(texts[fit | dose])          # texts the source seed fitted vectors or chose alpha on
        own = [r for r in json.load(open(f.parent / "test_rows.json")) if not r["sample"] and r["family"] in ("base", "prompt", *GRID)]
        moved = [dict(r, arm=f"[s{k} vectors] " + r["arm"]) for r in json.load(open(f)) if not r["sample"]]
        for part, keep in (("all", lambda t: True), (f"seen by s{k}", lambda t: t in seen), (f"unseen by s{k}", lambda t: t not in seen)):
            sub = [r for r in own + moved if keep(r["text"])]
            print(f"\nseed {s} test texts, {part} (n={len({r['text'] for r in sub})})")
            for arm, x in summarise(sub).items():
                print(f"  {arm:58s} mirage {x['mirage']:.3f} abstain {x['abstain']:.3f} acc {x['acc']:.3f} loop {x['loop']:.2f}")


@torch.no_grad()
def geometry(model, proc, arms, recs, a, probes):
    """No generation: mean steered last-prompt state per arm at each probe layer, on that layer's base
    mirage-abstain axis, in units of its fit gap. probes: {layer: (unit v, gap)}."""
    out = {}
    for name, _, _, hon, vecs in arms:
        with contextlib.ExitStack() as st:
            for v, alpha, *mode in vecs:
                st.enter_context(Steer(model, a.layer, v, alpha, *mode))
            X = prompt_states(model, proc, with_prompt(recs, hon), list(probes), a.bs)
        out[name] = {l: float((X[:, k] @ v).mean() / g) for k, (l, (v, g)) in enumerate(probes.items())}
    return out


def sweep(model, proc, a, out, src, S, L, lab, cat, texts, fit, no_img, with_img, arm, V):
    """--stage sweep (see top). S: base states [text, layer, d]; arms use this seed's vectors at a.layer."""
    j = L.index(a.layer)
    X = S[:, j]
    vg, gg = V["vg"], float(V["gg"])
    nrm = float(np.median(np.linalg.norm(X[fit], axis=1)))
    unit = lambda v: v / np.linalg.norm(v)
    ch = json.load(open(src / "dose_choice.json"))["chosen"]["combo"] or "prompt + combo g2 l1"
    cv = next(x for x in [arm("combo", g, l) for g, l in GRID["combo"]] if x[0] == ch)[4]
    mix = sum(al * v for v, al in cv)
    cpush = float(np.linalg.norm(mix))
    g = json.load(open(src / "gate.json"))
    rng = np.random.default_rng(1)
    Xc, Xt = X[fit] - X[fit].mean(0), A.center_by(X, cat, fit)[fit]

    def est():      # diff of means of random fit-text groups the leftover fit's size (type-centred, as the leftover)
        ii = rng.permutation(len(Xt))
        return unit(Xt[ii[:g["n_mirage"]]].mean(0) - Xt[ii[g["n_mirage"]:g["n_mirage"] + g["n_abstain"]]].mean(0))
    dirs = {"general": vg, "leftover": V["vl"], "combo": unit(mix), "template": template_direction(X, texts, cat, fit)}
    rand = {"isotropic": lambda: unit(rng.normal(size=vg.shape)), "activation": lambda: unit(Xc.T @ rng.normal(size=len(Xc))),
            "estimation": est}
    fams, shares, draws, two = (["combo"], (10, 20), 1, (10,)) if a.smoke else (list(dirs), SHARES, a.draws, (10, 20))
    P = lambda sh: sh / 100 * nrm
    pv = (("v1", True), ("v2", HONEST_V2))
    real = [("prompt", "prompt", 0, True, []), ("prompt v2", "prompt v2", 0, HONEST_V2, [])] + [
        (f"{f} {sh}% {pn}", f, sh, h, [(dirs[f], P(sh))]) for f in fams for sh in shares for pn, h in pv]
    ctrl = [(f"random {d} {sh}% #{k}", f"random {d}", sh, True, [(rand[d](), P(sh))]) for sh in shares for d in rand for k in range(draws)]
    ctrl += [(f"general + random-for-leftover {sh}% #{k}", "random combo", sh, True,
              [(vg, cv[0][1] * P(sh) / cpush), (rand["isotropic"](), cv[1][1] * P(sh) / cpush)]) for sh in shares for k in range(draws)]
    ctrl += [("swap: combo direction at general x1 size", "swap", round(100 * gg / nrm, 1), True, [(dirs["combo"], gg)]),
             ("swap: general at combo size", "swap", round(100 * cpush / nrm, 1), True, [(vg, cpush)])]
    ctrl += [(f"rotate {f} {sh}%", "rotate", sh, True, [(dirs[f], P(sh), "rotate")]) for f in ("general", "combo", "template") for sh in two]
    ctrl += [(f"scale {nm} {sh}%", "scale", sh, True, [(None, sgn * P(sh), "scale")]) for sh in two for nm, sgn in (("shrink", 1), ("grow", -1))]
    print(f"sweep: layer {a.layer} median state norm {nrm:.1f} | combo {ch} push {cpush:.1f} ({100 * cpush / nrm:.1f}%) "
          f"| {len(real)} real arms (+ with image) {len(ctrl)} controls", flush=True)
    path = out / "dose_sweep_rows.json"
    run_arms(model, proc, real, no_img, with_img, path, a)
    rows = run_arms(model, proc, ctrl, no_img, [], path, a)

    probes = {l: direction(S[:, L.index(l)], lab, fit, "mirage") for l in L if l >= a.layer}
    geo = geometry(model, proc, real + ctrl, no_img, a, probes)
    summ = summarise(rows)
    table = {}
    for name, fam, sh, hon, vecs in real + ctrl:
        x = dict(summ[name], share=sh)
        if vecs and vecs[0][0] is not None:
            p = sum(al * v for v, al, *m in vecs)        # Steer subtracts p: along > 0 pushes toward abstain
            n, along = float(np.linalg.norm(p)), float(p @ vg)
            x.update(push=n, along=along, off=float(np.sqrt(max(n * n - along * along, 0))), x_spread=n / float(np.std(X[fit] @ unit(p))))
        ref = geo["prompt v2" if hon is HONEST_V2 else "prompt"]
        x["geometry"] = {l: geo[name][l] - ref[l] for l in probes}         # shift on the mirage axis vs prompt only, gap units
        if fam in dirs and hon is True:                 # real v1 arm vs the random draws of the same size
            rm = [summ[c[0]]["mirage"] for c in ctrl if c[1].startswith("random ") and c[1] != "random combo" and c[2] == sh]
            x["random_mirage"], x["beats_random"] = rm, float(np.mean([m > x["mirage"] for m in rm]))
        table[name] = x
    json.dump(dict(layer=a.layer, state_norm=nrm, combo=ch, combo_push=cpush, arms=table), open(out / "dose_sweep.json", "w"), indent=1)
    for name, x in table.items():
        print(f"{name:44s} {x['share']:5.1f}% mirage {x['mirage']:.2f} abstain {x['abstain']:.2f} acc {x['acc']:.3f} loop {x['loop']:.2f}"
              + (f" | along {x['along']:6.1f} off {x['off']:6.1f} x{x['x_spread']:.0f} spread" if "along" in x else "")
              + f" | geo L{a.layer} {x['geometry'][a.layer]:+.2f}" + (f" | beats {x['beats_random']:.2f} of randoms" if "beats_random" in x else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", choices=["fit", "dose", "test", "sweep", "pool"], required=True)
    ap.add_argument("--dose_image", type=int, default=60)
    ap.add_argument("--test_image", type=int, default=200)
    ap.add_argument("--samples", type=int, default=0, help="test stage: extra T=0.7 no-image passes for prompt + chosen combo")
    ap.add_argument("--sample_arms", default="prompt,combo", help="test stage: families or arm names that get the --samples passes")
    ap.add_argument("--seed", type=int, default=0, help="split seed; outputs go to dose_s<seed> (seed 0: dose)")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="pool stage: split seeds to combine")
    ap.add_argument("--vectors_from", type=int, default=None,
                    help="test stage: apply seed N's vectors, layer and chosen alphas to this seed's test texts "
                         "(vector arms + random/shuffled controls only) -> test_rows_from_s<N>.json")
    ap.add_argument("--model", default=None, help="test stage: transfer.MODELS tag (e.g. think4b) instead of the 4B-Instruct")
    ap.add_argument("--prompts_only", action="store_true", help="--model: base / prompt / prompt v2 only (any width)")
    ap.add_argument("--draws", type=int, default=2, help="sweep: random draws per design and size")
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=None, help="default 384, or the --model's own")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0, help="only this many texts / image records per stage; gate not enforced")
    a = ap.parse_args()
    if a.stage == "pool":
        return pool(a.run, [int(s) for s in a.seeds.split(",")])
    from .transfer import MODELS, prep_for, think_label          # here: transfer imports this module
    assert not a.model or a.stage == "test", "--model: test stage only"
    home = lambda s: Path(a.run) / ("dose" + (f"_s{s}" if s else ""))
    tag = f"_{a.model}" if a.model else "_sweep" if a.stage == "sweep" else ""
    out = Path(a.run) / (("dose" + tag + "_smoke" if a.smoke else "dose" + (tag if a.model else "")) + (f"_s{a.seed}" if a.seed else ""))
    out.mkdir(exist_ok=True)
    hf = MODELS[a.model]["hf"] if a.model else MODEL
    a.max_new_tokens = a.max_new_tokens or (MODELS[a.model]["max_new_tokens"] if a.model else 384)
    kw = dict(label=think_label, prep=prep_for(a.model)) if a.model and MODELS[a.model]["think"] else dict(prep=prep_for(a.model)) if a.model else {}

    df, acts, meta = A.load(a.run)
    L = meta["layers"]
    idx, lab, _, _ = A.unique_texts(df)
    texts = df.question.map(A.norm_text).values[idx]
    cat = df.category.values[idx]
    fit, dose, test = split(texts, lab, a.seed)
    S = np.asarray(acts[idx, 1, 0], np.float32)
    allrecs = {(r["unique_id"], r["variant_id"]): r for r in json.load(open(Path(a.run) / "responses.json"))}
    recs = np.array([allrecs[tuple(meta["keys"][i])] for i in idx], dtype=object)    # one no-image record per text
    lim = lambda xs: list(xs[: a.smoke] if a.smoke else xs)

    def image_recs(mask, n):
        rows = np.where(df.question.map(A.norm_text).isin(set(texts[mask])).values)[0]
        return [allrecs[tuple(meta["keys"][i])] for i in np.random.default_rng(0).permutation(rows)[:n]]

    m = fit & np.isin(lab, ["mirage", "abstain"])
    aucs = [A.one_vector_auc_cv(S[m, j], lab[m] == "mirage", np.arange(m.sum()), center=cat[m]) for j in range(len(L))]
    a.layer = L[int(np.argmax(aucs))]
    j = L.index(a.layer)
    print(f"texts: fit {fit.sum()} dose {dose.sum()} test {test.sum()} | layer {a.layer} "
          f"(fit CV AUC {max(aucs):.3f})", flush=True)

    if a.model:                                      # before loading: the 4B vectors need the 4B width
        d = AutoConfig.from_pretrained(hf).text_config.hidden_size
        assert a.prompts_only or d == S.shape[-1], f"{a.model}: hidden size {d} != 4B vectors' {S.shape[-1]} (use --prompts_only)"
    proc = AutoProcessor.from_pretrained(hf)
    proc.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(hf, dtype=torch.bfloat16, device_map="cuda").eval()

    vec_file = out / "vectors.npz"
    if a.stage == "fit":
        fi = np.where(fit)[0]
        if a.smoke:                                   # base mirages first: the prompt's leftovers come from them
            fi = fi[np.argsort(lab[fi] != "mirage", kind="stable")][: a.smoke]
        fr = list(recs[fi])
        # sanity: batched left-padded states must match the extraction's single-record states
        chk = prompt_states(model, proc, fr[:4], a.layer, a.bs)
        ref = S[fi[:4], j]
        cos = (chk * ref).sum(1) / np.linalg.norm(chk, axis=1) / np.linalg.norm(ref, axis=1)
        print("state check cos vs extraction:", np.round(cos, 4), flush=True)
        assert cos.min() > 0.99, "batched prompt states disagree with extract.py"

        hon = [dict(r, system_prompt=r["system_prompt"] + HONEST) for r in fr]
        Sh = prompt_states(model, proc, hon, a.layer, a.bs)
        resps = generate(model, proc, hon, False, a.max_new_tokens, a.bs)
        labh = np.array([label(x)[1] for x in resps])
        cath = cat[fi]
        json.dump([dict(text=A.norm_text(r["variant_question_text"]), category=c, response=x, outcome=o)
                   for r, c, x, o in zip(fr, cath, resps, labh)], open(out / "fit_honest.json", "w"))
        print("honesty prompt on fit texts:", dict(collections.Counter(labh)), flush=True)
        if a.smoke and not (labh == "mirage").any():
            print("smoke: no leftover mirage, hedged stands in for mirage to exercise the code", flush=True)
            labh = np.where(labh == "hedged", "mirage", labh)

        vg, gg = direction(S[:, j], lab, fit, "mirage")
        keep = np.isin(labh, ["mirage", "abstain"])
        y = labh[keep] == "mirage"
        assert y.sum() >= 2 and (~y).sum() >= 2, f"too few leftover mirages to fit ({y.sum()})"
        allfit = np.ones(len(labh), bool)
        vl, gl = direction(Sh, labh, allfit, "mirage", cat=cath)
        rng = np.random.default_rng(0)

        def shuffled(lb):
            lb = lb.copy()
            for c in np.unique(cath):
                ii = np.where((cath == c) & keep)[0]
                lb[ii] = lb[rng.permutation(ii)]
            return lb
        vs, _ = direction(Sh, shuffled(labh), allfit, "mirage", cat=cath)
        rv = rng.normal(size=vl.shape); rv /= np.linalg.norm(rv)

        k = min(5, int(y.sum()), int((~y).sum()))           # folds: every train fold must hold both classes
        auc = lambda lb: A.one_vector_auc_cv(Sh[keep], lb[keep] == "mirage", np.arange(keep.sum()), n_splits=k, center=cath[keep])
        obs = auc(labh)
        null = np.array([auc(shuffled(labh)) for _ in range(a.perms)])
        null = null[np.isfinite(null)]
        p = float((1 + (null >= obs).sum()) / (1 + len(null)))
        gate = bool(np.isfinite(obs) and len(null) >= 0.9 * a.perms and p < 0.05)   # NaN AUC never passes
        np.savez(vec_file, layer=a.layer, vg=vg, gg=gg, vl=vl, gl=gl, vs=vs, rv=rv)
        json.dump(dict(layer=a.layer, cv_folds=k, n_mirage=int(y.sum()), n_abstain=int((~y).sum()), auc=obs,
                       null_mean=float(null.mean()), null_95=float(np.quantile(null, .95)), p=p, gate=gate,
                       gap_general=gg, gap_leftover=gl, cos_general_leftover=float(vg @ vl), cos_leftover_shuffled=float(vl @ vs)),
                  open(out / "gate.json", "w"), indent=1)
        print(f"GATE {'PASS' if gate else 'FAIL'}: leftover CV AUC {obs:.3f} vs null mean {null.mean():.3f} "
              f"95% {np.quantile(null, .95):.3f} p={p:.3f} | mirage {y.sum()} abstain {(~y).sum()} | "
              f"cos(general, leftover) {vg @ vl:.2f}", flush=True)
        return

    xfer = a.vectors_from is not None
    src = home(a.vectors_from) if xfer else home(a.seed) if a.model or a.stage == "sweep" else out     # vectors, gate, choice
    V = np.load(src / "vectors.npz")
    vg, gg, vl, gl, vs, rv = (V[k] for k in ("vg", "gg", "vl", "gl", "vs", "rv"))
    if xfer:                                  # the source seed's layer, not this seed's pick
        a.layer, j = int(V["layer"]), L.index(int(V["layer"]))
        print(f"vectors from seed {a.vectors_from}: layer {a.layer}", flush=True)
    assert int(V["layer"]) == a.layer, "layer changed since the fit stage"
    arm = lambda fam, g, l, hon=True: (f"prompt + {fam} g{g} l{l}", fam, g + l, hon,
                                       [x for x in ((vg, g * float(gg)), (vl, l * float(gl))) if x[1]])

    if a.stage == "sweep":
        return sweep(model, proc, a, out, src, S, L, lab, cat, texts, fit, lim(recs[dose]), image_recs(dose, a.smoke or a.dose_image), arm, V)

    if a.stage == "dose":
        arms = [("prompt", "prompt", 0, True, [])] + [arm(f, g, l) for f, gl_ in GRID.items() for g, l in gl_]
        rows = run_arms(model, proc, arms, lim(recs[dose]), image_recs(dose, a.smoke or a.dose_image), out / "dose_rows.json", a)
        summ = summarise(rows)
        chosen = {f: choose(summ, f) for f in GRID}
        json.dump(dict(summary=summ, chosen=chosen), open(out / "dose_choice.json", "w"), indent=1)
        print("CHOSEN:", chosen, flush=True)
        return

    gate = json.load(open(src / "gate.json"))["gate"] or a.smoke
    chosen = json.load(open(src / "dose_choice.json"))["chosen"]
    fams = {f: next(x for x in [arm(f, g, l) for g, l in GRID[f]] if x[0] == chosen[f])
            for f in GRID if chosen[f] and (gate or f == "general")}
    if not gate:
        print("leftover gate FAILED: testing the general vector and controls only", flush=True)
    lx = float(fams["leftover"][4][0][1]) if "leftover" in fams else 2 * float(gl)
    arms = [("base", "base", 0, False, []), ("prompt", "prompt", 0, True, [])] + list(fams.values()) + [
        ("prompt + shuffled-label (leftover alpha)", "control", 0, True, [(vs, lx)]),
        ("prompt + random (leftover alpha)", "control", 0, True, [(rv, lx)]),
        ("prompt + leftover REVERSED", "control", 0, True, [(vl, -lx)])]
    sfit = split(texts, lab, a.vectors_from)[0] if xfer else fit      # the vectors' own fit texts
    vt = template_direction(S[:, j], texts, cat, sfit)
    vp = vl - (vl @ vt) * vt                                # leftover with its template component removed
    print(f"cos(leftover, template) {vl @ vt:.2f}", flush=True)
    arms += [("prompt + template direction (leftover alpha)", "control", 0, True, [(vt, lx)]),
             ("prompt + leftover minus template (leftover alpha)", "control", 0, True, [(vp / np.linalg.norm(vp), lx)])]
    # size-matched controls at the combo's actual push |g*gg*vg + l*gl*vl| (seed 0: g2 l1, ~86)
    cv = (fams["combo"] if "combo" in fams else arm("combo", 2, 1))[4]
    push = float(np.linalg.norm(sum(al * v for v, al in cv)))
    Xc = S[sfit, j] - S[sfit, j].mean(0)
    va = Xc.T @ np.random.default_rng(0).normal(size=len(Xc)); va /= np.linalg.norm(va)   # where fit states spread
    print(f"combo push {push:.1f}", flush=True)
    arms += [("prompt + random (combo norm)", "control", 0, True, [(rv, push)]),
             ("prompt + activation random (combo norm)", "control", 0, True, [(va, push)]),
             ("prompt + general + random-for-leftover (combo alphas)", "control", 0, True, [(vg, cv[0][1]), (rv, cv[1][1])])]
    arms += [("prompt v2", "prompt v2", 0, HONEST_V2, [])] + (
        [("prompt v2 + combo (chosen)", "prompt v2", fams["combo"][2], HONEST_V2, fams["combo"][4])] if "combo" in fams else [])
    if xfer:
        arms = [x for x in arms if x[1] in GRID or "random" in x[0] or "shuffled" in x[0]]
    if a.model:
        prompts = ("base", "prompt", "prompt v2")
        arms = [x for x in arms if x[1] in prompts and not x[4] or not a.prompts_only and (x[1] in prompts or x[1] in GRID or "random" in x[0])]
        if not a.prompts_only:                       # same push as a share of the state norm: rescale by the norm ratio
            nI = float(np.median(np.linalg.norm(S[fit, j], axis=1)))
            nM = float(np.median(np.linalg.norm(prompt_states(model, proc, [kw["prep"](r) for r in lim(recs[fit])[:64]], a.layer, a.bs), axis=1)))
            print(f"{a.model}: median state norm at layer {a.layer} {nM:.1f} vs 4B-Instruct {nI:.1f} -> alphas x{nM / nI:.3f}", flush=True)
            arms = [x[:4] + ([(v, al * nM / nI, *m) for v, al, *m in x[4]],) for x in arms]
    rows = run_arms(model, proc, arms, lim(recs[test]), image_recs(test, a.smoke or a.test_image),
                    out / (f"test_rows_from_s{a.vectors_from}.json" if xfer else "test_rows.json"), a, **kw)
    if a.samples and not xfer and not a.model:
        want = a.sample_arms.split(",")
        pick = [x for x in arms if x[0] in want or (x[1] in want and x[1] != "control")]
        run_arms(model, proc, [(x[0] + " [sampled]",) + x[1:] for x in pick],
                 lim(recs[test]), [], out / "test_rows.json", a, samples=a.samples)


if __name__ == "__main__":
    main()
