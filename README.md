# Mirage probes: recreation and steering on medical VQA

Do probes and steering vectors catch a vision-language model that answers about an image it never saw?

[Report](https://ztgunderson.github.io/mirage-probes-steering/) | [Full tables](docs/details.html) | [Upstream paper repo](https://github.com/danielbenlevi/mirage_probes) | [Cite](#cite)

## Abstract

Ben-Levi et al. call a confident answer about a missing image a "mirage" and show that linear probes can read it from model activations. This repo recreates their probes on Qwen3-VL-4B-Instruct, a much smaller model than the ones in the paper, and the recreation held. It then checks what the strongest probe reads: mostly the model's own reply, not the state before it answers. It adds steering vectors, used on open-ended medical questions (SLAKE) as a second test of the probes, compared against a one-sentence honesty instruction. In the report the instruction did better than the vector on small held-out sets, and controls do not yet separate the vector's direction from the size of the push. Results are preliminary: test sets are small, labels come from a regex, and one section is unreviewed.

The method, code, prompts, generator and labels for the recreation are Ben-Levi et al.'s. The model swap, label fix, probe controls, SLAKE analysis and steering work are new here.

## Repository layout

```
miragelab/   package: extraction, probes, SLAKE generation, steering, controls
tests/       test_miragelab.py (no model loads), dryrun.py
review/      executed notebooks 01-05 (outputs kept)
scripts/     queue/chain-*.sh, the overnight run sequences as they were run
patches/     one patch adding Qwen3-VL-4B to upstream's model registry
docs/        report (index.html), full tables (details.html), figures
upstream/    git submodule: danielbenlevi/mirage_probes, pinned at 06c8d2d
```

## Setup and reproduce

```bash
git clone --recurse-submodules https://github.com/Ztgunderson/mirage-probes-steering
cd mirage-probes-steering
python -m venv .venv && . .venv/bin/activate
pip install torch transformers accelerate scikit-learn numpy pandas pillow requests tqdm matplotlib phate pytest jupyter
python -m pytest -q tests

export MIRAGE_ROOT=/path/to/data-root     # holds runs/, raw_data/, data/
cd "$MIRAGE_ROOT" && export PYTHONPATH=<repo>
python -m miragelab.slake --out runs/slake
python -m miragelab.extract --dataset slake --responses runs/slake/responses.json --out runs/slake/acts
python -m miragelab.steer --run runs/slake
python -m miragelab.pressure --run runs/slake
python -m miragelab.dose --run runs/slake --stage {fit,dose,test,sweep,pool} [--seed N]
python -m miragelab.introspect --run runs/slake
cd <repo>/review && jupyter lab           # notebooks 01-05, in order
```

- Versions used: torch 2.11.0, transformers 5.16.1, scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, phate 2.0.0 (Jetson Orin, Python 3.10). There is no pyproject or requirements file.
- Generation needs an OpenAI-compatible vLLM server serving Qwen3-VL-4B-Instruct as `qwen3-vl-4b` at `http://127.0.0.1:8000/v1`.
- Datasets are not included. SLAKE is expected under `raw_data/slake/`; VQA-RAD and MedXpertQA-MM under `data/` via upstream's loaders.
- Run outputs (about 2.5 GB) are not in the repo. The notebooks need those outputs to re-execute, so a clean clone shows the saved outputs only. The VQA-RAD generation step is in the report's Appendix F and `scripts/queue/`; its exact original invocation was not recorded.

## Status and caveats

- Preliminary. Main results use one model, greedy decoding, regex outcome labels, and held-out sets of 85 to 114 texts.
- Section 4.5 of the report (direction versus size of push, cross-model runs) comes from runs that no notebook has reviewed. Tables 27 to 42 in the full tables are the same unreviewed runs. Treat them as leads.
- Several runs never completed (extra seeds, sampled decoding, cross-model steering transfer). The details page lists them.

## Cite

```bibtex
@misc{gunderson2026mirageprobes,
  author = {Gunderson, Zachary},
  title  = {Mirage probes: recreation and steering on medical {VQA}},
  year   = {2026},
  url    = {https://ztgunderson.github.io/mirage-probes-steering/}
}
```

Original paper:

```bibtex
@misc{benlevi2026mirageprobes,
  author = {Ben-Levi, D. and Goldfeder, J. and Zhao, W. and Lapid, R. and LeVi, A. and Roush, A. G. and Shwartz-Ziv, R. and Lipson, H.},
  title  = {Mirage Probes: How Vision Models Fake Visual Understanding},
  year   = {2026},
  note   = {arXiv:2606.13870},
  url    = {https://arxiv.org/abs/2606.13870}
}
```

## License

MIT, see `LICENSE`. The upstream repo has no license file, so its code is referenced as a submodule, not copied.

Contact: https://ztgunderson.github.io/
