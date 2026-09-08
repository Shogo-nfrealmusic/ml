# ml

Learning AI/ML from the ground up — not just calling APIs, but understanding what happens inside.

I build things (a photo-services company, web tools, AI agents). I can ship with AI assistance.
What I could not do was judge whether a model's output was actually right.
This repo is me fixing that.

## Approach

Top-down. Run a complete, working system first — then go back and dig into the parts.
Theory-first order (math → theory → implementation) is where most people quit.

## Progress

| Phase | Focus | Status |
|---|---|---|
| **0** | Run two finished systems end to end | **In progress** |
| 1 | Rebuild them, understand each step | — |
| 2 | Classical ML on real data | — |
| 3 | Deep learning | — |
| 4 | LLMs, evals, RAG, tools | — |
| 5 | Ship something people pay for | — |

## phase0

Run two finished systems end to end. Understanding comes later — the goal here is to see
the shape of the thing once, by hand.

**1 · Tabular classification** — `01_first_loop.ipynb`

scikit-learn, one full loop: load data → inspect → train/test split → fit → predict →
evaluate → feature importance. The features arrived as columns someone else had chosen.

**2 · Image classification** — `02_fastai_lesson1.ipynb` · [notes](phase0/02_notes.md)

fast.ai Lesson 1, on Kaggle. Collect images → clean → label → resize → split → fine-tune a
pretrained ResNet18 → predict on an image it never saw. No columns here: the network works
out what to look at on its own.

The gap between those two is the interesting part. In the first, a human decided what was
worth measuring. In the second, nobody did.

**Still to do in phase 0:** structured output from an LLM — a schema, retries on violation,
ten test cases with expected values, cost and latency per call.

## Setup

```bash
cd phase0
uv sync
uv run jupyter lab
```

`02` was run on Kaggle rather than locally — fastai wants a GPU, and Kaggle gives one for
free with everything preinstalled. The notebook is committed here; the environment above is
for the local work.

## Stack

Python 3.12 · uv · scikit-learn · pandas · numpy · matplotlib · Jupyter · fastai (Kaggle)
