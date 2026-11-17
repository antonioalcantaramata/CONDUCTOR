# Papers

The evidence behind papers written with CONDUCTOR: what was run, what came out, and the scripts that made each
table, figure and case study. Each paper has its own folder with a README that maps every folder to the section or
table it supports.

| Folder | Paper | Status |
|---|---|---|
| [`cim_layer/`](cim_layer/README.md) | *The Grid's Own Language: A Common Information Model Layer for LLM Grid Assistants* (A. Alcántara, I. Karampinis, S. Chatzivasileiadis) | Under review; preprint [arXiv:2611.20026](https://arxiv.org/abs/2611.20026) |

## What a paper folder holds

- **`runs/`**: one folder per experiment. Each has a `manifest.json` with the code commit, the model and the exact
  command it ran with. It also has the graded results (`runs.csv`, `checks.csv`) and a written analysis
  (`analysis.md`), plus the transcript of every run (`runs/**/*.jsonl`).
- **Scripts**: the code that produced each figure, case study or measurement, re-runnable from the repository root.
- **`README.md`**: which folder supports which part of the paper, and how to reproduce it.

## Checking a paper

To check a table, start from the paper folder's README: each table has a `runs.csv` or `analysis.md` that holds
its numbers. Re-grading the recorded runs and the error injection call no AI model:

```
python -m evaluation.ablation analyse papers/<paper>/runs/<experiment>
```

New runs call the AI model configured in `llm_agent/.env`. Their results vary between runs and models, so compare
them with the paper's shares rather than expecting the same answers.

## What is not here

Server logs and scratch caches from the runs (`runs/*/backends/`, `runs/*/cache/`) are git-ignored. They hold
repeated library warnings and local paths, and nothing in a paper is computed from them.
