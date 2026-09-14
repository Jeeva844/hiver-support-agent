# Hiver AI Customer-Support Agent (SpotifyCares)

A reproducible end-to-end customer-support agent built for the Hiver SDE-Intern
take-home: intent classification, grounded reply generation, escalation with
reasons, golden-set evaluation, baselines, an automated judge, failure analysis,
and an honest discussion of what is misleading about the headline metric.

**One hard constraint protects every number in this repo:** the 200-example
golden set (`data/golden/golden_set.csv`, hand-reviewed) is held out from every
training and retrieval component. Golden conversations are provably absent from
the retrieval index (asserted at build time and checked on every load).

## Quick start

```powershell
pip install -r requirements.txt
python scripts/run_experiment.py --quick   # reuse preprocessed artifacts, ~40 s
python scripts/run_experiment.py           # rebuilds any missing artifact
python scripts/run_experiment.py --from-scratch  # raw -> processed -> index
python -m pytest tests -q                  # 17 tests
```

`--quick` finishes in tens of seconds (target was < 15 minutes). All results are
written to `results/*.json`; the human-readable story is in `REPORT.md`.

## Web UI

```powershell
streamlit run app.py        # opens http://localhost:8501 in your browser
```

Two tabs: **Run the agent** (type a message, see intent / confidence /
escalation / grounded reply + retrieval provenance) and **Golden-set explorer**
(test any of the 200 hand-reviewed cases and see ✓/✗ vs the human label).

## Project structure

```
data/
  raw/twcs.csv                     # Kaggle customer-support-on-twitter (downloaded once)
  processed/spotify_pairs.csv      # 41,064 usable SpotifyCares pairs (Phase 2)
  processed/retrieval_index/       # deterministic retrieval index (golden excluded)
  golden/golden_set.csv            # 200 hand-reviewed held-out examples
  golden/human_review.json         # reviewer audit trail (labels + escalation + notes)
  intent_taxonomy.json             # 11-intent taxonomy with priorities
src/
  intents.py rule_labeler.py llm_classifier.py
  retrieval.py escalation.py reply_generator.py pipeline.py judge.py
scripts/
  preprocess.py build_golden_set.py finalize_golden_set.py
  majority_baseline.py rule_baseline.py classifier_baseline.py llm_intent_eval.py
  retrieval_knn_eval.py pipeline_eval.py judge_eval.py failure_analysis.py
  misleading_metrics.py run_experiment.py
tests/                             # 17 pytest tests incl. golden held-out + leak check
results/                           # every metric, confusion matrix, failure summary
DECISION_LOG.md                     # 14 key decisions with rationale
REPORT.md                           # full results + honest limitations
```

## How the agent works

`pipeline.run(query)`:

1. **Classify** the message into one of 11 intents (`src/llm_classifier.py`).
2. **Escalate-or-resolve** (`src/escalation.py`): billing/family/other always
   escalate; account_access escalates on security signals; subscription on
   status mismatch; technical/playback escalate when fixes are already
   exhausted. Deliberately conservative — under-escalation costs more than
   over-escalation on public support.
3. **Reply** (`src/reply_generator.py`): for auto-resolved messages, retrieve
   the best historical `SpotifyCares` reply from a similar conversation and
   return it verbatim with provenance.

## Modes of operation

- **mock (default, this machine):** fully deterministic and offline. The LLM
  classifier falls back to the rule labeler; retrieval uses TF-IDF cosine (a
  documented stand-in for dense embeddings/FAISS); the judge is a deterministic
  proxy (containment + cosine). Every results file records `"backend": "mock"`.
- **real LLM:** set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL` in
  `.env`), `pip install openai`. The same code switches to the OpenAI API with
  identical JSON contracts (strict-JSON classifier, rewritten replies, LLM
  judge). This repo ships and was validated in mock mode.

## Headline results (golden set, n=200, mock backend)

| component | metric | value |
|---|---|---|
| majority baseline | accuracy / macro-F1 | 0.035 / 0.006 |
| rule labeler | accuracy / macro-F1 | 0.745 / 0.695 |
| tfidf+logreg (weak labels) | accuracy / macro-F1 | 0.740 / 0.691 |
| LLM classifier (mock) | accuracy / macro-F1 | 0.745 / 0.695 |
| retrieval kNN (k=5) | accuracy / macro-F1 | 0.415 / 0.425 |
| **pipeline** | **intent acc / macro-F1** | **0.745 / 0.695** |
| pipeline escalation | acc / prec / rec / F1 | 0.780 / 0.738 / 0.616 / 0.672 |
| reply (auto cases) | judged "good" vs historical reply | 30.9% |
| judge↔human escalation | accuracy / Cohen κ | 0.780 / 0.508 |
| safe-to-auto | share of traffic | 48.5% |

## Honest limitations (why "74.5%" is misleading if quoted alone)

See `scripts/misleading_metrics.py` and `results/misleading_metrics.md`. In
short: (1) label agreement ≠ resolution — only 30.9% of auto-replies are
judged lexically adequate; (2) accuracy hides an under-escalation bias
(recall 0.616) exactly where money/security are involved; (3) golden-set class
mix is intentionally balanced, not production-scaled; (4) all training labels
are weak (keyword) labels, so the number includes label noise; (5) the rule
baseline already reaches the same number; (6) all data is 2017 — the agent is
temporally blind after that.