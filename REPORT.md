# Agent Evaluation Report — SpotifyCares Support Agent

## 1. Executive summary

We built an end-to-end customer-support agent for **@SpotifyCares** on the
Kaggle customer-support-on-twitter dataset. A 200-example **golden set** was
hand-reviewed by a human with a written rubric; every metric below is computed
on that held-out set, with golden conversations **excluded from training and
retrieval**.

The pipeline classifies an incoming message into 1 of 11 intents, decides
whether a human must act (escalation), and otherwise auto-replies with a
retrieved, provenance-carrying historical brand response.

| headline summary | value |
|---|---|
| intent accuracy (golden, n=200) | 0.745 |
| intent macro-F1 | 0.695 |
| escalation accuracy / F1 | 0.780 / 0.672 |
| escalation recall (the binding constraint) | 0.616 |
| reply "good" rate vs historical human reply | 0.309 |
| safe-to-auto traffic share | 0.485 |

**The core honest claim:** the agent *classifies* support intents about as well
as the deterministic rule system it replaces, but its auto-replies are
frequently not adequate, and it dangerously under-escalates money/security
cases. Reporting the 0.745 accuracy number alone would mislead (see §7).

## 2. Requirements → deliverables

| requirement | where |
|---|---|
| One brand, justified | `DATA_SELECTION.md`, D1 |
| Intent taxonomy (≥5) | `data/intent_taxonomy.json` (11 intents, priorities) |
| Golden set 150–250 + docs | `data/golden/golden_set.csv` (200), `labeling_guidelines.md`, `human_review.json` |
| Leak-free evaluation | `Retriever` golden-exclusion asserted + tested |
| Baselines vs pipeline | `results/*_results.json`, §4 |
| Escalation with reasons | `src/escalation.py`, pipeline_eval |
| Reply generation | `src/reply_generator.py` (retrieval-grounded) |
| Agent pipeline | `src/pipeline.py` |
| Judge + human agreement | `src/judge.py`, `scripts/judge_eval.py`, §5 |
| Failure analysis | `scripts/failure_analysis.py`, §6 |
| Misleading headline analysis | `scripts/misleading_metrics.py`, §7 |
| Decision log | `DECISION_LOG.md` (14 entries) |
| Tests | `tests/` (17 passing, incl. leak + held-out) |
| Reproducibility | `scripts/run_experiment.py --quick` (~40 s) |
| README | `README.md` |

## 3. Method

1. **Data** — Kaggle `thoughtvector/customer-support-on-twitter`; reconsted
   threads via reply chains; selected SpotifyCares (43,092 usable pairs,
   41,064 after cleaning; 99.96% of data is 2017).
2. **Taxonomy** — 11 intents derived from TF-IDF/NMF topic modelling +
   reviewer iteration; explicit `priority_rules` resolve overlaps.
3. **Golden set** — 200 messages, per-bucket quotas, deterministic seed 42;
   every example labeled by the reviewer with intent, escalation, notes
   (25.5% reviewer/first-pass disagreement — used as an ambiguity proxy, not
   inter-annotator agreement).
4. **Pipeline** — LLM-intent-classifier (mock) → escalation decider →
   TF-IDF retrieval → verbatim historical reply.
5. **Evaluation** — all of the above on the golden set only; golden
   conversations excluded from index and training.

## 4. Results

### Intent classification (golden, n=200, mock backend)

| model | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| majority class | 0.035 | 0.006 | 0.002 |
| rule labeler | 0.745 | 0.695 | 0.755 |
| tfidf + logistic regression (weak labels) | 0.740 | 0.691 | 0.741 |
| LLM classifier (mock = rule labeler) | 0.745 | 0.695 | 0.755 |
| retrieval kNN (k=5) | 0.415 | 0.425 | 0.489 |

Per-class F1 (pipeline): account_access 0.905, payment_billing 0.857,
subscription_plan 0.846, technical_issue 0.846, playback_issue 0.829,
family_plan 0.815, appreciation 0.684, content_missing 0.645, other 0.533,
feature_and_feedback 0.457, device_compatibility 0.222.

Observations: (a) the TF-IDF learner cannot beat its weak teacher — evidence
that label noise, not model capacity, caps the number; (b) sparse-TF-IDF
retrieval alone is far weaker (kNN 0.415), motivating dense embeddings in the
real-LLM build; (c) the class that matters most for user trust, payment_billing,
is a highlight; feature/device confusion is the main error cluster.

### Escalation (golden)

| metric | value |
|---|---|
| accuracy | 0.780 |
| precision / recall / F1 | 0.738 / 0.616 / 0.672 |
| human escalate rate | 0.365 |
| model escalate rate | 0.370 |
| false positives / negatives | 16 / 28 |

The agent slightly *under*-escalates overall — worst spot: subscription/status
mismatch and security-adjacent account cases the regexes miss. This is the #1
improvement target.

### Replies (auto-served, n=139 with gold reference)

| metric | value |
|---|---|
| judge "good" rate | 0.309 |
| mean token containment vs historical reply | 0.248 |
| mean tfidf cosine vs historical reply | 0.217 |
| intent-consistent retrieval (auto & human-auto, n=111) | 0.730 |

Verbatim retrieval does not rewrite, name-check, or personalize — its lexical
overlap with the actual human reply is low, and only a third of auto-replies
are judged adequate.

## 5. Judge ↔ human agreement

| judge↔human | value |
|---|---|
| intent agreement | 0.745 |
| escalation agreement (accuracy / Cohen κ) | 0.780 / 0.508 |
| safe-to-auto rate (judge) | 0.485 |

The judge is deterministic (mock): escalation agreement via labels, reply
adequacy via containment/cosine vs the *actual* SpotifyCares reply to that very
conversation. Agreement of κ=0.508 is moderate — acceptable for a
screening-grade judge but a real LLM judge would be the better final gate.

## 6. Failure analysis (n=200)

- **Intent errors: 51 (25.5%)**, largest confusions:
  `payment_billing→device_compatibility` (5), `feature_and_feedback→content_missing` (4),
  `appreciation→technical_issue` (3), `family_plan→subscription_plan` (3),
  playback↔feature mutual confusion (6).
- **Escalation:** 16 false positives (safe money/security framing missed) and
  28 false negatives (account/subscription state changes under-triggered).
- **Replies:** 96 of 139 auto-replies judged poor; the top failure is choosing
  a generic historical reply over a message-specific one.
- Details: `results/failure_analysis.md`.

## 7. Why "0.745 accuracy" would be a misleading headline

Full write-up: `results/misleading_metrics.md`. Abridged:

1. **Label agreement ≠ resolved.** If someone says "74.5% of messages were
   correctly understood", that is true; if they imply "74.5% resolved", it is
   false — only 30.9% of auto-replies are judged adequate and just 48.5% are
   safe-to-auto.
2. **Accuracy is prior-dominated.** macro-F1 (0.695) < accuracy; tiny classes
   (device_compatibility support=4) are invisible to accuracy but not to users.
3. **Escalation "accuracy" hides asymmetry.** 28 false negatives (under-escalate
   on money/security) are far costlier than 16 false positives.
4. **Artificial class mix.** Golden-set quota sampling is deliberately balanced;
   production traffic is ~all 2017 and differently skewed. The number does not
   transfer.
5. **Weak labels.** The human reviewer disagrees with the training labeler on
   25.5%; part of the "error" is label noise the model can never see past.
6. **No lift over the rule baseline.** The mock LLM classifier *is* the rule
   labeler (0.745 = 0.745). A headline implying intelligence would be false.
7. **Temporal blindness.** Training is 99.96% 2017; the agent has never seen a
   post-2017 product decision.
8. **Mock ceiling.** Numbers are for the deterministic architecture; a real LLM
   can only improve them, so quoting them as "the system's number" is unstable.

**The honest headline:** *"A deterministic prototype classifies 2017-era
Spotify support intents at 74.5% accuracy (macro-F1 0.695, escalation recall
0.616, reply adequacy 30.9%) under leak-free, fully reproducible evaluation —
and must not be read as 'resolves 74.5% of customers'."*

## 8. Limitations

- Single brand, single language; 2017 temporal window.
- Weak-supervised training labels; no inter-annotator agreement (one reviewer).
- Mock/deterministic judge and replies; no production LLM.
- Monolingual English-focused taxonomy; non-English messages are rare and noisy.
- 33.6% of historical replies are "DM us" gestures — ambiguous for reply quality
  judgement.

## 9. Reproducibility

- Everything re-runs via `python scripts/run_experiment.py --quick` (~40 s on
  this machine; target <15 min) or `--from-scratch`.
- Deterministic seed 42 throughout; mock backend is fully deterministic.
- Test suite (17 tests) covers taxonomy integrity, golden-set invariants, the
  held-out/retrieval-leak contract, classifier JSON contract, escalation policy,
  and pipeline determinism.

## 10. Future work

1. Real-LLM backend (classifier/replies/judge) behind the identical contracts.
2. Dense embeddings + FAISS for retrieval.
3. Oversample human labels itinerantly to break the weak-label ceiling.
4. Re-tune the escalation decider (raise recall on account/subscription cases).
5. Multi-turn state and a mistake-driven feedback loop.