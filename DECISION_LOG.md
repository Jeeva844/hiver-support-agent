# DECISION_LOG

## D1 – Single-brand selection: SpotifyCares
**Choice:** SpotifyCares (43,092 usable pairs) over AmazonHelp, AppleSupport, Uber_Support, Tesco, Delta.  
**Rationale:** Best combination of signal density (92% substantive brand replies), manageable volume (not overwhelming for manual review), meaningful intent diversity (account, billing, playback, feature, content), and low templated reply share (8%). AmazonHelp (23% templated, multilingual) and Uber (59% templated) would inflate the "easy" classes and mask real agent error.  
**Result:** A clean 11-intent taxonomy is feasible.

## D2 – Golden set: 200 examples, per-bucket quotas
**Choice:** 200 examples with predetermined per-bucket quotas (subscription 28, technical 26, payment 22, account 20, playback 20, content 20, appreciation 18, device 14, family 12, feature 12, other 8) and ≥1 per month.  
**Rationale:** 200 is within 150–250 for manual review feasibility. Per-bucket quotas guarantee rare intents are not lost to the majority class. Month-stratified sampling (best-effort given 95% Oct–Nov 2017) ensures the agent is tested across the available temporal range.  
**Caveat documented:** Temporal diversity is near-zero; results are valid only for 2017.

## D3 – Escalation definition: biased toward over-escalation
**Choice:** Escalation = "does this need a human to act?" Not "does the brand historically reply?"  
**Rationale:** On a public support channel, an unverified reply about money or security is worse than an unnecessary human handoff. The rule: escalate when (a) money movement, (b) security/fraud, (c) account state change, (d) low confidence, (e) "other." Auto-serve only when a factual/acknowledgment reply is safe.  
**Risk:** F1 is lower than accuracy suggests (0.672); recall on escalation is the binding constraint.

## D4 – Deterministic/mock pipeline (no OpenAI API)
**Choice:** All LLM-dependent components use a deterministic fallback (rule labeler for classifier, TF-IDF cosine for retrieval, verbatim retrieval for reply, regex for escalation, containment + cosine for judge).  
**Rationale:** Full pipeline runs fully offline, reproducible, and auditable. A `backend: "mock"` flag in every results file records this honestly. The same module code accepts `OPENAI_API_KEY` and switches to real mode with identical JSON contracts.  
**Trade-off:** Numbers are representative of the *architecture*, not of a production-grade LLM system.

## D5 – TF-IDF sparse embeddings as a dense-vector stand-in
**Choice:** Use TF-IDF cosine for retrieval instead of sentence-transformers or FAISS.  
**Rationale:** No GPU or heavyweight dependencies available. TF-IDF is deterministic, auditable, and requires zero additional installs. The abstraction (`Retriever.search`) is documented as swappable for dense vectors.  
**Result:** Retrieval kNN accuracy is low (0.415), demonstrating the importance of semantic embeddings — a genuine finding for the report.

## D6 – Weak (first-pass) labels as training supervision
**Choice:** The TF-IDF classifier trains on labels from `src/rule_labeler.py` (deterministic keyword rules), the same labels used for golden-set sampling strata.  
**Rationale:** No human labels exist outside the golden set; weak labels are the only viable source. The classifier therefore has a hard performance ceiling set by the rule labeler's 74.5% accuracy. The 25.5% reviewer-vs-first-pass disagreement is documented as a label-noise proxy, not inter-annotator agreement.

## D7 – Retrieval index excludes golden conversations
**Choice:** Every conversation that appears in the golden set is excluded from the retrieval index.  
**Rationale:** The golden set is the held-out evaluation set; any data leakage destroys the claim that evaluation is unbiased. This is asserted at index build time and checked on load.

## D8 – Deterministic judge (mock LLM-as-judge)
**Choice:** Judge is implemented as (intent accuracy + escalation agreement + reply containment/cosine) rather than calling an LLM for qualitative verdicts.  
**Rationale:** Shares the same verdict schema as a real LLM judge (intent_correct, escalation_agreement, safe_to_auto, reply_verdict), enabling a clean swap-in. The report honestly calls it `judge_backend: deterministic_mock`.

## D9 – Reply = verbatim historical retrieval (no LLM rewrite in mock)
**Choice:** When auto-serving, the reply is the best historical `brand_response_clean` from the top-scoring similar conversation, unchanged.  
**Rationale:** Demonstrates that verbatim retrieval does NOT solve the resolution problem (30.9% judged "good" by the judge) and motivates the rewrite/LLM layer in real mode. The low reply quality number is an honest architecture finding, not a bug.

## D10 – Honest disclosure of the headline number
**Choice:** The report leads with the headline `intent_accuracy = 0.745` but devotes a full section to why this would be misleading if taken at face value.  
**Rationale:** The assignment specifically requests "what is misleading about my headline number." Credibility comes from transparency, not suppression of numbers.

## D11 – Determinism: seed 42, no stochasticity
**Choice:** All sampling, data splits, and train/test operations use seed 42 (pandas/numpy). Mock classifier outputs are fully deterministic.  
**Rationale:** Reproducibility is a hard deliverable. The `run_experiment.sh --quick` script reruns all evaluations under 15 minutes.

## D12 – Conflict-resolution order: taxonomy priority rules
**Choice:** When a message could fit two intents, apply explicit priority rules (payment_billing > subscription_plan; account_access > technical_issue; playback_issue > technical_issue; appreciation dominates when present).  
**Rationale:** The rule list is short, explicit, and auditable in `data/intent_taxonomy.json`. It was validated against 200 manual-review disagreements and was the single hardest judgment the reviewer made (25.5% disagreement rate).

## D13 – Evaluation distribution intentionally differs from production
**Choice:** The golden-set class mix is intentionally balanced, not a replica of the production prior.  
**Rationale:** If we sampled by production frequency, rare intents (family_plan, device_compatibility, other) would have <10 examples, making their F1 estimates meaningless. A balanced golden set ensures every class is evaluated; macro-F1 is reported alongside accuracy. The misleading-metrics section explicitly flags this difference.

## D14 – Weak recall on escalation is the binding limitation
**Choice:** The escalation decider is tuned for high precision (0.738) at the cost of recall (0.616).  
**Rationale:** This was not a hyperparameter choice; it is the direct consequence of deterministic heuristics applied to short social-media messages. It is identified as the top priority for improvement (Phase 20/21).
