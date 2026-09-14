# What is misleading about the headline number

The agent correctly resolves 74.5% of customer messages and escalates the rest.

## 1. Accuracy hides class-prior effects
Intent accuracy is dominated by the common classes. macro-F1 (0.695) is below accuracy (0.745); classes with tiny support (device_compatibility support=4) contribute almost nothing to accuracy but matter to a customer whose device is the problem. A 0.745 accuracy claim says almost nothing about the tail.

## 2. Intent-correct does NOT mean resolved
Label agreement is the wrong unit for an agent. The judge found only 30.9% of auto-replies lexically adequate vs the historical human reply (mean containment 0.248), and only 48.5% of traffic is judged 'safe to auto-serve'. Most promising deflection is right, but resolution quality is the binding constraint — accuracy does not measure it at all.

## 3. Escalation 'accuracy' masks cost asymmetry
Escalation accuracy 0.780 treats false positives and false negatives equally, but their costs differ: FP (16) wastes a human's time; FN (28) sends an unverified reply about money/security to a user. Precision 0.738 vs recall 0.616 shows the agent under-escalates exactly where it is most dangerous.

## 4. The evaluation distribution is artificial
The golden set used per-bucket quotas so rare intents are guaranteed (e.g. other=8, device_compatibility=4), but real production traffic is heavily skewed (99.96% of all data falls in 2017; the underlying label distribution is different). Reported metrics are calibrated to an intentionally balanced set, not to production reality.

## 5. Weak (first-pass) labels cap everything
All training labels come from a keyword labeler whose disagreement with the human reviewer is 25.5%. The TF-IDF classifier (0.740) cannot beat its weak teacher, so the 0.745 headline includes label noise as if it were model error.

## 6. Baselines vitiate the claim
The rule labeler alone already reaches 0.745 accuracy and 0.695 macro-F1 it is the same component as the mock LLM (0.745). A headline that implies novelty over trivial baselines would overstate the contribution.

## 7. Temporal blindness
Training is ~all from Oct-Nov 2017 (95%). The agent has never seen a post-2017 Spotify product cycle; any accuracy number generalises only to 2017-era English-language support interactions.

## Bottom line
The honest summary: the agent *classifies* well but *resolves* poorly;
macro-F1 0.695, escalation recall 0.616, reply good-rate 0.309, safe-to-auto 0.485. Reporting only label accuracy would be misleading.