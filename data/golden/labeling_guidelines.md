# Golden Set Labeling Guidelines

## Purpose
This guide defines how examples in `data/golden/golden_candidates.csv` are
manually reviewed into the final evaluation set `data/golden/golden_set.csv`.
The golden set is **held out** from every training and retrieval component.

## 1. How examples were sampled
- Source pool: `data/processed/spotify_pairs.csv` (41,064 customer → brand
  response pairs for SpotifyCares, 2013–2017).
- Target: 200 examples (within the 150–250 range requested).
- Seed: 42 (`numpy`/`pandas` deterministic, Python built-in `random` seeded).
- The pool was split into first-pass buckets using `src/rule_labeler.py`, a
  deliberate, transparent keyword-rule classifier. Bucket quotas were set
  upfront so that both common intents (subscription, technical, payment,
  playback) and rare intents (family plan, feature feedback, other) are
  guaranteed representation:
  subscription_plan 28, technical_issue 26, payment_billing 22,
  account_access 20, playback_issue 20, content_missing 20,
  appreciation 18, device_compatibility 14, family_plan 12,
  feature_and_feedback 12, other 8.
- Within each bucket we over-sampled: ambiguous messages (≥2 rules fired),
  noisy messages (emojis, all-caps, repeated punctuation, non-ASCII),
  short/long messages, and we guaranteed ≥1 example per month available.
- Sampling summary is saved in `data/golden/sampling_summary.json`.
- IMPORTANT temporal caveat: 99.96% of SpotifyCares pairs fall in 2017 and
  ~95% in Oct–Nov 2017. The golden set therefore has *near-zero genuine
  temporal diversity*; month coverage is a coarse best-effort (6 distinct
  months, 87% from the Oct–Nov tail). This is a documented limitation, not a
  sampling failure we try to hide.

## 2. The first-pass label is a SUGGESTION
The `first_pass_label` column was machine-generated for sampling only. A
human reads every candidate text and writes the final `intent`. If the human
disagrees with the suggestion, the human label wins and a note is added.
The final labels live in `golden_set.csv` (`intent` column).

## 3. Intent definitions
The taxonomy is `data/intent_taxonomy.json`. Reviewers must read the
`description`, `examples`, `negative_examples`, and `boundaries` for each
intent before labelling. The taxonomy's `priority_rules` resolve overlaps
(e.g. payment_billing > subscription_plan when money movement is explicit;
account_access > technical_issue for login problems; playback_issue >
technical_issue for playback behavior).

## 4. Escalation definition
`should_escalate` = would a competent human support agent need to act on
this thread to resolve it safely?
- **escalate (True)** when any of the following are present:
  - account-specific verification is required (verify a payment, unlock an
    account, change billing details, refund a charge, remove a family member)
  - security/fraud signals (hacked, unauthorized charge, account takeover)
  - the message is in "other" and we cannot determine intent
  - high frustration + no known resolution pattern in the history
  - the request implies a real-world obligation (money back, cancel billing)
- **auto (False)** when a safe, generic, informational reply can fully
  resolve the user's question without touching account data (e.g. "how do I
  get premium", "is Spotify on Samsung TV", "thanks it works now",
  "when is album X available", "please add feature Y" — acknowledge only).

Escalation is about the *customer message and its safe resolution*, NOT a
prediction of the brand's historical behavior.

## 5. Ambiguity rules
- If two intents genuinely both apply, apply taxonomy `priority_rules`, then
  `human_notes` explains the tie-break.
- If the text is truly content-free ("hello", "dm me"), label `other`.
- Noise (typos, missing punctuation, ALL CAPS, emoji) is expected and must
  NOT change the intent decision.
- The label reflects what the user is ASKING about, not what they are doing
  ("I sent a DM" → the underlying problem, if recoverable; else `other`).

## 6. How disagreements were resolved
- A single primary reviewer labels every example.
- The reviewer's label is always compared against the first-pass suggestion.
- Where they disagree, the example gets the reviewer label with a `human_notes`
  entry explaining why (this produces a natural set of "hard cases").
- Any example the reviewer marks as genuinely ambiguous AND requiring a human
  nuance decision additionally gets `should_escalate=True` unless retrieval
  later shows a high-confidence standard resolution.
- There is no independent second annotator in this project; agreement numbers
  in the report therefore measure reviewer-vs-first-pass-disagreement as an
  ambiguity proxy, NOT inter-annotator agreement. This is stated honestly.

## 7. Output schema of golden_set.csv
| column | meaning |
|---|---|
| id | stable example id `g001..g200` |
| text | evaluation input (cleaned message; matches pipeline input) |
| raw_text | original tweet text (transparency) |
| intent | final reviewed intent label |
| should_escalate | final reviewed escalation flag (True/False) |
| human_notes | reviewer notes, esp. for hard/ambiguous/boundary cases |
| customer_tweet_id, conversation_id | source keys, used to EXCLUDE from retrieval index |
| month, flag_* , stratum_bucket | sampling metadata |