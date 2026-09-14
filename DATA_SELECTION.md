# Brand Selection — Customer Support on Twitter (twcs)

## Dataset
- Source: `thoughtvector/customer-support-on-twitter` (Kaggle), downloaded locally.
- File: `data/raw/twcs.csv` (492.6 MB)
- Rows: **2,811,774** · Unique authors: 702,777 · No duplicate tweet IDs.
- Columns: `tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`
- **No `conversation_id` column** → conversations reconstructed from reply chains
  (`in_response_to_tweet_id`). Root tweet id used as conversation id.
- Missing values: `response_tweet_id` (1.04M), `in_response_to_tweet_id` (794K);
  no missing `tweet_id / author_id / text / inbound / created_at`.

## Method
1. Measure per-author support volume (`inbound` = customer → company, outbound = company → customer).
2. Define a *usable pair* = brand outbound reply directly replying to a customer's inbound message.
3. Reconstruct 798,012 conversations from reply chains.
4. Qualify reply **quality**, not just volume:
   - templated / "DM us" / link-only redirects vs. substantive replies
   - multi-turn structure share
   - language uniformity

## Candidate comparison (measured)

| Brand | Usable pairs | Customer msgs | Conv. | Median conv size | ≥3-tweet conv | Templated/DM share | Substantive | Non-English |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AmazonHelp | 168,814 | 203,598 | 82,534 | 3 | 62% | 23% | 77% | some (JA/DE) |
| AppleSupport | 106,646 | 131,764 | 80,702 | 2 | 35% | 17% | 83% | 0% |
| Uber_Support | 56,160 | 72,154 | 41,923 | 2 | 36% | **59%** | 41% | 0% |
| **SpotifyCares** | 43,092 | 48,543 | 28,280 | 2 | 37% | **8%** | **92%** | 0% |
| Delta | 42,114 | 45,296 | 26,166 | 2 | 43% | 21% | 79% | 0% |
| Tesco | 38,468 | 34,228 | 16,721 | 4 | 70% | 24% | 76% | 0% |

## Recommendation: **SpotifyCares**

Evidence:
1. **Best reply substance (92% substantive) and lowest templated share (8%)** of any
   brand with >30K pairs. The public thread contains real resolutions (restart the
   app, reinstall, check subscription status, adjust settings) rather than "DM us".
   This maximizes the quality of ~43K retrievable grounded examples.
2. **Single-language, uniform** data (no non-English noise, unlike AmazonHelp).
3. **Well-bounded product surface** — one music-streaming app → 8–12 operational
   intents map naturally (e.g., playback, account access, payment/billing,
   subscription/cancellation, device compatibility, offline/download, playlist
   sync, family plan). Recurring topics confirmed by keyword analysis
   (`songs, music, version, iphone, app, tried, play, same`).
4. **Enough, but not unmanageable, scale** (43K pairs, 28K conversations,
   48.5K customer messages) → a subset pipeline stays locally runnable.
5. **Consistent branded voice** and short actionable replies → ideal for
   retrieval + grounded reply generation and an LLM-as-judge.

### Why not the others
- **AmazonHelp**: 23% templated + multilingual (JA/DE) noise; very broad product
  surface (Prime Video, Fire TV, orders, Alexa, Kindle) → fuzzy intents; reply
  register is formal-closing heavy.
- **AppleSupport**: high volume but 17% templated *and* most conversations resolve
  via DM, so public replies are dominated by triage ("Which iOS version?",
  "Let's go to DM"). Weak grounding material for reply generation.
- **Uber_Support**: 59% templated/DM → disqualified on retrieval value despite volume.
- **Delta**: good substance (79%) but replies routinely demand PNR/confirmation
  numbers → account-specific, high-overturn to human, fewer auto-handle cases.
- **Tesco**: long threads (median 4) but skewed toward individual store-staff
  complaints → weak recurring generic-intent structure.

## Data quality notes for the chosen brand
- Preserve original noisy text (lowercase, typos, no punctuation, emotion) — this
  is the real problem surface; do not over-clean.
- Replies contain mentions (`@SpotifyCares`, numeric handle ids), URLs,
  and agent initials (`/LS`, `/CB`) — strip for modeling, keep originals.
- Mentions/handles and `https://t.co/...` artifacts will be removed before
  intent modeling, not in the raw store.
- ~63% of conversations are isolated customer-msg → brand-reply pairs; that is
  the natural unit for the processed `(customer_text, brand_response)` dataset.

## Generated artifacts
- `data/analysis/brand_summary_table.csv` — all 108 brand candidates
- `data/analysis/brand_analysis.json` — top 50 brand stats
- `data/analysis/brand_selection_summary.json` — ranked comparison
- `data/analysis/brand_deep_dive.json` — reply-quality deep dive
- `data/analysis/brand_turn_structure.json` — conversation structure
- `data/analysis/brand_keywords.json` — top keywords per brand
- `data/analysis/conversation_stats.json` — thread reconstruction stats
- `data/analysis/sample_threads/*.txt` — hand-inspectable sample pairs