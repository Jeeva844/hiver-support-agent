# Failure Analysis

- examples: 200

## Intent confusions (human -> predicted)
- payment_billing -> device_compatibility: 5
- feature_and_feedback -> content_missing: 4
- appreciation -> technical_issue: 3
- family_plan -> subscription_plan: 3
- feature_and_feedback -> playback_issue: 3
- playback_issue -> feature_and_feedback: 3
- other -> appreciation: 2
- feature_and_feedback -> appreciation: 2
- feature_and_feedback -> other: 2
- appreciation -> subscription_plan: 2
- technical_issue -> content_missing: 2
- account_access -> device_compatibility: 2

## Escalation false positives: 16
- (6) account access, low signal -> verify
- (5) other requires account verification or human review
- (3) payment_billing requires account verification or human review
- (1) family_plan requires account verification or human review
- (1) standard fixes already attempted

## Escalation false negatives: 28
- (10) informational plan/premium FAQ
- (7) device_compatibility is informational/acknowledge-only
- (3) password guidance is self-serve
- (3) content_missing is informational/acknowledge-only
- (3) standard troubleshooting offered
- (2) appreciation = no action

## Reply failures (poor vs gold): 96 (69.1% of auto)