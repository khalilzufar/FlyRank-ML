# Automation workflow v2

## Purpose

Turn a completed research artifact into a reviewable portfolio update without losing the evidence or overstating the claim.

## Step diagram

```text
brief + evidence
      ↓
1. extract the claim and its boundary
      ↓
2. select one chart, one metric, and one decision
      ↓
3. draft the plain-language explanation
      ↓
4. run the public-safety and link checks
      ↓
5. publish the page + record the change
      ↓
6. human review: approve, revise, or stop
```

## Prompt/configuration contract

1. **Claim extractor** — “Read only the supplied evidence. Return `question`, `measured_result`, `comparison`, `limitations`, and `unsupported_claims`. Use `observed` and `measured`; never infer causality.”
2. **Story editor** — “Write a 90-word explanation for a non-technical reader. Keep the metric, split, base rate, and one limitation. Do not add facts not present in the evidence.”
3. **Reviewer** — “Check every number against the receipt, every URL for public safety, and every recommendation for a human-review owner. Return PASS or a numbered fix list.”
4. **Publisher** — “Update only the intended static page and its change note. Do not add credentials, raw rows, client names, or private URLs.”

## Five dry runs

| Run | Input | Output | Result |
|---|---|---|---|
| 1 | grouped holdout receipt | paper abstract | PASS; claim kept directional |
| 2 | action-mix JSON | recommendation table | PASS; human review retained |
| 3 | feature-importance chart | employer summary | PASS; no causal language |
| 4 | broken paper URL | publish checklist | STOP; link must be fixed first |
| 5 | text containing a private-looking URL | safety checklist | STOP; URL removed before publication |

## Time/value estimate

A manual pass took about 15 minutes in the dry-run estimate. The reusable workflow takes about 4 minutes for extraction, checks, and formatting, saving approximately 11 minutes per update. The estimate is directional, not a production KPI.

## Known failure points

- A metric can be correct but attached to the wrong split.
- A static link can return 404 while the repository itself is healthy.
- A recommendation can sound causal even when the model is only decision support.
- A future editor may paste raw client material into the prompt; the workflow must stop and remove it.
- A successful automated check does not replace a human reviewer.
