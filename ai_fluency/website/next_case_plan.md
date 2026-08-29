# The plan to keep building

## Next named piece

**Future case: time-aware content refresh ranking.** Re-run the current case on a future window, compare the grouped and time-aware splits, and publish the result as a second evidence ribbon on the portfolio.

## How to add the next case

1. Add a public-safe case record with question, data boundary, metric, split, limitation, and next check.
2. Add one card to the case explorer and one entry to the agent registry.
3. Run the public-safety and link checks.
4. Ask one human reviewer to test the page on a phone.
5. Update the changelog and the evidence ribbon only after the checks pass.

## Reminder evidence

The repository's `data-path-smoke` scheduled workflow is the current reminder mechanism. It keeps the data-path check visible in GitHub Actions; it does not automatically publish a new case.
