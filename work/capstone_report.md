# Capstone Report — Content Refresh Priority

- **Author:** Khalil Zufar
- **Lane:** Content Refresh Priority
- **Repo:** https://github.com/khalilzufar/FlyRank-ML

## 0. Abstract

We asked whether a model could rank content pages for refresh review better than a transparent stale-and-visible rule. We used 111,133 anonymized rows across 49 pseudonymous client groups and used `last-30d impressions < 80% of prior-30d impressions` as an observed target. A seeded random forest was compared with the rule under a client-grouped holdout. The model reached grouped Precision@50 0.580 versus 0.580 for the rule, with a 59.8% base rate. The output is decision support for human review, not a causal claim about Google traffic.

## 1. Problem framing

Editorial teams have limited review capacity. The output is a ranked queue of pages to inspect first, where a human checks intent, demand, seasonality, business priority, and the live page before choosing an action.

## 2. Data safety

This run uses the gated FlyRank warehouse release. Label-derived fields and pseudonymous IDs are excluded from the model matrix; IDs are used only for tracing and grouped validation. No names, domains, URLs, titles, raw queries, credentials, or raw warehouse exports are written to public artifacts.

## 3. Baseline

The transparent baseline combines percentile ranks for staleness, visibility, CTR risk, and position risk. On the grouped holdout it achieved Precision@50 0.580.

## 4. Model / analysis

The model is a seeded random forest over numeric performance/content fields with missingness flags and one-hot categorical context. The target is the observed outcome `last-30d impressions < 80% of prior-30d impressions`.

## 5. Evaluation

The primary split is a client-grouped holdout: 39 client groups for training and 10 for testing. The model's grouped Precision@50 was 0.580, average precision 0.716, and ROC AUC 0.696. The test base rate was 59.8%. A random row holdout is reported in `work/outputs/capstone_results.json` as a comparison, not the primary estimate.

## 6. Interpretation

The strongest model features in this run were previous/current impression measures, visibility days, content age, position, and current engagement/search fields. These are associations used for prioritization; feature importance is not a causal effect.

## 7. Recommendation

Start with high-confidence rows combining model risk and visible demand. Review CTR/snippets, engagement/readability, staleness, and content depth as appropriate. Never auto-publish, delete, redirect, or claim that a refresh will cause recovery from this queue.

## 8. Reproducibility

Install `requirements.txt`, then run `work/notebooks/w06_validation_audit.ipynb`, `work/notebooks/w07_action_playbook.ipynb`, and `work/notebooks/capstone.ipynb` top to bottom. The seed is 42. Receipts are in `work/outputs/`, and charts are in `work/figures/` and `docs/assets/`.

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai). This paper uses the gated FlyRank warehouse release.
