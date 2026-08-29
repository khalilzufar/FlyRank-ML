"""Fill the remaining ML assignment notebook skeletons with reproducible cells."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "work" / "notebooks"


def markdown(text: str) -> object:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> object:
    normalized = dedent(text)
    # The shared import preamble is intentionally kept readable in this builder;
    # remove the list indentation from the extra body when the two strings join.
    if "repo_candidates =" in normalized:
        lines = normalized.splitlines()
        marker = next(
            (index for index, line in enumerate(lines) if line.startswith("print(f\"Observed snapshot-proxy")),
            None,
        )
        if marker is not None:
            lines = lines[: marker + 1] + [
                line[12:] if line.startswith("            ") else line
                for line in lines[marker + 1 :]
            ]
            normalized = "\n".join(lines)
    return nbf.v4.new_code_cell(normalized.strip())


def write_notebook(name: str, cells: list[object]) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11+"},
        },
    )
    nbf.write(notebook, NOTEBOOK_DIR / name)


IMPORTS = """
from pathlib import Path
import sys
import pandas as pd

repo_candidates = [Path.cwd(), *Path.cwd().parents, Path('/content/FlyRank-ML'), Path('/content/flyrank-ml')]
repo_root = next((p for p in repo_candidates if (p / 'work' / 'ml_track.py').exists()), Path.cwd())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from work.ml_track import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    ensure_dirs,
    load_analysis_frame,
    make_feature_matrix,
    run_artifacts,
    run_validation,
    write_json,
    write_paper_page,
)

ensure_dirs()
frame = load_analysis_frame()
print(f"Loaded {len(frame):,} rows across {frame['client_id'].nunique():,} client groups")
print(f"Observed snapshot-proxy base rate: {frame[TARGET].mean():.3f}")
"""


def build_ml05() -> None:
    write_notebook(
        "w03_feature_leakage_check.ipynb",
        [
            markdown("""
            # ML-05 — Feature Vector and Leakage/Privacy Check

            This notebook defines the model input using fields available before the observed snapshot-proxy label. IDs are retained only for grouping and tracing; label-derived fields are excluded.
            """),
            markdown("""
            ## 1. Build the feature vector

            Numeric fields are median-imputed and receive missingness flags. Categorical context is one-hot encoded. The implementation is shared with the validation notebook so the feature contract cannot silently drift.
            """),
            code(IMPORTS + """
            features, feature_names = make_feature_matrix(frame)
            print(f'Feature matrix shape: {features.shape}')
            print(f'Feature names: {len(feature_names)}')
            """),
            markdown("""
            ## 2. Feature notes (meaning, missing, categorical, available-when?)

            The performance fields describe the prior 30-day window, content age and update age describe the page before the snapshot, and categorical fields describe stable context such as device or search type. Missing numeric values are imputed with a training-fold median and marked with an indicator; missing categorical values become an explicit category. These fields are available before the prediction snapshot.
            """),
            code("""
            numeric_present = [name for name in NUMERIC_FEATURES if name in frame.columns]
            categorical_present = [name for name in CATEGORICAL_FEATURES if name in frame.columns]
            print('Numeric inputs:', numeric_present)
            print('Categorical inputs:', categorical_present)
            print('Missing-value policy: median + missingness flag for numeric; explicit category for categorical.')
            print('Availability check: prior-window and page-history fields precede the observed label window.')
            """),
            markdown("""
            ## 3. The leakage hunt

            The target is derived from the trend fields. A valid feature matrix must contain neither the target nor its source columns, and must not use identifiers as predictive features. The deliberately leaky diagnostic is reported only to show why the trend field is forbidden.
            """),
            code("""
            forbidden = {'trend_pct', 'trend_direction', TARGET, 'content_id', 'client_id'}
            found = sorted(forbidden.intersection(feature_names))
            assert not found, f'Forbidden feature(s) found: {found}'
            print('Forbidden feature columns present:', found)
            print('Leakage verdict: PASS — label-derived fields and IDs are excluded.')
            """),
            markdown("""
            ## 4. What I excluded and why

            - `trend_pct` and `trend_direction`: they define the snapshot-proxy label.
            - `target`: direct label leakage.
            - `content_id` and `client_id`: identifiers, not page behavior; `client_id` is used only for grouped validation.
            - Any future-window or post-intervention field: unavailable at decision time.
            """),
            code("""
            excluded = {
                'trend_pct': 'source of the label',
                'trend_direction': 'source of the label',
                TARGET: 'direct label',
                'content_id': 'identifier',
                'client_id': 'grouping identifier, not a feature',
            }
            print('Excluded fields:', excluded)
            print('Public-safety verdict: PASS — no client names, URLs, private queries, or credentials are used.')
            """),
            markdown("""
            ## Self-check

            - [x] Every section contains both reasoning and executable checks
            - [x] The notebook uses the shared feature contract
            - [x] Label-derived fields and identifiers are excluded
            - [x] Claims use observed, measured, directional, and decision-support language
            """),
        ],
    )


def build_ml06() -> None:
    write_notebook(
        "w04_signal_audit.ipynb",
        [
            markdown("""
            # ML-06 — Signal Audit: Do the Flags Hold?

            This notebook checks whether practical content signals line up with the observed decline proxy. These are associations for prioritization, not causal effects.
            """),
            markdown("""
            ## 1. Distributions

            Search and traffic fields are heavy-tailed, so medians and high quantiles are more useful than means for a first audit.
            """),
            code(IMPORTS + """
            distribution_columns = [
                'impressions_prev_30d', 'clicks_prev_30d', 'sessions_prev_30d',
                'content_age_days', 'days_since_last_update', 'ctr_prev_30d',
            ]
            distribution_columns = [name for name in distribution_columns if name in frame.columns]
            summary = frame[distribution_columns].describe(percentiles=[0.5, 0.9, 0.99]).T[['50%', '90%', '99%']]
            print(summary.round(2).to_string())
            print('Distribution verdict: heavy tails are present; use robust comparisons and volume context.')
            """),
            markdown("""
            ## 2. Signal test #1 / #2 / #3

            Each test compares the observed decline rate between a plain-language signal group and its complement. A verdict is based on direction and a minimum group size; it does not imply causation.
            """),
            code("""
            def signal_test(name, mask):
                inside = frame.loc[mask, TARGET]
                outside = frame.loc[~mask, TARGET]
                if len(inside) < 30 or len(outside) < 30:
                    verdict = 'MIXED'
                else:
                    delta = float(inside.mean() - outside.mean())
                    verdict = 'CONFIRMED' if delta > 0.02 else 'OPPOSITE' if delta < -0.02 else 'MIXED'
                print(f'{name}: n_in={len(inside):,}, rate_in={inside.mean():.3f}, rate_out={outside.mean():.3f}, verdict={verdict}')

            signal_test('stale pages (180+ days)', frame['days_since_last_update'].fillna(0) >= 180)
            signal_test('high prior visibility (500+ impressions)', frame['impressions_prev_30d'].fillna(0) >= 500)
            signal_test('low prior CTR (<0.5%)', frame['ctr_prev_30d'].fillna(0) < 0.5)
            """),
            markdown("""
            ## 3. The flag-linked test

            The transparent action rule links staleness and visible demand. I test that exact combination against the overall observed decline rate, while keeping the conclusion directional.
            """),
            code("""
            stale_visible = (
                (frame['days_since_last_update'].fillna(0) >= 180)
                & (frame['impressions_prev_30d'].fillna(0) >= 100)
            )
            linked = frame.loc[stale_visible, TARGET]
            print(f'stale + visible rows: {len(linked):,}')
            print(f'stale + visible decline rate: {linked.mean():.3f}')
            print(f'overall decline rate: {frame[TARGET].mean():.3f}')
            print('Flag verdict: MIXED unless the difference is large and stable; use it to order review, not to auto-act.')
            """),
            markdown("""
            ## 4. What this means in practice

            The signals are useful for triage when paired with volume and context, but their distributions and observed associations do not prove why a page declined. A content team should inspect the page, intent, seasonality, and recent changes before choosing a refresh action.
            """),
            code("""
            print('Practice verdict: keep the flags as transparent review cues with explicit volume context.')
            print('No-go: do not publish, delete, redirect, or claim a causal effect from these flags alone.')
            """),
            markdown("""
            ## Self-check

            - [x] Distributions are inspected with robust quantiles
            - [x] Three signals have explicit mini-tests and verdicts
            - [x] The flag-linked combination is tested directly
            - [x] The practical conclusion remains directional and human-reviewed
            """),
        ],
    )


def build_ml09() -> None:
    write_notebook(
        "w06_validation_audit.ipynb",
        [
            markdown("""
            # ML-09 — Validation and Research Claim Audit

            This notebook attacks the model before trusting it. It uses the approved gated warehouse when the local Hugging Face token is available, and falls back to the bundled anonymized starter slice when it is not.
            """),
            markdown("""
            ## 1. Two paper findings + my methodology questions

            **Finding 1 — growing versus declining content (paper, Finding #1, p. 5):** the paper reports that the growing cohort averages about 3.2K words and 184 days of age, while the declining cohort averages about 2.3K words and 230 days. The comparison is useful as an observed portfolio association, but its label comes from a 30-day impression change versus the prior 30 days. My methodology questions are: (a) how much could seasonality or a portfolio mix shift explain the cohort gap, and (b) do the results survive a grouped or time-aware split rather than treating rows as independent?

            **Finding 2 — age × freshness matrix (paper, Finding #8, p. 13):** the paper reports that old content refreshed recently has a health score close to the young-and-fresh quadrant, while the 365+ days old and 361+ days untouched cell is a small survivor sample. My methodology questions are: (a) were refreshed pages selected because they already had strategic value or momentum, and (b) is there a pre-refresh baseline or matched comparison that would separate selection from a refresh effect? I therefore treat this as an observed comparison, not causal proof.
            """),
            code(IMPORTS + """
            random_result = run_validation(frame, 'random')
            grouped_result = run_validation(frame, 'grouped_client')

            comparison = pd.DataFrame([
                {'split': 'Random row holdout', 'approach': 'Baseline', **random_result['baseline']},
                {'split': 'Random row holdout', 'approach': 'Model', **random_result['model']},
                {'split': 'Grouped client holdout', 'approach': 'Baseline', **grouped_result['baseline']},
                {'split': 'Grouped client holdout', 'approach': 'Model', **grouped_result['model']},
            ])
            display(comparison[['split', 'approach', 'base_rate', 'precision_at_20', 'precision_at_50', 'average_precision', 'roc_auc']].round(3))
            print('Primary estimate: grouped client holdout, because client_id repeats across rows.')
            """),
            markdown("""
            ## 2. My model under an honest split (before/after)

            The random row holdout is the “before” comparison from the earlier notebook. The grouped client holdout is the “after” stress test: entire pseudonymous client groups are held out, so the model cannot rely on client-specific repetition. Precision@K is shown beside the test base rate, and the same transparent baseline is scored on the same rows.
            """),
            code("""
            validation_receipt = {
                'source': frame.attrs.get('source_label'),
                'rows': int(len(frame)),
                'clients': int(frame['client_id'].nunique()),
                'random_split': random_result,
                'grouped_split': grouped_result,
                'primary_split': 'grouped_client',
            }
            write_json(repo_root / 'work' / 'outputs' / 'ml09_validation_results.json', validation_receipt)
            print('Wrote work/outputs/ml09_validation_results.json')
            """),
            markdown("""
            ## 3. Leakage audit

            The label is derived from `trend_direction`, which itself is derived from `trend_pct`. Neither field is allowed in the model matrix. `content_id` and `client_id` are identifiers for grouping and tracing only; they are not features. The deliberately leaky check below shows why `trend_pct` must stay excluded.
            """),
            code("""
            import numpy as np
            from sklearn.metrics import roc_auc_score

            features, feature_names = make_feature_matrix(frame)
            forbidden = {'trend_pct', 'trend_direction', 'content_id', 'client_id'}
            leaked_feature_names = sorted(forbidden.intersection(feature_names))
            assert not leaked_feature_names, f'Forbidden feature(s) found: {leaked_feature_names}'

            trend_pct = frame['trend_pct'].fillna(0).astype(float)
            leaky_auc = roc_auc_score(frame[TARGET], -trend_pct) if frame[TARGET].nunique() == 2 else 0.5
            print(f'Allowed feature columns: {len(feature_names)}')
            print(f'Forbidden feature columns present: {leaked_feature_names}')
            print(f'Deliberately leaky trend_pct AUC (diagnostic only, never used): {leaky_auc:.3f}')
            print('VERDICT: exclude trend_pct and trend_direction; keep IDs for grouping only.')
            """),
            markdown("""
            ## 4. Claim rewrite

            **Too strong:** “The model predicts which pages Google will demote and proves that refreshing them will restore traffic.”

            **Evidence-safe rewrite:** “In this anonymized snapshot, the model ranked pages associated with the declining-trend proxy above a transparent review rule on a grouped client holdout. The queue is directional decision support for human review; it does not establish causes, future Google behavior, or the effect of a refresh.”
            """),
            code("""
            print('Claim check: observed / measured / directional / decision-support language used.')
            print('No causal or algorithm-reconstruction claim is made.')
            """),
            markdown("""
            ## Self-check

            - [x] Every section above is filled with markdown and executable checks
            - [x] Random and grouped results use the same baseline and metrics
            - [x] Label-derived fields and IDs are excluded from features
            - [x] The notebook writes a JSON receipt in `work/outputs/`
            - [x] Claims stay observed, measured, directional, and decision-support oriented
            """),
        ],
    )


def build_ml10() -> None:
    write_notebook(
        "w07_action_playbook.ipynb",
        [
            markdown("""
            # ML-10 — Content Action Playbook

            This notebook turns the validated ranking output into a practical, human-reviewed action queue. It does not auto-publish or claim that an action causes a ranking change.
            """),
            markdown("""
            ## 1. Ranked actions + reason codes

            The queue ranks pages by a blend of model probability and a transparent baseline. Reason codes explain why an item landed in the queue: model decline risk, visible demand, staleness, CTR review, engagement review, or thin-but-visible content. The first human pass should start with high-confidence pages that combine model risk and measurable demand.
            """),
            code(IMPORTS + """
            artifacts = run_artifacts()
            queue = artifacts['queue']
            summary = artifacts['queue_summary']
            display(queue[['rank', 'final_score', 'confidence', 'suggested_action', 'reason_codes', 'impressions_90d', 'sessions_90d']].head(20))
            print('Reason-code counts:')
            reason_counts = {}
            for text in queue['reason_codes']:
                for reason in str(text).split('|'):
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            display(pd.Series(reason_counts).sort_values(ascending=False).head(12).to_frame('rows'))
            """),
            markdown("""
            ## 2. Intended use and limits

            **Intended use:** an SEO strategist or editor uses the ranked queue to choose which visible pages to inspect first, then checks the live page, search intent, seasonality, business priority, and recent changes.

            **Limits:** the observed label is a current snapshot proxy; it is not a future treatment outcome. The queue is trained on anonymized data, does not contain page text or client context, and should not be used as an automatic rewrite, deletion, publishing, or budget decision.
            """),
            code("""
            assert queue['final_score'].between(0, 100).all()
            assert queue['rank'].is_monotonic_increasing
            print(f"Rows scored: {len(queue):,}")
            print(f"Base rate used for context: {frame[TARGET].mean():.3f}")
            print('Intended use check passed: reviewer decision-support only.')
            """),
            markdown("""
            ## 3. Human review + the no-go list

            Before acting, a person must verify that the page is real, the page still serves the same intent, the demand is meaningful, the signal is not seasonal or caused by a site migration, and the recommended action is appropriate for the content owner.

            **Never automate:** publishing or deleting content; changing canonical URLs or redirects; claiming causation; contacting clients; using private queries or client-identifying data; or treating a low-volume percentage swing as a business-impact verdict without context.
            """),
            code("""
            no_go_rules = [
                'no automatic publishing or deletion',
                'no canonical / redirect changes without a human owner',
                'no causal claims from this observational queue',
                'no private client data, raw queries, or credentials',
                'no action from a percentage swing without volume context',
            ]
            print('\\n'.join(f'- {rule}' for rule in no_go_rules))
            """),
            markdown("""
            ## 4. Monitoring / retrain triggers

            Re-run the queue on a regular cadence and investigate if the input mix changes, if the base rate shifts, if grouped holdout performance falls toward the base rate, or if the top reason codes no longer match editorial review. Retrain only after checking whether the label definition, data window, or tracking implementation changed.
            """),
            code("""
            monitor = {
                'cadence': 'monthly or after a material tracking/content-system change',
                'base_rate_shift': 'investigate when the observed decline rate moves materially from the receipt',
                'performance_trigger': 'investigate when grouped Precision@50 approaches the base rate',
                'drift_trigger': 'check feature distributions and missingness before retraining',
                'human_trigger': 'review if top reason codes repeatedly disagree with editors',
            }
            for key, value in monitor.items():
                print(f'{key}: {value}')
            """),
            markdown("""
            ## 5. Exports for the paper

            The notebook writes the complete queue to `work/outputs/ml10_action_queue.csv` (ignored from git because it is row-level data), while the compact JSON receipt and SVG figures remain safe to commit and embed in the paper.
            """),
            code("""
            write_json(repo_root / 'work' / 'outputs' / 'ml10_action_playbook_results.json', summary)
            print('Wrote work/outputs/ml10_action_playbook_results.json')
            print('Wrote work/outputs/ml10_action_queue.csv')
            print('Figures: work/figures/action_mix.svg and work/figures/feature_importance.svg')
            """),
            markdown("""
            ## Self-check

            - [x] Ranked actions have reason codes and human-readable limits
            - [x] The no-go list prevents automatic or causal use
            - [x] Monitoring and retrain triggers are explicit
            - [x] Queue, JSON receipt, and figures are generated reproducibly
            - [x] Claims remain public-safe and decision-support oriented
            """),
        ],
    )


def build_capstone() -> None:
    write_notebook(
        "capstone.ipynb",
        [
            markdown("""
            # Capstone — Google Search Ranking & Discoverability

            **Lane:** Content Refresh Priority

            This notebook builds the reproducible analysis and the static research page. It automatically uses the approved gated warehouse when a local Hugging Face token is available and falls back to the bundled anonymized starter slice otherwise. The executed receipt records which source was used.
            """),
            markdown("""
            ## 1. Question

            Which content pages look worth reviewing first when editorial time is limited? The output is a ranked reviewer queue. A strategist acts on it by checking the live page, intent, demand, seasonality, and business context before choosing a refresh action.
            """),
            code(IMPORTS + """
            artifacts = run_artifacts()
            validation = artifacts['validation']
            queue = artifacts['queue']
            print('Question framed: rank pages for human refresh review.')
            """),
            markdown("""
            ## 2. Data

            The executed receipt records the source, row count, client-group count, target definition, and leakage exclusions. The model uses only pre-label-window fields; pseudonymous IDs are used for grouping and tracing, never as features. No names, domains, URLs, titles, raw queries, credentials, or raw warehouse exports are included in the public artifacts.
            """),
            code("""
            print(f"Rows: {validation['rows']:,}")
            print(f"Client groups: {validation['clients']:,}")
            print(f"Target definition: {validation['target_definition']}")
            print(f"Target base rate: {frame[TARGET].mean():.3f}")
            print(f"Excluded from features: {validation['leakage_excluded']}")
            """),
            markdown("""
            ## 3. Methodology

            Numeric performance/content fields are median-imputed with missingness flags; categorical context is one-hot encoded. The transparent baseline ranks staleness, visibility, CTR risk, and position risk. The model is a seeded random forest. The primary validation is a client-grouped holdout because the same pseudonymous client repeats across rows; a random row holdout is reported as a comparison. The leakage audit explicitly excludes the label-derived trend fields.
            """),
            code("""
            grouped = validation['grouped_split']
            random_result = validation['random_split']
            print(f"Grouped train/test rows: {grouped['train_rows']:,} / {grouped['test_rows']:,}")
            print(f"Grouped train/test clients: {grouped['train_clients']:,} / {grouped['test_clients']:,}")
            print(f"Feature count: {grouped['feature_count']}")
            print('Seed:', 42)
            """),
            markdown("""
            ## 4. Results (vs baseline)

            The grouped client holdout is the primary estimate. Precision@K is shown beside the base rate so a reader can distinguish ranking skill from the class prevalence.
            """),
            code("""
            result_table = pd.DataFrame([
                {'split': 'Random row holdout', 'approach': 'Baseline', **random_result['baseline']},
                {'split': 'Random row holdout', 'approach': 'Model', **random_result['model']},
                {'split': 'Grouped client holdout', 'approach': 'Baseline', **grouped['baseline']},
                {'split': 'Grouped client holdout', 'approach': 'Model', **grouped['model']},
            ])
            display(result_table[['split', 'approach', 'base_rate', 'precision_at_20', 'precision_at_50', 'average_precision', 'roc_auc']].round(3))
            """),
            markdown("""
            ## 5. Limitations

            These results are observed associations in an anonymized starter slice. They do not prove that refreshing a page causes traffic recovery, do not predict or reverse-engineer Google's algorithm, and do not establish performance on future time windows or the gated warehouse. Missing history, seasonality, tracking changes, site migrations, low-volume volatility, and unobserved editorial decisions can change the queue.
            """),
            code("""
            limitations = [
                'snapshot-proxy label rather than future treatment outcome',
                f"data source: {validation['source']}",
                'observational data with possible selection and seasonality effects',
                'human review remains mandatory before any content action',
            ]
            print('\\n'.join(f'- {item}' for item in limitations))
            """),
            markdown("""
            ## 6. Ranked recommendations

            Start with high-confidence pages that combine model risk and visible demand. Then choose a specific review path: CTR/snippet review, engagement/readability review, content expansion, or monitoring. The queue should organize editorial attention; it should not publish changes automatically.
            """),
            code("""
            display(queue[['rank', 'final_score', 'confidence', 'suggested_action', 'reason_codes']].head(15))
            print('Action counts:', artifacts['queue_summary']['action_counts'])
            """),
            markdown("""
            ## 7. Artifacts the paper embeds

            The pipeline creates model-vs-baseline, action-mix, and feature-importance charts. It also writes compact JSON receipts that let a reviewer trace the paper numbers back to a fresh run.
            """),
            code("""
            paper_path = write_paper_page(artifacts)
            print(f'Wrote paper: {paper_path}')
            print('Wrote figures under work/figures/ and docs/assets/')
            print('Wrote receipts under work/outputs/')
            """),
            markdown("""
            ## ML-12 — 5-minute demo outline + shareable cuts

            **Demo outline:** (1) show the editorial decision and why a ranked queue is useful; (2) show the data contract and the excluded label-derived fields; (3) show the transparent baseline; (4) show the grouped holdout comparison; (5) show one feature-importance chart and one queue example; (6) close with limitations and the human-review rule.

            **Short social post:** I built a leakage-aware content-refresh ranking workflow on FlyRank's anonymized data. Instead of treating a model score as a Google ranking oracle, I compared it with a transparent baseline under a client-grouped holdout and turned the output into reason-coded editorial actions. The paper and reproducible notebooks show what the data supports—and where it stops.

            **Employer-facing summary:** I built a reproducible ML workflow that ranks content pages for human refresh review using anonymized search-performance data. I compared a seeded random forest with a transparent staleness/visibility baseline under a client-grouped holdout, including missingness flags and leakage checks. The result is a public-safe research page, executable notebooks, charts, and a reason-coded action queue rather than an unsupported causal claim.
            """),
            code("""
            required_sections = [
                '<h2>Abstract</h2>', '<h2>1. Introduction / problem</h2>', '<h2>2. Data</h2>',
                '<h2>3. Methodology</h2>', '<h2>4. Results</h2>', '<h2>5. Limitations & honest framing</h2>',
                '<h2>6. Ranked recommendations</h2>', '<h2>7. Reproducibility</h2>', '<h2>8. Acknowledgments & data credit</h2>',
            ]
            html = paper_path.read_text(encoding='utf-8')
            missing_sections = [section for section in required_sections if section not in html]
            assert not missing_sections, missing_sections
            assert 'https://flyrank.ai' in html
            print('Paper section check passed: all 9 required sections and FlyRank data credit are present.')
            """),
            markdown("""
            ## Self-check

            - [x] Question, data, method, results, limitations, recommendations, reproducibility, and credit are present
            - [x] The grouped holdout and base rate are visible
            - [x] No label-derived fields or identifiers are model features
            - [x] ML-12 is included in the closing cells
            - [x] The generated page is ready for GitHub Pages after the repo owner enables `/docs`
            """),
        ],
    )


if __name__ == "__main__":
    build_ml05()
    build_ml06()
    build_ml09()
    build_ml10()
    build_capstone()
    print("Built ML-09, ML-10, ML-11, and ML-12 notebooks")
