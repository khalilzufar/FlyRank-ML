# AI Fluency portfolio package

This folder contains the AI Fluency work built around the FlyRank ML case study.
The public showcase is deployed at:

<https://khalilzufar.github.io/FlyRank-ML/ai/>

## What is included

- `workflows/automation_workflow.md` — a repeatable research-to-story workflow with prompts, five dry runs, time-saved estimates, and failure points.
- `agent/` — a standard-library personal agent with a small tool registry, a live public-page fetch tool, design notes, and a raw run capture.
- `mcp/agent_concepts.md` — a plain-language workflow/agent/MCP explainer and three tool-call examples.
- `website/` — build notes, DNS walkthrough, responsive QA, self-review, break testing, analytics boundary, retrospective, and hours log.
- `submissions/ai_fluency_map.md` — a map from each remaining portal item to its evidence.

## Public-safe boundary

The site uses only public project information and the public GitHub/Paper links. It does not expose email, client names, private queries, domains, credentials, or raw warehouse exports. The optional analytics instrumentation is first-party and local-only; no third-party analytics account or tracking identifier is claimed. The official FlyRank graduate badge is shown as pending because badge issuance is controlled by capstone review.

## Run locally

```powershell
python ai_fluency/agent/personal_agent.py --query "What should I improve next?"
python ai_fluency/agent/personal_agent.py --query "Check the public paper" --site-url https://khalilzufar.github.io/FlyRank-ML/
```

The portfolio itself is static and can be served from the repository root with any static server. GitHub Pages publishes `docs/` from `main`.
