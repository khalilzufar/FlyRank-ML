# Agent concepts and MCP basics

## Workflow versus agent

| Workflow | Agent |
|---|---|
| fixed sequence of steps | selects the next tool from the user intent |
| predictable inputs and outputs | can handle a bounded class of questions |
| stops when the recipe ends | loops through tools, then grounds a response |
| easiest to audit | needs stronger tool and permission boundaries |

The portfolio agent is intentionally bounded: it has three read-only tools, one public-safe case registry, and no ability to publish or mutate external state.

## What MCP adds

Model Context Protocol is a standard shape for exposing tools and resources to an AI client. A connector is the product-specific implementation; MCP is the shared interface pattern. The important design question is not “is it an agent?” but “which tools can it call, with what inputs, and what can those tools change?”

## Three working tool-call examples

1. `search_cases("content refresh")` returned the public FlyRank refresh case.
2. `recommend_next_step("flyrank-refresh")` returned a time-aware validation next step.
3. `fetch_public_page("https://khalilzufar.github.io/FlyRank-ML/")` performed a read-only public GET and returned HTTP 200 during the run capture.

These calls are recorded in `agent/run_capture.md`. The implementation uses a small local registry so it can run without a paid service; the public HTTP fetch is the live external connection. The same contracts can be wrapped by an MCP server later without changing the agent's safety rules.
