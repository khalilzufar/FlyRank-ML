# Personal agent design

## Job to be done

Help Khalil decide what to improve next in a public portfolio case study, using only the repository's public-safe case metadata and an optional live check of a public page.

## User input

- a natural-language question such as “what should I improve next?” or “check the public paper”;
- optionally, a public HTTP(S) URL to inspect.

## Tool registry

| Tool | Input | Output | Side effect |
|---|---|---|---|
| `search_cases` | query string | matching public case records | none |
| `recommend_next_step` | case id | one bounded next step and rationale | none |
| `fetch_public_page` | public HTTP(S) URL | status, title, byte count | GET only; no credentials |

## Control loop

```text
question → classify intent → select the smallest tool set → call tools
         → ground answer in returned fields → state uncertainty → answer
```

## Guardrails

- Only `http` and `https` URLs are accepted.
- The fetch tool sends a GET request with a neutral user agent and never follows a user-provided credential.
- No email, token, cookie, private file, raw warehouse row, or client identifier is accepted as an input.
- The agent cannot publish, delete, edit, or submit anything.
- If evidence is missing, it says “not verified” instead of inventing a result.

## Success criteria

1. A question is answered with at least one evidence field.
2. A public-page check reports its actual HTTP status.
3. The answer contains a limitation or next action.
4. The same input produces a deterministic answer apart from live HTTP status.
