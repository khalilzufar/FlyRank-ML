# Break-your-own-site test

## Tested cases

| Test | Expected behavior | Current result |
|---|---|---|
| empty search | clear explanation and recovery action | fixed in the case explorer |
| unknown filter | zero-state instead of broken layout | fixed |
| JavaScript disabled | core links and paper remain readable | known limitation: interactive filtering is unavailable |
| slow network | page remains static and usable | fixed by avoiding external runtime dependencies |
| long title or URL | content wraps | fixed with overflow rules |
| keyboard-only navigation | visible focus and logical order | fixed |
| third-party analytics blocked | no error and no loss of core content | fixed; instrumentation is optional |

Basic SEO/meta coverage is present in `docs/ai/index.html`: title, description, canonical URL, Open Graph title/description, language, and a descriptive heading hierarchy.
