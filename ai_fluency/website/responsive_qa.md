# Responsive and phone QA

## Checks

- viewport meta tag is present;
- cards collapse to one column below 760px;
- controls have at least a comfortable touch target and visible keyboard focus;
- navigation wraps without horizontal scrolling;
- the case explorer remains usable without hover;
- `prefers-reduced-motion` disables decorative movement;
- long URLs wrap instead of widening the page;
- empty search results explain how to recover.

## Before/after log

| Check | Before | Fix |
|---|---|---|
| narrow layout | two-column cards squeezed the evidence | single-column stack below 760px |
| keyboard focus | default focus was low contrast | explicit yellow focus ring |
| empty search | no guidance | “No matching cases — clear the filter” message |
| link overflow | long repo URL could widen a card | `overflow-wrap:anywhere` |

The page is static and can be opened directly on a phone at the public URL; no app install is required.
