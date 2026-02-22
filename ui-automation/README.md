# ViralOps UI Automation (Playwright)

This folder contains **non-stealth** Playwright automation intended to run locally or in GitHub Actions.

## What it does

- **Publer (best-effort):** logs in and opens a given edit URL, then fills Pinterest fields using `pinterest_manual_edit_last.txt`.
- **Shopify (public):** checks a public article URL loads and has an `h1`.

## Environment variables

### Publer
- `PUBLER_EMAIL`
- `PUBLER_PASSWORD`
- `PUBLER_EDIT_URL` (the exact edit URL of the post inside Publer)
- `PUBLER_BASE_URL` (optional, default: `https://app.publer.io`)
- `PINTEREST_KIT_PATH` (optional, default: `../pinterest_manual_edit_last.txt`)

### Shopify
- `SHOPIFY_PUBLIC_URL`
- `SHOPIFY_EXPECT_TITLE` (optional)

## Local run

```bash
cd ui-automation
npm ci
npx playwright install --with-deps
npm test
```
