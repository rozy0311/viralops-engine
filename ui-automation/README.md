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

## ChatGPT UI (GPT-5.2) login for Actions

GitHub Actions cannot complete interactive login (SSO/2FA/CAPTCHA). To run the
`chatgpt_ui` provider, export a Playwright `storageState.json` locally and store
it as a base64 GitHub Actions secret.

### 1) Export storageState locally (headful)

From `ui-automation/`:

```bash
npm ci
npm run install:browsers
npm run chatgpt:login
```

This opens a real browser window. Login manually. When the chat textbox appears,
the script saves `.chatgpt-storageState.json`.

### 2) Set GitHub Secret

PowerShell (Windows):

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes(".chatgpt-storageState.json"))
gh secret set CHATGPT_UI_STORAGE_STATE_B64 -R rozy0311/viralops-engine -b $b64
```

Or set it via GitHub UI: Repo → Settings → Secrets and variables → Actions.
