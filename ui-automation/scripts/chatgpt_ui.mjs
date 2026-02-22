import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { chromium } from 'playwright';

function readStdin() {
  try {
    return fs.readFileSync(0, 'utf8');
  } catch {
    return '';
  }
}

function writeErr(msg) {
  try {
    process.stderr.write(String(msg || '') + '\n');
  } catch {
    // ignore
  }
}

function safeWriteFile(filePath, content) {
  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, content);
    return true;
  } catch {
    return false;
  }
}

function b64ToFile(b64, outPath) {
  const buf = Buffer.from(String(b64 || ''), 'base64');
  fs.writeFileSync(outPath, buf);
  return outPath;
}

function escapeRegex(s) {
  return String(s || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function main() {
  const prompt = (readStdin() || '').trim();
  if (!prompt) {
    writeErr('No prompt on stdin');
    process.exit(2);
  }

  const baseUrl = String(process.env.CHATGPT_UI_BASE_URL || 'https://chatgpt.com/').trim();
  const modelLabel = String(process.env.CHATGPT_UI_MODEL_LABEL || 'GPT-5.2').trim();

  const debugEnabled = String(process.env.CHATGPT_UI_DEBUG || '').trim().toLowerCase() in {
    '1': true,
    true: true,
    yes: true,
    on: true,
  };
  const debugDir = String(process.env.CHATGPT_UI_DEBUG_DIR || path.join(process.cwd(), 'artifacts')).trim();

  const storageStateB64 = String(process.env.CHATGPT_UI_STORAGE_STATE_B64 || '').trim();
  const storageStatePath = String(process.env.CHATGPT_UI_STORAGE_STATE_PATH || '').trim();

  let statePath = '';
  if (storageStatePath && fs.existsSync(storageStatePath)) {
    statePath = storageStatePath;
  } else if (storageStateB64) {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'chatgpt-state-'));
    statePath = b64ToFile(storageStateB64, path.join(tmpDir, 'storageState.json'));
  }

  const browser = await chromium.launch({ headless: true });
  const context = statePath ? await browser.newContext({ storageState: statePath }) : await browser.newContext();

  const page = await context.newPage();
  page.setDefaultTimeout(45_000);

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });

  const textbox = page
    .locator('textarea, div[contenteditable="true"][role="textbox"], div[contenteditable="true"][aria-label]')
    .first();

  try {
    await textbox.waitFor({ state: 'visible', timeout: 20_000 });
  } catch {
    let isCloudflareChallenge = false;
    try {
      const title = String((await page.title().catch(() => '')) || '').toLowerCase();
      const url = String(page.url() || '').toLowerCase();
      const html = String((await page.content().catch(() => '')) || '').toLowerCase();
      isCloudflareChallenge =
        title.includes('just a moment') ||
        url.includes('__cf_chl') ||
        html.includes('cdn-cgi/challenge-platform') ||
        html.includes('challenges.cloudflare.com') ||
        html.includes('turnstile');
    } catch {
      // ignore
    }

    if (debugEnabled) {
      try {
        const ts = Date.now();
        const shot = path.join(debugDir, `chatgpt_ui_auth_fail_${ts}.png`);
        const html = path.join(debugDir, `chatgpt_ui_auth_fail_${ts}.html`);
        await page.screenshot({ path: shot, fullPage: true }).catch(() => {});
        const content = await page.content().catch(() => '');
        safeWriteFile(html, content || '');
        writeErr(`Debug saved: ${shot}`);
        writeErr(`Debug saved: ${html}`);
        writeErr(`Current URL: ${page.url()}`);
      } catch {
        // ignore debug failures
      }
    }

    await browser.close();

    if (isCloudflareChallenge) {
      writeErr(
        'ChatGPT UI blocked by Cloudflare/Turnstile challenge on this runner/IP. Use a self-hosted runner or expect fallback providers.',
      );
    }
    writeErr('Not authenticated (no chat textbox). Provide CHATGPT_UI_STORAGE_STATE_B64.');
    process.exit(10);
  }

  if (modelLabel) {
    try {
      const maybeSwitcher = page.getByRole('button', { name: /model|gpt/i }).first();
      if (await maybeSwitcher.count()) {
        await maybeSwitcher.click({ timeout: 3_000 });
        const option = page.getByRole('menuitem', { name: new RegExp(escapeRegex(modelLabel), 'i') }).first();

        if (await option.count()) {
          await option.click({ timeout: 3_000 });
        } else {
          const any = page.getByText(modelLabel, { exact: false }).first();
          if (await any.count()) {
            await any.click({ timeout: 3_000 });
          }
        }
      }
    } catch {
      // ignore UI changes; continue with default model
    }
  }

  await textbox.click();

  const tag = String((await textbox.evaluate((el) => el?.tagName || '').catch(() => '')) || '').toLowerCase();
  if (tag === 'textarea') {
    await textbox.fill(prompt);
    await textbox.press('Enter');
  } else {
    await textbox.type(prompt, { delay: 5 });
    await page.keyboard.press('Enter');
  }

  const assistantMsgs = page.locator('[data-message-author-role="assistant"]');
  await assistantMsgs.first().waitFor({ state: 'visible', timeout: 60_000 });

  const last = assistantMsgs.last();
  let prev = '';
  let stableCount = 0;
  for (let i = 0; i < 20; i++) {
    const txt = String((await last.innerText().catch(() => '')) || '').trim();
    if (txt && txt === prev) {
      stableCount += 1;
      if (stableCount >= 2) break;
    } else {
      stableCount = 0;
    }
    prev = txt;
    await page.waitForTimeout(800);
  }

  const out = String((await last.innerText().catch(() => '')) || '').trim();
  await browser.close();

  if (!out) {
    writeErr('Empty assistant response');
    process.exit(11);
  }

  process.stdout.write(out);
}

main().catch((e) => {
  writeErr(String(e?.stack || e || 'Unknown error'));
  process.exit(1);
});
