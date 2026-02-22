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

function b64ToFile(b64, outPath) {
  const buf = Buffer.from(String(b64 || ''), 'base64');
  fs.writeFileSync(outPath, buf);
  return outPath;
}

async function main() {
  const prompt = (readStdin() || '').trim();
  if (!prompt) {
    writeErr('No prompt on stdin');
    process.exit(2);
  }

  const baseUrl = String(process.env.CHATGPT_UI_BASE_URL || 'https://chatgpt.com/').trim();
  const modelLabel = String(process.env.CHATGPT_UI_MODEL_LABEL || 'GPT-5.2').trim();

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
  const context = statePath
    ? await browser.newContext({ storageState: statePath })
    : await browser.newContext();

  const page = await context.newPage();

  // Best-effort: reduce flakiness.
  page.setDefaultTimeout(45_000);

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });

  // Fail fast if not authenticated (textarea won't appear on logged-out screens).
  const textbox = page.locator('textarea').first();
  try {
    await textbox.waitFor({ state: 'visible', timeout: 20_000 });
  } catch {
    await browser.close();
    writeErr('Not authenticated (no chat textbox). Provide CHATGPT_UI_STORAGE_STATE_B64.');
    process.exit(10);
  }

  // Best-effort model selection.
  if (modelLabel) {
    try {
      const maybeSwitcher = page.getByRole('button', { name: /model|gpt/i }).first();
      if (await maybeSwitcher.count()) {
        await maybeSwitcher.click({ timeout: 3_000 });
        const option = page.getByRole('menuitem', { name: new RegExp(modelLabel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i') }).first();
        if (await option.count()) {
          await option.click({ timeout: 3_000 });
        } else {
          // Fallback: click any element containing the label.
          const any = page.getByText(modelLabel, { exact: false }).first();
          if (await any.count()) {
            await any.click({ timeout: 3_000 });
          }
        }
      }
    } catch {
      // ignore UI changes; continue with whatever default model is selected
    }
  }

  await textbox.click();
  await textbox.fill(prompt);
  await textbox.press('Enter');

  // Wait for an assistant message to appear.
  const assistantMsgs = page.locator('[data-message-author-role="assistant"]');
  await assistantMsgs.first().waitFor({ state: 'visible', timeout: 60_000 });

  // Grab the last assistant message and wait for it to stabilize.
  const last = assistantMsgs.last();
  let prev = '';
  let stableCount = 0;
  for (let i = 0; i < 20; i++) {
    const txt = (await last.innerText().catch(() => ''))?.trim() || '';
    if (txt && txt === prev) {
      stableCount += 1;
      if (stableCount >= 2) break;
    } else {
      stableCount = 0;
    }
    prev = txt;
    await page.waitForTimeout(800);
  }

  const out = (await last.innerText().catch(() => ''))?.trim() || '';
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
