import fs from 'node:fs';
import path from 'node:path';

import { chromium } from 'playwright';

function escapeRegex(s) {
  return String(s || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function main() {
  const baseUrl = String(process.env.CHATGPT_UI_BASE_URL || 'https://chatgpt.com/').trim();
  const outPath = String(
    process.env.CHATGPT_UI_STORAGE_STATE_OUT || path.join(process.cwd(), '.chatgpt-storageState.json'),
  ).trim();

  const outDir = path.dirname(outPath);
  fs.mkdirSync(outDir, { recursive: true });

  // Headful on purpose: you will login manually.
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();
  page.setDefaultTimeout(60_000);

  console.log(`\n[chatgpt-ui] Opening: ${baseUrl}`);
  console.log('[chatgpt-ui] Please login manually (email/password/SSO/2FA) in the opened browser window.');
  console.log('[chatgpt-ui] When you see the chat textbox, this script will auto-export storageState.');

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });

  // Wait for the logged-in chat textbox.
  const textbox = page.locator('textarea').first();
  await textbox.waitFor({ state: 'visible', timeout: 10 * 60_000 });

  // Optional: click into textbox to ensure page is fully interactive.
  await textbox.click().catch(() => {});

  await context.storageState({ path: outPath });
  console.log(`\n[chatgpt-ui] Saved storageState to: ${outPath}`);

  console.log('\nNext: create a base64 secret for GitHub Actions.');
  console.log('PowerShell (Windows):');
  console.log(`  $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes('${outPath.replace(/'/g, "''")}'))`);
  console.log('  gh secret set CHATGPT_UI_STORAGE_STATE_B64 -R rozy0311/viralops-engine -b $b64');

  console.log('\nBash (macOS/Linux):');
  console.log(`  b64=$(base64 -w0 "${outPath}")`);
  console.log('  gh secret set CHATGPT_UI_STORAGE_STATE_B64 -R rozy0311/viralops-engine -b "$b64"');

  console.log('\nYou can close the browser window now.');

  // Keep browser open a moment so user sees the message.
  await page.waitForTimeout(1500);
  await browser.close();
}

main().catch((e) => {
  console.error(String(e?.stack || e || 'Unknown error'));
  process.exit(1);
});
