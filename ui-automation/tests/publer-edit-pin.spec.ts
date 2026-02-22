import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { loadPinterestManualEditKit } from '../src/kit.js';

test('Publer: edit Pinterest pin fields (best-effort)', async ({ page }) => {
  const email = (process.env.PUBLER_EMAIL ?? '').trim();
  const password = (process.env.PUBLER_PASSWORD ?? '').trim();
  const editUrl = (process.env.PUBLER_EDIT_URL ?? '').trim();
  const baseUrl = (process.env.PUBLER_BASE_URL ?? 'https://app.publer.io').trim();

  test.skip(!email || !password, 'PUBLER_EMAIL / PUBLER_PASSWORD not set');
  test.skip(!editUrl, 'PUBLER_EDIT_URL not set (provide the exact edit link for the Publer post)');

  const kitPath = (process.env.PINTEREST_KIT_PATH ?? '').trim() || path.resolve(process.cwd(), '..', 'pinterest_manual_edit_last.txt');
  test.skip(!fs.existsSync(kitPath), `Pinterest manual edit kit not found at: ${kitPath}`);
  const kit = loadPinterestManualEditKit(kitPath);

  // 1) Login
  await page.goto(`${baseUrl}/login`, { waitUntil: 'domcontentloaded' });

  // These selectors are intentionally broad; if Publer changes UI, we'll still capture trace.
  await page.getByRole('textbox', { name: /email/i }).fill(email);
  await page.getByRole('textbox', { name: /password/i }).fill(password);
  await page.getByRole('button', { name: /log in|sign in/i }).click();

  // 2) Go to edit URL
  await page.goto(editUrl, { waitUntil: 'domcontentloaded' });

  // 3) Fill fields (best-effort). Different Publer accounts may label these differently.
  const tryFill = async (label: RegExp, value: string) => {
    if (!value) return;
    const loc = page.getByLabel(label);
    if (await loc.count()) {
      await loc.first().fill(value);
      return;
    }

    // Fallback: placeholder search
    const byPlaceholder = page.getByPlaceholder(label);
    if (await byPlaceholder.count()) {
      await byPlaceholder.first().fill(value);
    }
  };

  await tryFill(/title/i, kit.title);
  await tryFill(/alt/i, kit.altText);
  await tryFill(/link|url/i, kit.link);

  // Description is often a textarea.
  if (kit.description) {
    const desc = page.getByLabel(/description|caption/i);
    if (await desc.count()) {
      await desc.first().fill(kit.description);
    }
  }

  // 4) Save/update
  const saveBtn = page.getByRole('button', { name: /save|update|schedule/i });
  await expect(saveBtn.first()).toBeVisible();
  await saveBtn.first().click();

  // 5) Quick confirmation: stay on page without an obvious error toast.
  const errorToast = page.locator('[role="alert"]').filter({ hasText: /error|failed/i });
  expect(await errorToast.count()).toBe(0);
});
