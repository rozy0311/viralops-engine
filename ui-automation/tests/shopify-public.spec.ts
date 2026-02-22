import { test, expect } from '@playwright/test';

test('Shopify public article loads', async ({ page }) => {
  const url = (process.env.SHOPIFY_PUBLIC_URL ?? '').trim();
  test.skip(!url, 'SHOPIFY_PUBLIC_URL not set');

  const titleHint = (process.env.SHOPIFY_EXPECT_TITLE ?? '').trim();

  const resp = await page.goto(url, { waitUntil: 'domcontentloaded' });
  expect(resp?.ok(), `HTTP not OK for ${url}`).toBeTruthy();

  // Basic sanity checks: should have a title and some content.
  await expect(page.locator('h1')).toBeVisible();

  if (titleHint) {
    await expect(page).toHaveTitle(new RegExp(titleHint.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i'));
  }
});
