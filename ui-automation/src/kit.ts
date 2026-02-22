import fs from 'node:fs';

export type PinterestKit = {
  title: string;
  altText: string;
  link: string;
  description: string;
};

export function loadPinterestManualEditKit(filePath: string): PinterestKit {
  const raw = fs.readFileSync(filePath, 'utf-8');

  const getLine = (prefix: string) => {
    const re = new RegExp(`^${prefix}:\\s*(.*)$`, 'm');
    const m = raw.match(re);
    return (m?.[1] ?? '').trim();
  };

  const title = getLine('Title');
  const altText = getLine('Alt text');
  const link = getLine('Link');

  const descMatch = raw.match(/^Description:\s*\n([\s\S]*)$/m);
  const description = (descMatch?.[1] ?? '').trim();

  if (!title || !link) {
    throw new Error(`Invalid kit file: missing Title/Link in ${filePath}`);
  }

  return { title, altText, link, description };
}
