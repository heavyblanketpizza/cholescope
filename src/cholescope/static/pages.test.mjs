import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';

const PAGES = ['index.html', 'student.html'];
const read = (name) => readFileSync(new URL(name, import.meta.url), 'utf8');
const escape = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function modules(entry, seen = new Set()) {
  if (seen.has(entry)) return seen;
  seen.add(entry);
  for (const [, name] of read(entry).matchAll(/from '\.\/([\w-]+\.mjs)'/g)) modules(name, seen);
  return seen;
}

// The markup of one element, found by id, including its children.
function elementHTML(html, id) {
  const at = html.indexOf(` id="${id}"`);
  const start = html.lastIndexOf('<', at);
  const tag = html.slice(start + 1).match(/^[\w-]+/)[0];
  const tags = new RegExp(`<(/?)${tag}[\\s>]`, 'g');
  tags.lastIndex = start;
  for (let depth = 0, match; (match = tags.exec(html));) {
    depth += match[1] ? -1 : 1;
    if (!depth) return html.slice(start, match.index);
  }
  return html.slice(start);
}

test('each page loads only the shared stylesheet', () => {
  assert.ok(existsSync(new URL('style.css', import.meta.url)));
  for (const page of PAGES) {
    const sheets = [...read(page).matchAll(/<link rel="stylesheet" href="\/static\/([\w.-]+)">/g)].map((match) => match[1]);
    assert.deepEqual(sheets, ['style.css'], page);
  }
});

test('every element a page script looks up exists on that page', () => {
  for (const page of PAGES) {
    const html = read(page);
    const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]));
    const [, entry] = html.match(/<script type="module" src="\/static\/([\w-]+\.mjs)">/);
    for (const name of modules(entry)) {
      const source = read(name);
      for (const [, id] of source.matchAll(/(?:\$|getElementById)\('([\w-]+)'\)/g)) {
        assert.ok(ids.has(id), `${page} has no #${id} for ${name}`);
      }
      for (const [, pattern] of source.matchAll(/\$\(`([^`]+)`\)/g)) {
        const shape = new RegExp(`^${pattern.split(/\$\{[^}]+\}/).map(escape).join('[\\w-]+')}$`);
        assert.ok([...ids].some((id) => shape.test(id)), `${page} has no element like ${pattern} for ${name}`);
      }
      for (const [, id, selector] of source.matchAll(/\$\('([\w-]+)'\)\.querySelector\('([^']+)'\)/g)) {
        const inner = elementHTML(html, id);
        const base = selector.replace(/:[\w-]+$/, '');
        const found = base.startsWith('.')
          ? new RegExp(`class="[^"]*\\b${escape(base.slice(1))}\\b`).test(inner)
          : /^[a-z]+$/.test(base) && new RegExp(`<${base}[\\s>]`).test(inner.slice(1));
        assert.ok(found, `${page} #${id} has no ${selector} for ${name}`);
      }
    }
  }
});
