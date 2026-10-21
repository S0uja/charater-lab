'use strict';

const { after, before, test } = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');

const port = 43000 + Math.floor(Math.random() * 15000);
const base = `http://127.0.0.1:${port}`;
let server;

before(async () => {
  server = spawn(process.execPath, ['server.js'], {
    cwd: process.cwd(),
    env: { ...process.env, PORT: String(port) },
    stdio: 'ignore',
  });
  const deadline = Date.now() + 15000;
  let lastError;
  while (Date.now() < deadline) {
    if (server.exitCode !== null) throw new Error(`Server exited early with code ${server.exitCode}`);
    try {
      const response = await fetch(base, { signal: AbortSignal.timeout(1000) });
      if (response.ok) return;
      lastError = new Error(`Unexpected startup status: ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`Server did not start: ${lastError?.message || 'timeout'}`);
});

after(() => {
  if (server && server.exitCode === null) server.kill();
});

test('serves the home page', async () => {
  const response = await fetch(base);
  assert.equal(response.status, 200);
  assert.match(response.headers.get('content-type') || '', /text\/html/);
  assert.match(await response.text(), /Какую историю начнём/);
});

test('serves all app pages', async () => {
  for (const page of ['characters.html', 'room.html', 'visual-novel.html']) {
    const response = await fetch(`${base}/${page}`);
    assert.equal(response.status, 200, page);
    assert.match(response.headers.get('content-type') || '', /text\/html/, page);
  }
});

test('health endpoint returns service diagnostics even when models are offline', async () => {
  const response = await fetch(`${base}/api/health`);
  assert.ok([200, 503].includes(response.status));
  const body = await response.json();
  assert.equal(typeof body.ok, 'boolean');
  assert.equal(typeof body.services?.ollama?.ok, 'boolean');
  assert.equal(typeof body.services?.comfyui?.ok, 'boolean');
});

test('does not expose server source or project metadata over HTTP', async () => {
  for (const file of ['server.js', 'README.md', '.gitignore', 'start-site.bat']) {
    const response = await fetch(`${base}/${file}`);
    assert.equal(response.status, 404, file);
  }
});

test('rejects unsupported methods on static pages', async () => {
  const response = await fetch(base, { method: 'POST' });
  assert.equal(response.status, 404);
});
