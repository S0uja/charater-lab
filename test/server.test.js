'use strict';

const { spawn } = require('node:child_process');
const { once } = require('node:events');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');

const ROOT = path.resolve(__dirname, '..');
let child;
let baseUrl;
let port;

async function unusedPort() {
  const server = net.createServer();
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  const result = address.port;
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  return result;
}

async function waitForServer(url, timeoutMs = 8000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    if (child && child.exitCode !== null) {
      throw new Error('Server exited early with code ' + child.exitCode);
    }
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(500) });
      return response;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw new Error('Server did not become ready: ' + (lastError && lastError.message));
}

before(async () => {
  port = await unusedPort();
  baseUrl = 'http://127.0.0.1:' + port;
  child = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: { ...process.env, PORT: String(port), OLLAMA_URL: 'http://127.0.0.1:1', COMFY_URL: 'http://127.0.0.1:1' },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  let output = '';
  child.stdout.on('data', (chunk) => { output += chunk; });
  child.stderr.on('data', (chunk) => { output += chunk; });
  child.on('error', (error) => { output += error.message; });
  try {
    await waitForServer(baseUrl + '/');
  } catch (error) {
    throw new Error(error.message + '\n' + output);
  }
});

after(async () => {
  if (!child || child.exitCode !== null) return;
  child.kill('SIGTERM');
  await Promise.race([
    once(child, 'exit'),
    new Promise((resolve) => setTimeout(resolve, 1500)),
  ]);
  if (child.exitCode === null) child.kill('SIGKILL');
});

test('serves the home page with HTML content type', async () => {
  const response = await fetch(baseUrl + '/');
  assert.equal(response.status, 200);
  assert.match(response.headers.get('content-type') || '', /text\/html/i);
  assert.match(await response.text(), /LiveRoom|<!doctype html/i);
});

test('supports HEAD for public pages without a response body', async () => {
  const response = await fetch(baseUrl + '/index.html', { method: 'HEAD' });
  assert.equal(response.status, 200);
  assert.equal(await response.text(), '');
});

test('rejects unsupported methods on static paths', async () => {
  const response = await fetch(baseUrl + '/', { method: 'POST' });
  assert.equal(response.status, 405);
  assert.match(response.headers.get('allow') || '', /GET/);
});

test('does not expose arbitrary files', async () => {
  for (const url of ['/server.js', '/README.md', '/.git/config', '/missing-file.txt']) {
    const response = await fetch(baseUrl + url);
    assert.equal(response.status, 404, url + ' should not be public');
  }
});

test('health endpoint reports unavailable dependencies without crashing', async () => {
  const response = await fetch(baseUrl + '/api/health');
  assert.equal(response.status, 503);
  const body = await response.json();
  assert.equal(body.ok, false);
  assert.equal(body.services.ollama.ok, false);
  assert.equal(body.services.comfyui.ok, false);
});

test('rejects malformed percent-encoding in a path', async () => {
  const response = await new Promise((resolve, reject) => {
    http.get(baseUrl + '/%E0%A4%A', resolve).on('error', reject);
  });
  response.resume();
  assert.equal(response.statusCode, 400);
});

async function postJson(pathname, body) {
  return fetch(baseUrl + pathname, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
}

test('chat API rejects malformed JSON with a client error', async () => {
  const response = await fetch(baseUrl + '/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{"message":',
  });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /JSON|Unexpected|property|position/i);
});

test('chat API rejects an empty message before calling Ollama', async () => {
  const response = await postJson('/api/chat', { message: '   ' });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /пустое сообщение/i);
});

test('story turn API rejects missing required fields before calling Ollama', async () => {
  const response = await postJson('/api/story-turn', { action: 'Посмотреть вокруг' });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /стартовую сцену|главный персонаж|действие игрока/i);
});

test('image API rejects an empty prompt before contacting ComfyUI', async () => {
  const response = await postJson('/api/image', { prompt: '   ' });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /сцену для фото/i);
});

test('image status API returns a clear 404 for unknown job IDs', async () => {
  const response = await fetch(baseUrl + '/api/image/00000000-0000-4000-8000-000000000000');
  assert.equal(response.status, 404);
  const body = await response.json();
  assert.match(body.error, /задача изображения не найдена/i);
});

test('generated-image endpoint rejects path traversal attempts', async () => {
  const response = await fetch(baseUrl + '/api/generated-image?filename=..%2Fserver.js');
  assert.equal(response.status, 400);
  assert.match(await response.text(), /Invalid image path/i);
});

test('visual description translation rejects missing scene/player fields', async () => {
  const response = await postJson('/api/translate-visual-descriptions', {
    story: { starting_scene: '', player: { name: '' }, participants: [] },
  });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /стартовая сцена|главный герой/i);
});

test('visual description translation rejects duplicate participant names', async () => {
  const response = await postJson('/api/translate-visual-descriptions', {
    story: {
      starting_scene: 'Комната у окна.',
      player: { name: 'Аня' },
      participants: [{ name: 'Лена' }, { name: 'Лена' }],
    },
  });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /уникальное имя/i);
});

test('chat API returns a controlled 502 when Ollama is unavailable', async () => {
  const response = await postJson('/api/chat', {
    character: { name: 'Тестовый персонаж' },
    message: 'Привет',
    history: [],
    state: {},
  });
  assert.equal(response.status, 502);
  const body = await response.json();
  assert.match(body.error, /Ollama|модель/i);
});

test('story turn API returns a controlled 502 when Ollama is unavailable', async () => {
  const response = await postJson('/api/story-turn', {
    story: { starting_scene: 'Комната у окна.', player: { name: 'Аня' }, participants: [] },
    action: 'Осмотреть комнату',
    turns: [],
  });
  assert.equal(response.status, 502);
  assert.ok((await response.json()).error);
});

test('image API returns a controlled 502 when ComfyUI is unavailable', async () => {
  const response = await postJson('/api/image', { prompt: 'A quiet room with a window' });
  assert.equal(response.status, 502);
  assert.ok((await response.json()).error);
});

test('image API rejects oversized JSON bodies with 413', async () => {
  const response = await fetch(baseUrl + '/api/image', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ prompt: 'x'.repeat(110_000) }),
  });
  assert.equal(response.status, 413);
  assert.match((await response.json()).error, /слишком большой/i);
});

test('image API rejects malformed JSON with a client error', async () => {
  const response = await fetch(baseUrl + '/api/image', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{"prompt":',
  });
  assert.equal(response.status, 400);
  assert.ok((await response.json()).error);
});

test('API routes reject unsupported methods explicitly', async () => {
  const response = await fetch(baseUrl + '/api/chat', { method: 'PUT' });
  assert.equal(response.status, 405);
  assert.match(response.headers.get('allow') || '', /GET, HEAD/i);
});

