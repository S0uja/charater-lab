'use strict';

const http = require('node:http');
const https = require('node:https');
const { spawn } = require('node:child_process');
const readline = require('node:readline');

const ROOT = __dirname;
const REFRESH_MS = 2500;
const REQUEST_TIMEOUT_MS = 1800;
const services = [
  { name: 'Сайт', url: 'http://127.0.0.1:4173/', detail: '127.0.0.1:4173' },
  { name: 'ComfyUI', url: 'http://127.0.0.1:8188/system_stats', detail: '127.0.0.1:8188' },
  { name: 'Ollama', url: 'http://127.0.0.1:11434/api/tags', detail: '127.0.0.1:11434' },
];

let busy = false;
let lastAction = 'Ожидание действий.';
let lastActionAt = '';
let lastStates = new Map();
let stopped = false;

function timestamp() {
  return new Date().toLocaleTimeString('ru-RU', { hour12: false });
}

function probe(url) {
  return new Promise((resolve) => {
    const client = url.startsWith('https:') ? https : http;
    const request = client.get(url, { timeout: REQUEST_TIMEOUT_MS }, (response) => {
      response.resume();
      resolve({ ok: response.statusCode >= 200 && response.statusCode < 400, status: response.statusCode });
    });
    request.on('timeout', () => request.destroy(new Error('timeout')));
    request.on('error', (error) => resolve({ ok: false, error: error.code === 'ECONNREFUSED' ? 'не запущен' : error.message }));
  });
}

function clearScreen() {
  if (process.stdout.isTTY) process.stdout.write('\x1b[2J\x1b[H');
}

function draw() {
  if (stopped) return;
  clearScreen();
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║                    LIVEROOM MONITOR                         ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log('Проверка каждые ' + (REFRESH_MS / 1000).toFixed(1) + ' сек.  •  ' + timestamp());
  console.log('');
  for (const service of services) {
    const state = lastStates.get(service.name);
    const status = !state ? '[ CHECKING ]' : state.ok ? '[  WORK   ]' : '[  DOWN   ]';
    const extra = !state ? 'проверка…' : state.ok ? 'HTTP ' + state.status : (state.error || 'HTTP ' + state.status);
    console.log(status.padEnd(13) + ' ' + service.name.padEnd(10) + ' ' + service.detail.padEnd(23) + ' ' + extra);
  }
  console.log('');
  console.log(' [R] Pull из GitHub    [Q] Выход');
  console.log('');
  console.log(busy ? 'Выполняется git pull…' : (lastActionAt ? '[' + lastActionAt + '] ' + lastAction : lastAction));
  console.log('');
}

async function refresh() {
  const results = await Promise.all(services.map(async (service) => [service.name, await probe(service.url)]));
  lastStates = new Map(results);
  draw();
}

function pull() {
  if (busy) {
    lastAction = 'git pull уже выполняется — дождись завершения.';
    lastActionAt = timestamp();
    draw();
    return;
  }
  busy = true;
  lastAction = 'Запускаю git pull --ff-only…';
  lastActionAt = timestamp();
  draw();

  const child = spawn('git', ['pull', '--ff-only'], {
    cwd: ROOT,
    shell: false,
    windowsHide: true,
  });

  let output = '';
  child.stdout.setEncoding('utf8');
  child.stderr.setEncoding('utf8');
  child.stdout.on('data', (chunk) => { output += chunk; });
  child.stderr.on('data', (chunk) => { output += chunk; });
  child.on('error', (error) => {
    busy = false;
    lastAction = 'Не удалось запустить Git: ' + error.message + '. Проверь, установлен ли Git и доступен ли он в PATH.';
    lastActionAt = timestamp();
    draw();
  });
  child.on('close', (code) => {
    busy = false;
    const lines = output.trim().split(/\r?\n/).filter(Boolean);
    const summary = lines.slice(-5).join(' | ');
    lastAction = code === 0
      ? 'Pull завершён успешно. ' + (summary || 'Рабочая ветка уже актуальна.')
      : 'Pull завершился с кодом ' + code + '. ' + (summary || 'Проверь Git, сеть и локальные изменения.');
    lastActionAt = timestamp();
    draw();
    refresh().catch(() => {});
  });
}

function onKey(key) {
  if (key === '\u0003' || key.toLowerCase() === 'q') {
    stopped = true;
    if (process.stdin.isTTY) process.stdin.setRawMode(false);
    process.stdin.pause();
    console.log('\nМонитор закрыт.');
    process.exit(0);
  }
  if (key.toLowerCase() === 'r') pull();
}

async function main() {
  if (!process.stdin.isTTY) {
    console.error('Запусти монитор в отдельном терминале, чтобы работали клавиши R и Q.');
    process.exitCode = 1;
    return;
  }
  process.stdin.setEncoding('utf8');
  readline.emitKeypressEvents(process.stdin);
  process.stdin.setRawMode(true);
  process.stdin.resume();
  process.stdin.on('data', onKey);

  draw();
  await refresh();
  const timer = setInterval(() => refresh().catch(() => {}), REFRESH_MS);
  process.on('exit', () => clearInterval(timer));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
