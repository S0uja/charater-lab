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
  console.log(' [S] Запустить недостающие службы    [R] Pull из GitHub    [Q] Выход');
  console.log('');
  console.log(busy ? 'Выполняется git pull…' : (lastActionAt ? '[' + lastActionAt + '] ' + lastAction : lastAction));
  console.log('');
}

async function refresh() {
  const results = await Promise.all(services.map(async (service) => [service.name, await probe(service.url)]));
  lastStates = new Map(results);
  draw();
}

function runGit(args) {
  return new Promise((resolve, reject) => {
    const child = spawn('git', args, { cwd: ROOT, shell: false, windowsHide: true });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve(stdout.trim());
      else reject(new Error((stderr || stdout || 'git завершился с кодом ' + code).trim()));
    });
  });
}

function restartAdvice(files) {
  if (!files.length) return 'Изменённых файлов нет — перезапуск не нужен.';

  const siteServer = files.some((file) => /^(server\.js|package\.json|package-lock\.json|start-site\.bat)$/.test(file));
  const browser = files.some((file) => /^(index\.html|[^/]+\.css)$/.test(file));
  const comfy = files.some((file) => /(^|\/)(comfyui|ComfyUI|workflows|workflow)(\/|$)/i.test(file) || /^(start-comfyui\.bat|extra_model_paths\.yaml)$/.test(file));
  const ollama = files.some((file) => /^(start-ai\.bat|Modelfile|ollama\.ya?ml)$/.test(file) || /(^|\/)ollama(\/|$)/i.test(file));
  const monitor = files.some((file) => /^(monitor\.js|start-monitor\.bat)$/.test(file));
  const advice = [];

  if (siteServer) advice.push('Сайт: перезапусти сервер через start-site.bat (изменились файлы сервера/запуска).');
  else if (browser) advice.push('Сайт: сервер перезапускать не нужно; обнови вкладку браузера (Ctrl+F5).');
  if (comfy) advice.push('ComfyUI: перезапусти только ComfyUI, если менялись его настройки или workflow.');
  if (ollama) advice.push('Ollama: перезапусти только Ollama, если менялись её настройки. Для одних лишь файлов модели перезапуск обычно не нужен.');
  if (monitor) advice.push('Монитор: перезапусти start-monitor.bat, чтобы загрузить новую версию самого монитора.');
  if (!advice.length) advice.push('Перезапуск сервисов не требуется для этих файлов.');

  return 'По изменённым файлам:\n' + advice.map((line) => '• ' + line).join('\n') +
    '\nПерезапускай только перечисленное и по одному компоненту; монитор ничего не перезапускает автоматически.\nФайлы: ' + files.join(', ');
}


function launchWindow(title, command) {
  const child = spawn('cmd.exe', ['/c', 'start', title, 'cmd.exe', '/k', command], {
    cwd: ROOT,
    shell: false,
    windowsHide: false,
    detached: true,
    stdio: 'ignore',
  });
  child.on('error', (error) => {
    lastAction = 'Не удалось открыть окно для запуска ' + title + ': ' + error.message;
    lastActionAt = timestamp();
    draw();
  });
  child.unref();
}

async function startDownServices() {
  if (busy) {
    lastAction = 'Подожди завершения текущей операции.';
    lastActionAt = timestamp();
    draw();
    return;
  }
  busy = true;
  lastAction = 'Проверяю службы перед запуском…';
  lastActionAt = timestamp();
  draw();
  try {
    await refresh();
    const down = services.filter((service) => !lastStates.get(service.name)?.ok);
    if (!down.length) {
      lastAction = 'Все службы уже работают — запуск не требуется.';
    } else {
      const started = [];
      for (const service of down) {
        if (service.name === 'Сайт') {
          launchWindow('LiveRoom site launcher', 'cd /d "' + ROOT + '" && call start-site.bat');
          started.push('сайт');
        } else if (service.name === 'ComfyUI') {
          launchWindow('LiveRoom ComfyUI', 'cd /d "' + ROOT + '" && call start-comfyui.bat');
          started.push('ComfyUI');
        } else if (service.name === 'Ollama') {
          let installed = false;
          const check = spawn('where.exe', ['ollama'], { cwd: ROOT, shell: false, windowsHide: true });
          installed = await new Promise((resolve) => {
            check.on('error', () => resolve(false));
            check.on('close', (code) => resolve(code === 0));
          });
          if (installed) {
            launchWindow('Ollama service', 'ollama serve');
            started.push('Ollama');
          } else {
            lastAction = 'Ollama не запущена, и команда ollama не найдена. Установи Ollama или проверь PATH.';
          }
        }
      }
      if (started.length) {
        lastAction = 'Отправил команды запуска только для неработающих служб: ' + started.join(', ') +
          '. Новые окна останутся открытыми для просмотра ошибок. Проверяю готовность…';
      }
    }
  } finally {
    busy = false;
    lastActionAt = timestamp();
    draw();
    setTimeout(() => refresh().catch(() => {}), 1500);
  }
}

async function pull() {
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

  try {
    const previousHead = await runGit(['rev-parse', 'HEAD']);
    const output = await runGit(['pull', '--ff-only']);
    const currentHead = await runGit(['rev-parse', 'HEAD']);
    const filesText = previousHead === currentHead ? '' : await runGit(['diff', '--name-only', previousHead, currentHead]).catch(() => '');
    const files = filesText ? filesText.split(/\r?\n/).filter(Boolean) : [];
    const summary = output.split(/\r?\n/).filter(Boolean).slice(-4).join(' | ');
    lastAction = (summary || 'Pull завершён успешно.') + '\n' + restartAdvice(files);
  } catch (error) {
    lastAction = 'Pull не выполнен: ' + error.message + '. Проверь Git, сеть и локальные изменения.';
  } finally {
    busy = false;
    lastActionAt = timestamp();
    draw();
    refresh().catch(() => {});
  }
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
  if (key.toLowerCase() === 's') startDownServices();
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
