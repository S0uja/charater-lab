'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { randomUUID } = require('node:crypto');

const root = __dirname;
const port = Number(process.env.PORT || 4173);
const comfy = process.env.COMFY_URL || 'http://127.0.0.1:8188';
const ollama = process.env.OLLAMA_URL || 'http://127.0.0.1:11434';
const chatModel = process.env.OLLAMA_MODEL || 'hf.co/mradermacher/Katarau-9B-ru-RP-nsfw-GGUF:Q4_K_M';
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp' };
const jobs = new Map();
const JOB_TTL_MS = 30 * 60 * 1000;
const MAX_TRACKED_IMAGE_JOBS = 8;
let pendingImageSubmissions = 0;
const PUBLIC_PAGES = new Set(['index.html', 'characters.html', 'room.html', 'visual-novel.html']);
const PUBLIC_ASSET = /^assets\/studio\/[a-z0-9_-]+\.(?:jpe?g|png|webp)$/i;

function json(res, status, data) {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
  res.end(JSON.stringify(data));
}

async function readJson(req, maxBytes = 3 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > maxBytes) throw Object.assign(new Error('Запрос слишком большой.'), { status: 413 });
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function pruneExpiredImageJobs(now = Date.now()) {
  for (const [jobId, job] of jobs) {
    if (now - job.createdAt > JOB_TTL_MS) jobs.delete(jobId);
  }
}

async function comfyFetch(url, options = {}) {
  return fetch(`${comfy}${url}`, { ...options, signal: AbortSignal.timeout(options.timeout || 15_000) });
}

function buildImageWorkflow(promptText, prefix, negativeText = '', fixedSeed = null) {
  return {
    '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'lustifySDXLNSFWSFW_v20.safetensors' } },
    '2': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: promptText } },
    '3': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: ['low quality, blurry, oversharpened, jpeg artifacts, bad anatomy, malformed hands, extra fingers, duplicate people, text, watermark, logo', negativeText].filter(Boolean).join(', ') } },
    '4': { class_type: 'EmptyLatentImage', inputs: { width: 832, height: 1216, batch_size: 1 } },
    '5': { class_type: 'KSampler', inputs: { model: ['1', 0], seed: Number.isInteger(fixedSeed) ? fixedSeed >>> 0 : Math.floor(Math.random() * 2 ** 32), steps: 30, cfg: 5.5, sampler_name: 'dpmpp_2m_sde', scheduler: 'karras', positive: ['2', 0], negative: ['3', 0], latent_image: ['4', 0], denoise: 1 } },
    '6': { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } },
    '7': { class_type: 'SaveImage', inputs: { images: ['6', 0], filename_prefix: prefix } },
  };
}

function parseJsonText(text) {
  try { return JSON.parse(text); } catch {}
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start >= 0 && end > start) {
    try { return JSON.parse(text.slice(start, end + 1)); } catch {}
  }
  return null;
}

function describesOfflineInteraction(reply) {
  return /\b(?:садись|присаживайся|заходи|приходи)\b|(?:нали(?:ла|л|ваю)|принес(?:у|ла)|переда(?:ю|ла)|держи)\s+(?:тебе\s+)?(?:чашк\w*|кофе|чай|ед\w*|торт\w*)|\b(?:обниму|поцелую|коснусь|посажу)\s+тебя|пока\s+мы\s+(?:поговорим|говорим)/iu.test(reply);
}

function usesWrongPersonaName(reply, characterName) {
  const match = String(reply || '').match(/(?:меня\s+зовут|я\s*[—-])\s*([\p{L}\p{N}-]+)/iu);
  if (!match) return false;
  const claimedName = normalizeReply(match[1]);
  const expectedName = normalizeReply(characterName).split(' ')[0];
  return Boolean(claimedName && expectedName && claimedName !== expectedName);
}

function normalizeReply(reply) {
  return String(reply || '').toLocaleLowerCase('ru-RU').replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
}

function repeatsRecentReply(reply, recentReplies) {
  const candidate = normalizeReply(reply);
  if (candidate.length < 18) return false;
  return recentReplies.some((previous) => {
    const normalized = normalizeReply(previous);
    return normalized.length >= 18 && (candidate === normalized || (Math.min(candidate.length, normalized.length) / Math.max(candidate.length, normalized.length) > 0.75 && (candidate.startsWith(normalized.slice(0, 45)) || normalized.startsWith(candidate.slice(0, 45)))));
  });
}

function outfitChanged(text) {
  return /(?:переодел(?:ась|ся)?|одел(?:ась|ся)?|раздел(?:ась|ся)?|снял[а]?\s+(?:с\s+себя\s+)?(?:одежду|платье|рубашку|футболку|бель[её])|надел[а]?\s+(?:на\s+себя\s+)?(?:одежду|платье|рубашку|футболку|бель[её])|сменил[а]?\s+одежду|раздеваюсь|одеваюсь|переодеваюсь|сбросил[а]?\s+одежду|(?:теперь|сейчас|пусть|на тебе).{0,32}(?:голая|голый|обнаж[её]нн\w*|без одежды|одета в|одет в|рубашк\w*|футболк\w*|плать\w*|юбк\w*|бель[её])|(?:я|она)\s+(?:сейчас\s+)?(?:голая|обнаж[её]нная|без одежды)|put on (?:a |the )?(?:shirt|dress|clothes)|took off (?:her |my )?(?:clothes|shirt|dress)|got dressed|undressed|now wearing|wearing a (?:shirt|dress))/iu.test(String(text || ''));
}

function keepGroundedMemoryFacts(facts, userMessage) {
  const generic = new Set('пользователь собеседник я мне мой моя мое мои меня моего моей у он она его ее мы вы твой твоя твое ваши запомни также теперь сегодня вчера люблю нравится хочу факт пользовательский'.split(' '));
  const tokenize = (text) => String(text || '').toLocaleLowerCase('ru-RU').replace(/ё/g, 'е').match(/[\p{L}\p{N}]{3,}/gu) || [];
  const evidence = tokenize(userMessage);
  return (Array.isArray(facts) ? facts : []).filter((fact) => {
    if (typeof fact !== 'string') return false;
    return tokenize(fact).filter((word) => !generic.has(word)).every((word) => {
      const stem = word.slice(0, Math.min(word.length, 4));
      return evidence.some((sourceWord) => sourceWord === word || sourceWord.startsWith(stem) || (stem.length >= 4 && word.startsWith(sourceWord.slice(0, 4))));
    });
  }).slice(0, 2).map((fact) => fact.slice(0, 240));
}

function extractExplicitMemoryFact(userMessage) {
  const match = String(userMessage || '').match(/(?:запомни|запиши|не забудь)(?:\\s*,?\\s*пожалуйста)?(?:\\s*[:,—–-]\\s*|\\s*,?\\s+(?:что\\s+)?)(.{4,500})$/iu);
  if (!match) return '';
  return match[1].trim().replace(/^[«“"]|[»”"]$/gu, '').replace(/[.!?]+$/u, '').trim();
}

async function requestRpCompletion(messages) {
  const response = await fetch(`${ollama}/api/chat`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: chatModel, stream: false, format: rpSchema, messages, options: { temperature: 0.82, top_p: 0.92, num_ctx: 8192, num_predict: 650 } }),
    signal: AbortSignal.timeout(180_000),
  });
  const result = await response.json();
  if (!response.ok) throw Object.assign(new Error(result.error || 'Локальная модель не ответила.'), { status: response.status });
  const parsed = parseJsonText(result.message?.content || '');
  if (!parsed || typeof parsed.reply !== 'string') throw Object.assign(new Error('Модель вернула ответ в неверном формате. Повтори сообщение.'), { status: 502 });
  return parsed;
}

const rpSchema = {
  type: 'object',
  properties: {
    reply: { type: 'string' },
    mood: { type: 'string', enum: ['calm', 'happy', 'curious', 'shy', 'affectionate', 'tired', 'upset', 'playful'] },
    relationship_delta: { type: 'integer', minimum: -2, maximum: 2 },
    relationship_changes: { type: 'object', properties: { trust: { type: 'integer', minimum: -2, maximum: 2 }, romance: { type: 'integer', minimum: -2, maximum: 2 }, affection: { type: 'integer', minimum: -2, maximum: 2 }, comfort: { type: 'integer', minimum: -2, maximum: 2 }, respect: { type: 'integer', minimum: -2, maximum: 2 }, tension: { type: 'integer', minimum: -2, maximum: 2 }, familiarity: { type: 'integer', minimum: -2, maximum: 2 } }, required: ['trust', 'romance', 'affection', 'comfort', 'respect', 'tension', 'familiarity'], additionalProperties: false },
    memory_facts: { type: 'array', items: { type: 'string' }, maxItems: 2 },
    current_scene: { type: 'string' },
    current_outfit: { type: 'string' },
    image_intent: { type: 'boolean' },
    image_prompt: { type: 'string' },
  },
  required: ['reply', 'mood', 'relationship_delta', 'relationship_changes', 'memory_facts', 'current_scene', 'current_outfit', 'image_intent', 'image_prompt'],
  additionalProperties: false,
};

const relationshipProfileSchema = {
  type: 'object',
  properties: {
    relationship: { type: 'integer', minimum: 0, maximum: 100 },
    relationships: { type: 'object', properties: { trust: { type: 'integer', minimum: 0, maximum: 100 }, romance: { type: 'integer', minimum: 0, maximum: 100 }, affection: { type: 'integer', minimum: 0, maximum: 100 }, comfort: { type: 'integer', minimum: 0, maximum: 100 }, respect: { type: 'integer', minimum: 0, maximum: 100 }, tension: { type: 'integer', minimum: 0, maximum: 100 }, familiarity: { type: 'integer', minimum: 0, maximum: 100 } }, required: ['trust', 'romance', 'affection', 'comfort', 'respect', 'tension', 'familiarity'], additionalProperties: false },
  },
  required: ['relationship', 'relationships'],
  additionalProperties: false,
};

const storySchema = {
  type: 'object',
  properties: {
    narration: { type: 'string' },
    dialogue: { type: 'array', items: { type: 'object', properties: { speaker: { type: 'string' }, text: { type: 'string' } }, required: ['speaker', 'text'], additionalProperties: false }, maxItems: 4 },
    choices: { type: 'array', items: { type: 'string' }, minItems: 2, maxItems: 4 },
    current_scene: { type: 'string' },
    image_prompt: { type: 'string' },
    world_state: {
      type: 'object',
      properties: {
        location: { type: 'string' }, time: { type: 'string' }, atmosphere: { type: 'string' },
        characters: { type: 'array', items: { type: 'object', properties: {
          name: { type: 'string' }, present: { type: 'boolean' }, visible: { type: 'boolean' }, outfit: { type: 'string' }, position: { type: 'string' }, pose: { type: 'string' }, expression: { type: 'string' },
        }, required: ['name', 'present', 'visible', 'outfit', 'position', 'pose', 'expression'], additionalProperties: false }, maxItems: 7 },
        objects: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, state: { type: 'string' }, location: { type: 'string' } }, required: ['name', 'state', 'location'], additionalProperties: false }, maxItems: 12 },
      },
      required: ['location', 'time', 'atmosphere', 'characters', 'objects'], additionalProperties: false,
    },
  },
  required: ['narration', 'dialogue', 'choices', 'current_scene', 'image_prompt', 'world_state'],
  additionalProperties: false,
};

const visualTranslationSchema = {
  type: 'object',
  properties: {
    premise: { type: 'string' },
    starting_scene: { type: 'string' },
    player: { type: 'object', properties: { name: { type: 'string' }, role: { type: 'string' }, description: { type: 'string' }, appearance: { type: 'string' }, outfit: { type: 'string' } }, required: ['name', 'role', 'description', 'appearance', 'outfit'], additionalProperties: false },
    participants: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, role: { type: 'string' }, description: { type: 'string' }, appearance: { type: 'string' }, outfit: { type: 'string' } }, required: ['name', 'role', 'description', 'appearance', 'outfit'], additionalProperties: false }, maxItems: 6 },
  },
  required: ['premise', 'starting_scene', 'player', 'participants'],
  additionalProperties: false,
};

async function handleTranslateVisualDescriptions(req, res) {
  let body;
  try { body = await readJson(req, 30_000); } catch (error) { return json(res, error.status || 400, { error: error.message }); }
  const story = body.story && typeof body.story === 'object' ? body.story : {};
  const player = story.player && typeof story.player === 'object' ? story.player : {};
  const participants = Array.isArray(story.participants) ? story.participants.slice(0, 6) : [];
  if (!String(story.starting_scene || '').trim() || !String(player.name || '').trim()) return json(res, 400, { error: 'Нужны стартовая сцена и главный герой для подготовки иллюстраций.' });
  const source = {
    premise: String(story.premise || ''), starting_scene: String(story.starting_scene || ''),
    player: { name: String(player.name || ''), role: String(player.role || ''), description: String(player.description || ''), appearance: String(player.appearance || ''), outfit: String(player.outfit || '') },
    participants: participants.map((person, index) => ({ name: String(person.name || `Участник ${index + 1}`).trim(), role: String(person.role || ''), description: String(person.description || ''), appearance: String(person.appearance || ''), outfit: String(person.outfit || '') })),
  };
  const participantKeys = source.participants.map((person) => person.name.toLocaleLowerCase('en-US'));
  if (new Set(participantKeys).size !== participantKeys.length) return json(res, 400, { error: 'У каждого участника истории должно быть уникальное имя.' });
  const prompt = 'Translate the story and character descriptions below into clear, natural English specifically for a photorealistic image-generation model. Preserve names exactly. Translate every non-empty role, description, appearance, premise, and starting scene. Preserve every numeric age exactly and attach it only to the same named character it belongs to; never transfer, average, or merge traits between characters. For each character, split appearance into stable physical identity (age, face, hair, body) and a separate outfit field containing only clothing explicitly specified in the source. Do not invent clothes; leave outfit empty if none were specified. Keep all details; leave empty source strings empty. Return only the requested JSON object.';
  try {
    const upstream = await fetch(`${ollama}/api/chat`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: chatModel, stream: false, messages: [{ role: 'system', content: `${prompt} Return valid JSON with the exact same keys. Do not copy Russian values; translate them.` }, { role: 'user', content: JSON.stringify(source) }], options: { temperature: 0.1, top_p: 0.8, num_ctx: 4096, num_predict: 1200 } }),
      signal: AbortSignal.timeout(90_000),
    });
    const result = await upstream.json();
    if (!upstream.ok) return json(res, upstream.status, { error: result.error || 'Не удалось подготовить английские описания.' });
    const parsed = parseJsonText(result.message?.content || '');
    if (!parsed?.player || !Array.isArray(parsed.participants)) return json(res, 502, { error: 'Модель вернула описания в неверном формате.' });
    const translatedByName = new Map(parsed.participants.map((person) => [String(person?.name || '').trim().toLocaleLowerCase('en-US'), person]));
    if (source.participants.some((person) => !translatedByName.has(person.name.toLocaleLowerCase('en-US')))) {
      return json(res, 502, { error: 'Переводчик изменил или пропустил имя участника. Попробуй подготовить историю ещё раз.' });
    }
    return json(res, 200, { visual_context: {
      premise: String(parsed.premise || '').slice(0, 1600), starting_scene: String(parsed.starting_scene || '').slice(0, 1600),
      player: { name: source.player.name, role: String(parsed.player.role || '').slice(0, 160), description: String(parsed.player.description || '').slice(0, 900), appearance: String(parsed.player.appearance || '').slice(0, 500), outfit: String(parsed.player.outfit || '').slice(0, 300) },
      participants: source.participants.map((person) => {
        const translated = translatedByName.get(person.name.toLocaleLowerCase('en-US'));
        return { name: person.name, role: String(translated.role || '').slice(0, 160), description: String(translated.description || '').slice(0, 900), appearance: String(translated.appearance || '').slice(0, 500), outfit: String(translated.outfit || '').slice(0, 300) };
      }),
    } });
  } catch (error) {
    return json(res, 502, { error: error.name === 'TimeoutError' ? 'Перевод описаний занял слишком много времени.' : 'Не удалось подключиться к Ollama.', detail: error.message });
  }
}

function initialWorldState(story, cast, currentScene = '') {
  const visual = story.visual_context || {};
  const profiles = [visual.player, ...(Array.isArray(visual.participants) ? visual.participants : [])];
  return {
    location: String(currentScene || visual.starting_scene || story.starting_scene || '').slice(0, 500),
    time: 'As established in the opening scene', atmosphere: 'As established in the opening scene',
    characters: cast.map((character, index) => {
      const profile = profiles.find((item) => item && item.name === character.name) || {};
      const appearance = String(profile.appearance || character.appearance || character.description || 'Appearance not specified').slice(0, 700);
      return { name: character.name, role: String(profile.role || character.role).slice(0, 160), appearance, description: String(profile.description || character.description || '').slice(0, 700), present: true, visible: true, outfit: String(profile.outfit || '').slice(0, 300), position: 'As established in the opening scene', pose: 'As established in the opening scene', expression: 'As established in the opening scene' };
    }),
    objects: [],
  };
}

function ageBoundIdentity(person) {
  const details = `${person.appearance || ''} ${person.description || ''} ${person.role || ''}`;
  const ageMatch = details.match(/\b(1[89]|[2-9]\d)\s*[- ]year[- ]old\b|\b(1[89]|[2-9]\d)\s+years?\s+old\b|\baged?\s*[:=]?\s*(1[89]|[2-9]\d)\b|(?:возраст\s*[:=]?\s*)?(1[89]|[2-9]\d)\s*(?:лет|года)\b/iu);
  if (!ageMatch) return '';
  const age = ageMatch.slice(1).find(Boolean);
  const feminine = /\b(?:woman|female|mother|daughter|wife|girl|blonde|brunette)\b|женщин|мать|мама|дочер|жен[аы]|блондинк|брюнетк/iu.test(details);
  const masculine = /\b(?:man|male|father|son|husband|boy)\b|мужчин|отец|папа|сын|муж/iu.test(details);
  const gender = feminine && !masculine ? 'adult woman' : masculine && !feminine ? 'adult man' : 'adult person';
  const maturity = Number(age) >= 40
    ? 'with mature facial features appropriate for this age, clearly not a young woman or teenager'
    : 'with a distinctly youthful young-adult face and skin, clearly adult but not middle-aged; no gray hair or older-looking features';
  return `EXACT AGE AND IDENTITY: ${age}-year-old ${gender}, ${maturity}`;
}

function mergeWorldState(raw, previous, cast, story, allowOutfitChange, allowLocationChange) {
  const baseline = previous && Array.isArray(previous.characters) ? previous : initialWorldState(story, cast);
  const returned = raw && typeof raw === 'object' ? raw : {};
  const byName = new Map((Array.isArray(returned.characters) ? returned.characters : []).map((person) => [String(person?.name || '').toLocaleLowerCase(), person]));
  const characters = baseline.characters.map((oldPerson, index) => {
    const update = byName.get(oldPerson.name.toLocaleLowerCase()) || {};
    return {
      name: oldPerson.name, role: oldPerson.role, appearance: oldPerson.appearance, description: oldPerson.description,
      present: typeof update.present === 'boolean' ? update.present : oldPerson.present,
      visible: typeof update.visible === 'boolean' ? update.visible : oldPerson.visible,
      outfit: (allowOutfitChange || !oldPerson.outfit) && String(update.outfit || '').trim() ? String(update.outfit).slice(0, 500) : oldPerson.outfit,
      position: String(update.position || oldPerson.position || '').slice(0, 240),
      pose: String(update.pose || oldPerson.pose || '').slice(0, 240),
      expression: String(update.expression || oldPerson.expression || '').slice(0, 180),
    };
  });
  const objectMap = new Map((Array.isArray(baseline.objects) ? baseline.objects : []).map((item) => [String(item.name).toLocaleLowerCase(), item]));
  for (const item of Array.isArray(returned.objects) ? returned.objects.slice(0, 12) : []) {
    const name = String(item?.name || '').slice(0, 100); if (!name) continue;
    const oldItem = objectMap.get(name.toLocaleLowerCase()) || {};
    objectMap.set(name.toLocaleLowerCase(), { name, state: String(item.state || oldItem.state || '').slice(0, 180), location: String(item.location || oldItem.location || '').slice(0, 180) });
  }
  const objects = [...objectMap.values()].slice(0, 12);
  return {
    location: String(allowLocationChange ? (returned.location || baseline.location) : baseline.location || returned.location || '').slice(0, 500),
    time: String(returned.time || baseline.time || '').slice(0, 160),
    atmosphere: String(returned.atmosphere || baseline.atmosphere || '').slice(0, 240),
    characters, objects,
  };
}

async function handleStoryTurn(req, res) {
  let body;
  try { body = await readJson(req, 80_000); } catch (error) { return json(res, error.status || 400, { error: error.message }); }
  const story = body.story && typeof body.story === 'object' ? body.story : {};
  const player = story.player && typeof story.player === 'object' ? story.player : {};
  const participants = Array.isArray(story.participants) ? story.participants.slice(0, 6) : [];
  const turns = Array.isArray(body.turns) ? body.turns.slice(-12) : [];
  const action = String(body.action || '').trim().slice(0, 2000);
  if (!String(story.starting_scene || '').trim() || !String(player.name || '').trim() || !action) return json(res, 400, { error: 'Для хода нужны стартовая сцена, главный персонаж и действие игрока.' });
  const cast = [player, ...participants].map((person, index) => ({
    name: String(person.name || (index === 0 ? 'Главный герой' : `Участник ${index}`)).slice(0, 80),
    role: String(person.role || (index === 0 ? 'Главный герой — управляется игроком' : 'Участник сцены')).slice(0, 160),
    description: String(person.description || '').slice(0, 1000),
    appearance: String(person.appearance || '').slice(0, 500),
  }));
  const sceneState = String(body.current_scene || story.starting_scene).slice(0, 1200);
  const previousWorldState = body.world_state && typeof body.world_state === 'object' ? body.world_state : initialWorldState(story, cast, turns.length ? sceneState : '');
  const history = turns.map((turn) => ({ action: String(turn.action || '').slice(0, 800), narration: String(turn.narration || '').slice(0, 1000), dialogue: Array.isArray(turn.dialogue) ? turn.dialogue.slice(0, 4) : [], choices: Array.isArray(turn.choices) ? turn.choices.slice(0, 4) : [], image_prompt: String(turn.image_prompt || '').slice(0, 1000), current_scene: String(turn.current_scene || '').slice(0, 800) }));
  const system = `Ты — ведущий и сценарист интерактивной визуальной новеллы. Игрок управляет главным персонажем ${String(player.name).slice(0, 80)}. Ты описываешь мир и управляешь только остальными персонажами. Никогда не придумывай за главного героя его слова, действия, решения или чувства: оставляй это игроку. На каждый ход продолжай именно текущую сцену, учитывай историю и последствия прежних решений, сохраняй характеры и имена участников. Никогда не переноси возраст, пол, волосы, одежду и черты внешности с одного именованного персонажа на другого. Каждый новый ход обязан заметно продвигать сцену: отвечать на последнее действие, сообщать новую уместную деталь или создавать естественное следствие. Не повторяй целиком реплики, описания, выборы и события из предыдущих ходов. Не перематывай время и не меняй место без сюжетной причины. Не добавляй новых участников без необходимости. Сохраняй одежду, причёски и другие визуальные детали каждого персонажа между ходами и иллюстрациями; меняй их только если переодевание или иное изменение явно произошло в сюжете. Указывай устойчивые визуальные детали в current_scene и image_prompt.

Название/жанр: ${String(story.title || 'Интерактивная история').slice(0, 120)} / ${String(story.genre || 'драма').slice(0, 100)}
Тон: ${String(story.tone || 'кинематографичный, живой').slice(0, 300)}
Завязка: ${String(story.premise || '').slice(0, 1600)}
Стартовая сцена: ${String(story.starting_scene).slice(0, 1600)}
Персонажи и роли: ${JSON.stringify(cast)}
Подготовленные английские описания для точного визуального образа: ${JSON.stringify(story.visual_context || {})}
Текущее положение сцены: ${sceneState}
АВТОРИТЕТНОЕ СОСТОЯНИЕ МИРА: ${JSON.stringify(previousWorldState)}
История ходов (не противоречь ей; используй её, чтобы продолжать с текущего момента): ${JSON.stringify(history)}

Ответь по-русски. narration — выразительное продолжение сцены, dialogue — реплики NPC, choices — 2–4 конкретных варианта, current_scene — краткое описание для игрока. world_state — машиночитаемый источник истины для следующих ходов и изображений. location, time, atmosphere, одежду, позиции, позы, выражения и предметы записывай на английском, имена сохраняй. Используй только заданных персонажей; present=true только если персонаж сейчас физически находится в сцене. Не меняй одежду, волосы, возраст, внешность или локацию без явного события; если изменений не было, скопируй прежнее состояние. Если одежда персонажа ещё не задана, выбери для первой сцены простую повседневную одежду, запиши её в state и сохраняй дальше без изменений. image_prompt строго на английском и должен соответствовать текущему действию и narration без добавленных прикосновений, жестов, людей или предметов. Для каждого видимого персонажа явно повтори его точный числовой возраст, пол и внешность из его собственного описания; не заменяй «45-year-old woman» словами «mature woman» и «20-year-old man» словом «young man». Верни только JSON по схеме.`;
  const turnMessages = [
    { role: 'system', content: system },
    { role: 'user', content: `Действие/реплика игрока в ЭТОМ ходу: ${action}\nПродолжи историю от текущего состояния. Дай реакцию персонажей на это действие и развивай сцену дальше, не повторяя предыдущие ходы.` },
  ];
  try {
    const upstream = await fetch(`${ollama}/api/chat`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: chatModel, stream: false, format: storySchema, messages: turnMessages, options: { temperature: 0.86, top_p: 0.92, num_ctx: 8192, num_predict: 1400 } }),
      signal: AbortSignal.timeout(180_000),
    });
    const result = await upstream.json();
    if (!upstream.ok) return json(res, upstream.status, { error: result.error || 'Модель истории не ответила.' });
    let parsed = parseJsonText(result.message?.content || '');
    if (!parsed || typeof parsed.narration !== 'string' || !Array.isArray(parsed.choices)) return json(res, 502, { error: 'Модель вернула ход в неверном формате.' });
    const priorNarration = turns.map((turn) => String(turn.narration || ''));
    const priorDialogue = turns.flatMap((turn) => (Array.isArray(turn.dialogue) ? turn.dialogue : []).map((line) => String(line?.text || '')));
    const priorChoices = turns.flatMap((turn) => Array.isArray(turn.choices) ? turn.choices.map(String) : []);
    const repeatsStoryText = (draft) => repeatsRecentReply(draft.narration, priorNarration)
      || (Array.isArray(draft.dialogue) && draft.dialogue.some((line) => repeatsRecentReply(line?.text, priorDialogue)))
      || (Array.isArray(draft.choices) && draft.choices.some((choice) => repeatsRecentReply(choice, priorChoices)));
    if (repeatsStoryText(parsed)) {
      try {
        const correction = await fetch(`${ollama}/api/chat`, {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model: chatModel, stream: false, format: storySchema, messages: [
            { role: 'system', content: `${system}\n\nКОРРЕКТИРОВКА: черновик повторяет уже сказанное. Перепиши ход с новой конкретной реакцией и новым событием или деталью, точно отвечая на последнее действие игрока. Не используй повторённую реплику и не пересказывай прошлый ход.` },
            ...turnMessages.slice(1),
            { role: 'assistant', content: JSON.stringify(parsed) },
            { role: 'user', content: 'Перепиши этот ход без повторов. Верни только полный JSON по заданной схеме.' },
          ], options: { temperature: 0.9, top_p: 0.94, num_ctx: 8192, num_predict: 1400 } }),
          signal: AbortSignal.timeout(120_000),
        });
        if (correction.ok) {
          const correctedResult = await correction.json();
          const corrected = parseJsonText(correctedResult.message?.content || '');
          if (corrected && typeof corrected.narration === 'string' && Array.isArray(corrected.dialogue) && Array.isArray(corrected.choices) && !repeatsStoryText(corrected)) parsed = corrected;
        }
      } catch { /* Keep the valid first response if the optional rewrite cannot complete. */ }
    }
    // Do not display a verbatim NPC line again if the model ignored the rewrite request.
    parsed.dialogue = parsed.dialogue.filter((line) => !repeatsRecentReply(line?.text, priorDialogue));
    const allowedNames = new Set(cast.slice(1).map((person) => person.name));
    const dialogue = parsed.dialogue.filter((line) => line && typeof line.text === 'string' && allowedNames.has(String(line.speaker))).slice(0, 4).map((line) => ({ speaker: String(line.speaker).slice(0, 80), text: line.text.slice(0, 1000) }));
    const visualDraft = String(parsed.image_prompt || '').slice(0, 1800);
    const worldEvents = `${action}\n${parsed.narration}`;
    const allowOutfitChange = outfitChanged(worldEvents);
    const allowLocationChange = !body.world_state || /(?:выш(?:ел|ла|ли)|вош(?:ел|ла|ли)|переш(?:ел|ла|ли)|покин(?:ул|ула|ули)|отправил(?:ся|ась|ись)|вернул(?:ся|ась|ись)|пришел|пришла|пош(?:ел|ла|ли)\s+(?:в|на)|left|entered|arrived|went\s+to|moved\s+to)/iu.test(worldEvents);
    const worldState = mergeWorldState(parsed.world_state, previousWorldState, cast, story, allowOutfitChange, allowLocationChange);
    const visualTranslationMessages = [
      { role: 'system', content: 'You are a literal Russian-to-English visual-prompt translator for photorealistic image generation. Return exactly one natural English image prompt, with no explanation and no Cyrillic characters. Preserve every concrete visual fact in the current action, narration and current world state, especially each character’s hair length and color, age, clothing, location and who is present. Never replace or shorten a stated hairstyle. Apply only changes explicitly described in this turn; otherwise preserve the current state. Do not invent props, gestures, contact, clothing changes or extra people. Refer to people by role (such as “the mother” or “the protagonist”), not by their names.' },
      { role: 'user', content: JSON.stringify({ player_action: action, current_scene: String(parsed.current_scene || sceneState), current_world_state: worldState, this_turn_narration: parsed.narration, this_turn_dialogue: dialogue, character_visual_context: story.visual_context || {}, image_draft: visualDraft }) },
    ];
    const visualTranslation = await fetch(`${ollama}/api/chat`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: chatModel, stream: false, messages: visualTranslationMessages, options: { temperature: 0.1, top_p: 0.8, num_ctx: 4096, num_predict: 700 } }),
      signal: AbortSignal.timeout(90_000),
    });
    const translatedVisual = await visualTranslation.json();
    if (!visualTranslation.ok) return json(res, 502, { error: translatedVisual.error || 'Не удалось подготовить английское описание иллюстрации.' });
    let imagePrompt = String(translatedVisual.message?.content || '').trim().replace(/^```(?:text)?\s*|\s*```$/g, '').slice(0, 1800);
    if (!imagePrompt) return json(res, 502, { error: 'Модель не вернула английское описание иллюстрации.' });
    if (/[\u0400-\u04ff]/u.test(imagePrompt)) {
      const corrected = await fetch(`${ollama}/api/chat`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: chatModel, stream: false, messages: [
          { role: 'system', content: 'Translate the supplied image prompt into English. Keep its meaning and every visual detail exactly. Replace Russian character names with their role (mother, protagonist, etc.). Output only English, with no Cyrillic characters and no explanation.' },
          { role: 'user', content: imagePrompt },
        ], options: { temperature: 0, num_ctx: 2048, num_predict: 500 } }),
        signal: AbortSignal.timeout(60_000),
      });
      if (!corrected.ok) return json(res, 502, { error: 'Не удалось проверить английский промпт для изображения.' });
      const correctedResult = await corrected.json();
      imagePrompt = String(correctedResult.message?.content || '').trim().replace(/^```(?:text)?\s*|\s*```$/g, '').slice(0, 1800);
      if (!imagePrompt || /[\u0400-\u04ff]/u.test(imagePrompt)) return json(res, 502, { error: 'Модель оставила русский текст в промпте иллюстрации.' });
    }
    return json(res, 200, {
      narration: parsed.narration.slice(0, 5000), dialogue,
      choices: parsed.choices.filter((choice) => typeof choice === 'string' && choice.trim()).slice(0, 4).map((choice) => choice.slice(0, 240)),
      current_scene: String(parsed.current_scene || sceneState).slice(0, 1200),
      world_state: worldState,
      image_prompt: imagePrompt,
    });
  } catch (error) {
    return json(res, 502, { error: error.name === 'TimeoutError' ? 'Ход истории генерируется слишком долго.' : 'Не удалось обратиться к модели истории.', detail: error.message });
  }
}

async function handleAnalyzeCharacter(req, res) {
  let body;
  try { body = await readJson(req, 30_000); } catch (error) { return json(res, error.status || 400, { error: error.message }); }
  const character = body.character && typeof body.character === 'object' ? body.character : {};
  const description = `Имя: ${String(character.name || 'Персонаж').slice(0, 80)}\nИстория и характер: ${String(character.bio || '').slice(0, 1200)}\nМанера общения: ${String(character.personality || '').slice(0, 1800)}`;
  const prompt = `Ты настраиваешь стартовые параметры отношений между пользователем и вымышленным персонажем для ролевого чата, который по умолчанию происходит через интернет-переписку. Определи только исходную динамику, прямо следующую из профиля, не изобретай совместное прошлое.\n\n${description}\n\nОцени по шкале 0–100: общая близость, доверие, романтика, нежность, комфорт рядом, уважение, напряжение/обида и знакомство. Учитывай семейные и другие заданные роли: если персонаж — мать пользователя, это уже близкие и хорошо знакомые люди, поэтому поставь высокими знакомство, нежность, доверие и уважение; романтику поставь ровно 0. Если персонаж обозначен как коллега или незнакомец, не назначай близость как родственнику. Не делай все значения одинаковыми. Верни только JSON по схеме.`;
  try {
    const upstream = await fetch(`${ollama}/api/chat`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: chatModel, stream: false, format: relationshipProfileSchema, messages: [{ role: 'system', content: prompt }], options: { temperature: 0.2, top_p: 0.8, num_ctx: 4096, num_predict: 160 } }),
      signal: AbortSignal.timeout(90_000),
    });
    const result = await upstream.json();
    if (!upstream.ok) return json(res, upstream.status, { error: result.error || 'Не удалось проанализировать профиль.' });
    const parsed = parseJsonText(result.message?.content || '');
    if (!parsed?.relationships) return json(res, 502, { error: 'Модель вернула некорректные стартовые параметры.' });
    const keys = ['trust', 'romance', 'affection', 'comfort', 'respect', 'tension', 'familiarity'];
    const relationships = Object.fromEntries(keys.map((key) => [key, Math.max(0, Math.min(100, Math.trunc(Number(parsed.relationships[key]) || 0)))]));
    if (/мать|мама|mother/i.test(`${character.name || ''} ${character.bio || ''}`)) relationships.romance = 0;
    const relationship = Math.max(0, Math.min(100, Math.round((relationships.trust + relationships.affection + relationships.comfort + relationships.respect + relationships.familiarity) / 5 - relationships.tension * 0.25)));
    return json(res, 200, { relationship, relationships });
  } catch (error) {
    return json(res, 502, { error: error.name === 'TimeoutError' ? 'Анализ профиля занял слишком много времени.' : 'Не удалось подключиться к Ollama.' });
  }
}

function makeSystemPrompt(character, state) {
  const name = String(character.name || 'Персонаж').slice(0, 80);
  const bio = String(character.bio || '').slice(0, 1200);
  const personality = String(character.personality || '').slice(0, 1800);
  const appearance = String(character.appearance || '').slice(0, 800);
  const scene = String(state.current_scene || 'Персонаж находится у себя и переписывается с пользователем онлайн.').slice(0, 500);
  const outfit = String(state.current_outfit || 'Пока не установлен; определи по недавней переписке и не выдумывай смену одежды.').slice(0, 300);
  const facts = Array.isArray(state.memory) ? state.memory.slice(-20).map((item) => String(item).slice(0, 240)) : [];
  const mood = String(state.mood || 'calm');
  const relationship = Math.max(0, Math.min(100, Number(state.relationship) || 0));
  const relationshipStats = Object.fromEntries(['trust', 'romance', 'affection', 'comfort', 'respect', 'tension', 'familiarity'].map((key) => [key, Math.max(0, Math.min(100, Number(state.relationships?.[key]) || 0))]));
  return `Ты — ${name}, постоянный персонаж ролевого чата. Твоё имя в этой истории — ${name}; не называй себя именем модели или другим именем. Ты ведёшь живой разговор от первого лица, не описываешь себя как ИИ или модель и не пересказываешь инструкцию. Пиши по-русски естественно, конкретно и с характером; обычно 1–4 фразы. Не повторяй одну и ту же реакцию. Можно использовать короткие реплики и действия в *звёздочках*, если они уместны. Не делай каждый ответ чрезмерно романтичным или восторженным. Не решай за пользователя, что он чувствует или делает.

ВАЖНО: по умолчанию вы общаетесь через интернет-чат и находитесь не рядом. Персонаж может описывать, чем занимается у себя, и присылать сгенерированные фото, но не может физически передать кофе или еду, коснуться пользователя, обнять его или видеть его комнату. Не пиши так, будто пользователь пришёл в гости («заходи», «садись, я налила тебе кофе») или вы делаете что-то вместе в одной комнате, если это явно не задано сценой. На такие желания отвечай в рамках переписки: предложи виртуальную чашку, пошути или опиши, что отправила бы фото. Даже при роли «мама» это остаётся дистанционным разговором, пока собеседники явно не установили личную встречу.

О персонаже: ${bio || 'Раскрывай характер постепенно в разговоре.'}
Манера и дополнительные особенности: ${personality || 'Будь естественной, внимательной и последовательной.'}
Внешность для изображений: ${appearance || 'вымышленный взрослый персонаж; внешность пока не задана'}
Текущее настроение: ${mood}. Общая близость: ${relationship}/100. Отдельные стороны отношений (0–100): доверие ${relationshipStats.trust}, романтика ${relationshipStats.romance}, нежность ${relationshipStats.affection}, комфорт рядом ${relationshipStats.comfort}, уважение ${relationshipStats.respect}, напряжение/обида ${relationshipStats.tension}, знакомство ${relationshipStats.familiarity}.
Текущая сцена персонажа (описывает только её местоположение и действия; собеседник остаётся у себя, если явно не установлена личная встреча): ${scene}
Текущая одежда персонажа: ${outfit}. Это постоянное состояние между сообщениями. Сохраняй её в следующих ответах и фото. Не добавляй одежду и не меняй её автоматически. Обновляй состояние только если в переписке явно произошло переодевание, персонаж надел или снял конкретную вещь, либо пользователь явно задал новую одежду. Если смены не было, верни current_outfit без изменений.
Проверенные факты памяти о пользователе: ${facts.length ? facts.join(' | ') : 'Пока нет сохранённых фактов.'} Используй только эти факты для утверждений о прошлых разговорах и предпочтениях пользователя. Не выдумывай, что он любит кофе, о чём обычно говорит или какие события пережил. Если факта нет в памяти и текущей переписке, честно скажи, что не помнишь.

Сохраняй непрерывность сцены и отношений. Настроение меняй по смыслу, а не случайно. Отвечай прежде всего на последнюю реплику: не перенаправляй разговор на «что на уме», «поговорим о личном» и похожие общие вопросы без причины. Не повторяй формулировки недавних ответов; не используй дежурные приветствия, терапевтические фразы и одинаковый вопрос в конце каждого сообщения. Если собеседник говорит, что у него всё хорошо, поддержи это и развивай именно эту тему. Учитывай обращение и заданную роль в профиле; не подменяй их внезапно. Для коротких реплик отвечай коротко и по-человечески, с конкретной реакцией и без лишнего пафоса. Не говори, что «даёшь» фото, пока действительно не запросила снимок — ответь естественно и продолжай сцену.

relationship_delta — изменение общей близости от -2 до 2, relationship_changes — отдельные изменения доверия, романтики, нежности, комфорта, уважения, напряжения и знакомства, каждое от -2 до 2. Меняй только то, на что реально повлиял этот ход; большинство значений обычно 0. Не превращай любое дружелюбие во флирт и не повышай показатели автоматически. Уважение и доверие могут расти независимо от романтики; напряжение растёт при конфликте и снижается при искреннем примирении. memory_facts — только новые устойчивые факты о пользователе, прямо сообщённые им в его последнем сообщении; никогда не добавляй факты о себе, догадки или пересказ. Если подходящего факта нет, верни пустой массив. current_scene — короткое обновление только того, что персонаж делает у себя, и дистанционного общения; не описывай физические действия с пользователем.

image_intent включай только если собеседник прямо просит фото/селфи/показать, что ты сейчас делаешь, либо ты естественно обещаешь прислать снимок в ответ на такую просьбу. Не создавай фото по одному лишь упоминанию еды, места или внешности. Если собеседник попросил снимок, image_prompt должен описывать визуальную сцену из контекста, а не надпись или интерфейс; используй текущую сцену, действие, настроение, ракурс обычного телефона и детали внешности персонажа. Обязательно соблюдай текущее состояние одежды. Пиши image_prompt на английском, без имени реального человека и без текста в кадре. Если фото не нужно — image_intent=false, image_prompt пустая строка.

Верни строго один JSON-объект по заданной схеме. reply — единственный текст, который увидит собеседник.`;
}

async function handleChat(req, res) {
  let body;
  try { body = await readJson(req); } catch (error) { return json(res, error.status || 400, { error: error.message }); }
  const character = body.character && typeof body.character === 'object' ? body.character : {};
  const state = body.state && typeof body.state === 'object' ? body.state : {};
  const history = Array.isArray(body.history) ? body.history.filter((item) => ['user', 'assistant'].includes(item?.role) && typeof item.content === 'string').slice(-24) : [];
  const message = typeof body.message === 'string' ? body.message.trim().slice(0, 3000) : '';
  const recentReplies = [...history.filter((item) => item.role === 'assistant').map((item) => item.content), ...(Array.isArray(state.recent_replies) ? state.recent_replies : [])]
    .filter((reply) => typeof reply === 'string' && !/^\[Фото из текущей сцены\]$/u.test(reply))
    .slice(-12).map((reply) => reply.slice(0, 500));
  if (!message) return json(res, 400, { error: 'Пустое сообщение.' });
  const messages = [
    { role: 'system', content: `${makeSystemPrompt(character, state)}\n\nПоследние ответы персонажа (только для избегания повторов; не цитируй их): ${recentReplies.length ? recentReplies.join(' | ') : 'это начало разговора'}` },
    ...history,
    { role: 'user', content: message },
  ];
  try {
    let parsed = await requestRpCompletion(messages);
    if (describesOfflineInteraction(parsed.reply) || repeatsRecentReply(parsed.reply, recentReplies) || usesWrongPersonaName(parsed.reply, character.name || 'Персонаж')) {
      const draft = JSON.stringify(parsed);
      const offlineViolation = describesOfflineInteraction(parsed.reply);
      const correctionMessages = [
        { role: 'system', content: `${messages[0].content}\n\nСТРОГОЕ ПРАВИЛО: это только переписка через интернет. Не приглашай пользователя сесть или прийти, не предлагай принести/передать ему предметы, не описывай прикосновения и совместное нахождение в комнате. Если он скучает или упоминает напиток, ответь тепло словами, шуткой или виртуальным жестом в чате. Исправляй такие нарушения даже если они звучат заботливо. Имя персонажа — ${String(character.name || 'Персонаж')}; не называй себя Katarau/Катарау. Не повторяй предыдущую реплику персонажа; ответь именно на последнюю реплику пользователя и выбери новую формулировку.` },
        ...messages.slice(1),
        { role: 'assistant', content: draft },
        { role: 'user', content: `Перепиши предыдущую реплику ${offlineViolation ? 'без физического взаимодействия' : 'так, чтобы не повторять недавний ответ и ответить на мой последний вопрос'}. Верни полный JSON по исходной схеме.` },
      ];
      try { parsed = await requestRpCompletion(correctionMessages); } catch {}
      if (describesOfflineInteraction(parsed.reply)) parsed.reply = 'Жаль, что я не могу угостить тебя по-настоящему. Могу хотя бы отправить тебе виртуальную чашку чая ☕ Рада, что ты написал.';
      if (repeatsRecentReply(parsed.reply, recentReplies)) parsed.reply = 'Ой, кажется, я повторилась. Извини 🙂 Расскажи, как у тебя проходит день?';
      if (usesWrongPersonaName(parsed.reply, character.name || 'Персонаж')) parsed.reply = parsed.reply.replace(/(?:меня\s+зовут|я\s*[—-])\s*[\p{L}\p{N}-]+/iu, `Я — ${String(character.name || 'персонаж')}.`);
    }
    return json(res, 200, {
      reply: parsed.reply.slice(0, 5000),
      mood: ['calm', 'happy', 'curious', 'shy', 'affectionate', 'tired', 'upset', 'playful'].includes(parsed.mood) ? parsed.mood : 'calm',
      relationship_delta: Math.max(-2, Math.min(2, Math.trunc(Number(parsed.relationship_delta) || 0))),
      relationship_changes: Object.fromEntries(['trust', 'romance', 'affection', 'comfort', 'respect', 'tension', 'familiarity'].map((key) => [key, Math.max(-2, Math.min(2, Math.trunc(Number(parsed.relationship_changes?.[key]) || 0)))])),
      memory_facts: extractExplicitMemoryFact(message) ? [extractExplicitMemoryFact(message)] : keepGroundedMemoryFacts(parsed.memory_facts, message),
      current_scene: String(parsed.current_scene || state.current_scene || '').slice(0, 500),
      current_outfit: String(state.current_outfit && !outfitChanged(`${message}\n${parsed.reply}`) ? state.current_outfit : parsed.current_outfit || state.current_outfit || '').slice(0, 300),
      image_intent: Boolean(parsed.image_intent) && /\b(?:photo|picture|selfie|pic)\b|фото|фотк|селфи|сним(?:и|ок)|покажи\s+(?:себя|что|как)|пришли\s+(?:фото|снимок)|скинь\s+(?:фото|фотку|снимок)/iu.test(message),
      image_prompt: String(parsed.image_prompt || '').slice(0, 1200),
    });
  } catch (error) {
    return json(res, 502, { error: error.name === 'TimeoutError' ? 'Локальная модель отвечает слишком долго.' : 'Не удалось подключиться к Ollama. Проверь, что Ollama запущена.', detail: error.message });
  }
}

async function handleImage(req, res) {
  let body;
  try { body = await readJson(req, 100_000); } catch (error) { return json(res, error.status || 400, { error: error.message }); }
  const visual = typeof body.prompt === 'string' ? body.prompt.trim().slice(0, 1200) : '';
  const characters = Array.isArray(body.characters) ? body.characters.slice(0, 7).map((person) => ({ name: String(person?.name || '').slice(0, 80), role: String(person?.role || '').slice(0, 120), appearance: String(person?.appearance || '').slice(0, 500), description: String(person?.description || '').slice(0, 400), outfit: String(person?.outfit || '').slice(0, 300), position: String(person?.position || '').slice(0, 160), pose: String(person?.pose || '').slice(0, 160), expression: String(person?.expression || '').slice(0, 120) })) : [];
  const appearance = characters.length ? characters.map((person, index) => `PERSON ${index + 1} — one distinct individual${person.name ? ` named ${person.name}` : ''}${person.role ? `, role: ${person.role}` : ''}. ${ageBoundIdentity(person)}. Exact physical identity and appearance: ${person.appearance || person.description || 'not specified'}. Exact current clothing: ${person.outfit || 'not specified'}. Position: ${person.position || 'as established'}. Pose: ${person.pose || 'natural'}. Expression: ${person.expression || 'natural'}. Do not transfer any attributes from another numbered person.`).join('\n') .slice(0, 3500) : (typeof body.appearance === 'string' ? body.appearance.trim().slice(0, 800) : '');
  const characterCount = characters.length || Math.max(1, Math.min(7, Number.parseInt(body.character_count, 10) || 1));
  const outfit = typeof body.outfit === 'string' ? body.outfit.trim().slice(0, 300) : '';
  const visualState = typeof body.visual_state === 'string' ? body.visual_state.trim().slice(0, 1600) : '';
  const fixedSeed = Number.isInteger(body.seed) ? body.seed >>> 0 : null;
  const strictScene = body.strict_scene === true;
  const sceneFacts = [body.player_action, body.narration, body.dialogue].filter((value) => typeof value === 'string').join(' ').slice(0, 5000);
  // Enforce a hairstyle only when this turn explicitly mentions it; don't turn a one-off request into permanent character state.
  const requestedLongHair = /\b(?:long|waist[- ]length|shoulder[- ]length)\s+(?:blonde|blond|fair|light[- ](?:blonde|brown))\s+hair\b|\b(?:long|waist[- ]length|shoulder[- ]length)\s+hair\b|длинн\w*\s+(?:светл\w*|бел\w*|рус\w*)?\s*волос/iu.test(`${visual} ${String(body.player_action || '')}`);
  const contactWasExplicit = /\b(?:hug|embrace|kiss|touch|hold hands|grab|pull|tug|undress|change clothes|adjust(?:ing)? (?:her|his|their) clothes)\b|обним|поцел|дотрон|трогать|схват|потянуть|стянуть|раздеть|переод|поправ(?:ить|ляет) одежд|держ(?:ать|ит) за руку/iu.test(sceneFacts);
  if (!visual) return json(res, 400, { error: 'Не удалось определить сцену для фото.' });

  // Avoid flooding the GPU queue with repeated clicks or forgotten polling sessions.
  pruneExpiredImageJobs();
  if (jobs.size + pendingImageSubmissions >= MAX_TRACKED_IMAGE_JOBS) {
    res.setHeader('retry-after', '10');
    return json(res, 429, {
      error: 'Слишком много незавершённых генераций изображений. Дождись завершения одной из задач и попробуй снова.',
      limit: MAX_TRACKED_IMAGE_JOBS,
    });
  }

  // Reserve capacity before the first await so simultaneous requests cannot all pass the limit.
  pendingImageSubmissions += 1;
  try {
    // Ollama and SDXL share the same 12 GB GPU; release the chat model before ComfyUI runs.
    try {
      await fetch(`${ollama}/api/chat`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: chatModel, messages: [], keep_alive: 0, stream: false }),
        signal: AbortSignal.timeout(15_000),
      });
    } catch { /* Continue: ComfyUI may still have enough free GPU memory. */ }
    const sceneAccuracy = strictScene ? ' Literal scene accuracy: depict only actions and interactions explicitly established by the player action and current narration. Keep characters in natural, independent poses; do not make them touch or handle each other’s clothing unless that interaction is explicitly described.' : '';
    const hairAnchor = requestedLongHair ? ' The character explicitly described with long light or blonde hair in this scene must have long hair extending clearly below her shoulders, with long strands visible over the shoulders or back. Preserve the stated length and color for this image only; do not transfer this hairstyle to other characters.' : '';
    const separatedAgeSpecs = characters.map((person) => ageBoundIdentity(person)).filter(Boolean);
    const ageSeparation = separatedAgeSpecs.length > 1 ? ` Keep the listed ages attached to their numbered people: ${separatedAgeSpecs.join('; ')}. Do not swap, average, or blend their ages, faces, hair, or clothing.` : '';
    const prompt = `Photorealistic high-detail candid smartphone photo, natural unretouched look, believable indoor or outdoor light, authentic camera perspective, realistic skin and fabric texture, crisp facial and hair detail, spontaneous moment, vertical 2:3 composition. SUBJECTS — preserve separately, never combine attributes:\n${appearance || 'As established in the conversation.'}\n${ageSeparation} Current visual continuity and outfit state: ${visualState || 'as established in the current story scene'}. Scene and action: ${visual}.${hairAnchor}${sceneAccuracy} Show exactly ${characterCount} distinct characters. Keep every listed person's identity, age, gender, hair, and outfit bound to that same person; natural anatomy, no captions, no watermark.`;
    const hasYoungMan = characters.some((person) => /\b(1[89]|2\d)\s*[- ]year[- ]old\b|\b(1[89]|2\d)\s+years?\s+old\b|\baged?\s*[:=]?\s*(1[89]|2\d)\b/iu.test(`${person.appearance || ''} ${person.description || ''} ${person.role || ''}`) && /\b(?:man|male|son|boy)\b|мужчин|сын|муж/iu.test(`${person.appearance || ''} ${person.description || ''} ${person.role || ''}`));
    const ageNegative = hasYoungMan && !characters.some((person) => /\b(?:3\d|[4-9]\d)\s*[- ]year[- ]old\b|\b(?:3\d|[4-9]\d)\s+years?\s+old\b|\baged?\s*[:=]?\s*(?:3\d|[4-9]\d)\b/iu.test(`${person.appearance || ''} ${person.description || ''} ${person.role || ''}`) && /\b(?:man|male|father|son|husband)\b|мужчин|отец|сын|муж/iu.test(`${person.appearance || ''} ${person.description || ''} ${person.role || ''}`)) ? 'middle-aged man, older man, elderly man, gray-haired man, salt-and-pepper hair on a young man, mature male face, deep wrinkles on a young man' : '';
    const strictNegative = [strictScene && !contactWasExplicit ? 'unrequested physical contact, embracing, romantic couple pose, leaning heads together, hands on another person’s waist, holding hands, grabbing another person, grabbing clothes, pulling a shirt, tugging clothing, clothing adjustment' : '', requestedLongHair ? 'short hair, bob cut, bob haircut, chin-length hair, cropped hair, pixie cut, short blonde hair' : '', ageNegative].filter(Boolean).join(', ');
    const id = randomUUID();
    const response = await comfyFetch('/prompt', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt: buildImageWorkflow(prompt, `LiveRoom/chat-${id}`, strictNegative, fixedSeed), client_id: 'local-liveroom-chat' }),
      timeout: 20_000,
    });
    const queued = await response.json();
    if (!response.ok || !queued.prompt_id) {
      const details = queued.error?.message || queued.error || 'ComfyUI отклонил граф изображения.';
      return json(res, 502, { error: typeof details === 'string' ? details : JSON.stringify(details) });
    }
    jobs.set(id, { promptId: queued.prompt_id, createdAt: Date.now(), state: 'queued' });
    return json(res, 202, { jobId: id });
  } catch (error) { return json(res, 502, { error: error.message || 'Не удалось поставить генерацию изображения в очередь.' }); }
  finally { pendingImageSubmissions = Math.max(0, pendingImageSubmissions - 1); }
}

async function handleImageStatus(jobId, res) {
  const job = jobs.get(jobId);
  if (!job) return json(res, 404, { error: 'Задача изображения не найдена или срок её хранения истёк.' });
  if (Date.now() - job.createdAt > JOB_TTL_MS) {
    jobs.delete(jobId);
    return json(res, 410, { status: 'error', error: 'Время ожидания изображения истекло. Запусти генерацию ещё раз.' });
  }
  const promptId = job.promptId;
  try {
    const response = await comfyFetch(`/history/${encodeURIComponent(promptId)}`, { timeout: 6000 });
    if (!response.ok) return json(res, 502, { error: 'ComfyUI не вернул состояние генерации.' });
    const history = await response.json();
    const entry = history[promptId];
    if (entry) {
      const status = entry.status || {};
      if (status.status_str === 'error') {
        jobs.delete(jobId);
        const detail = (status.messages || []).map((item) => item?.[1]?.exception_message).filter(Boolean).join('\n');
        return json(res, 200, { status: 'error', error: detail || 'ComfyUI завершил генерацию с ошибкой.' });
      }
      const image = Object.values(entry.outputs || {}).flatMap((output) => output.images || []).find((item) => item.filename && item.type === 'output');
      if (image) {
        jobs.delete(jobId);
        return json(res, 200, { status: 'done', image: { filename: image.filename, subfolder: image.subfolder || '', type: image.type || 'output' } });
      }
      if (status.completed) {
        jobs.delete(jobId);
        return json(res, 200, { status: 'error', error: 'ComfyUI завершил задачу, но не сохранил изображение.' });
      }
    }
    const queueResponse = await comfyFetch('/queue', { timeout: 5000 });
    if (!queueResponse.ok) return json(res, 502, { error: 'ComfyUI не смог проверить очередь генерации. Повтори проверку позже.' });
    const queue = await queueResponse.json();
    const contains = (items) => (items || []).some((item) => Array.isArray(item) ? item[1] === promptId : item.prompt_id === promptId);
    if (contains(queue.queue_running)) {
      job.state = 'running';
      return json(res, 200, { status: 'running' });
    }
    if (contains(queue.queue_pending)) {
      job.state = 'queued';
      return json(res, 200, { status: 'queued' });
    }
    jobs.delete(jobId);
    return json(res, 200, { status: 'error', error: 'ComfyUI больше не видит эту задачу. Попробуй создать изображение заново.' });
  } catch (error) { return json(res, 502, { error: `Ошибка связи с ComfyUI: ${error.message}` }); }
}

async function handleGeneratedImage(url, res) {
  const filename = url.searchParams.get('filename') || '';
  const subfolder = url.searchParams.get('subfolder') || '';
  const type = url.searchParams.get('type') || 'output';
  const subfolderParts = subfolder ? subfolder.split(/[\\/]/) : [];
  if (path.basename(filename) !== filename || !/^[\w. -]+$/.test(filename) || type !== 'output' || subfolder.startsWith('/') || subfolder.startsWith('\\\\') || subfolderParts.some((part) => !part || part === '.' || part === '..' || !/^[\w. -]+$/.test(part))) {
    res.writeHead(400); return res.end('Invalid image path');
  }
  try {
    const result = await comfyFetch(`/view?${new URLSearchParams({ filename, subfolder, type })}`, { timeout: 60_000 });
    if (!result.ok || !result.body) { res.writeHead(502); return res.end('ComfyUI could not serve the image'); }
    res.writeHead(200, { 'content-type': result.headers.get('content-type') || 'image/png', 'cache-control': 'private, max-age=3600' });
    for await (const chunk of result.body) res.write(Buffer.from(chunk));
    res.end();
  } catch { res.writeHead(502); res.end('ComfyUI connection failed'); }
}

async function handleHealth(res) {
  const check = async (url) => {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(2500) });
      return { ok: response.ok, status: response.status };
    } catch (error) {
      return { ok: false, error: error.name === 'TimeoutError' ? 'timeout' : 'unavailable' };
    }
  };
  const [ollamaStatus, comfyStatus] = await Promise.all([
    check(`${ollama}/api/tags`),
    check(`${comfy}/system_stats`),
  ]);
  const ok = ollamaStatus.ok && comfyStatus.ok;
  return json(res, ok ? 200 : 503, {
    ok,
    services: { ollama: ollamaStatus, comfyui: comfyStatus },
    model: chatModel,
    uptime_seconds: Math.floor(process.uptime()),
  });
}

const server = http.createServer(async (req, res) => {
  let url;
  try { url = new URL(req.url, `http://127.0.0.1:${port}`); }
  catch { res.writeHead(400); return res.end('Bad request'); }
  if (req.method === 'GET' && url.pathname === '/api/health') return handleHealth(res);
  if (req.method === 'POST' && url.pathname === '/api/chat') return handleChat(req, res);
  if (req.method === 'POST' && url.pathname === '/api/translate-visual-descriptions') return handleTranslateVisualDescriptions(req, res);
  if (req.method === 'POST' && url.pathname === '/api/story-turn') return handleStoryTurn(req, res);
  if (req.method === 'POST' && url.pathname === '/api/analyze-character') return handleAnalyzeCharacter(req, res);
  if (req.method === 'POST' && url.pathname === '/api/image') return handleImage(req, res);
  const imageJob = /^\/api\/image\/([0-9a-f-]{36})$/i.exec(url.pathname);
  if (req.method === 'GET' && imageJob) return handleImageStatus(imageJob[1], res);
  if (req.method === 'GET' && url.pathname === '/api/generated-image') return handleGeneratedImage(url, res);
  if (req.method !== 'GET' && req.method !== 'HEAD') { res.writeHead(405, { allow: 'GET, HEAD' }); return res.end('Method not allowed'); }
  let requested;
  try { requested = decodeURIComponent(url.pathname === '/' ? 'index.html' : url.pathname.replace(/^\/+/, '')); }
  catch { res.writeHead(400); return res.end('Bad request'); }
  if (!PUBLIC_PAGES.has(requested) && !PUBLIC_ASSET.test(requested)) { res.writeHead(404); return res.end('Not found'); }
  const file = path.resolve(root, requested);
  if (!file.startsWith(root + path.sep)) { res.writeHead(404); return res.end('Not found'); }
  fs.readFile(file, (error, data) => {
    if (error) { res.writeHead(404); return res.end('Not found'); }
    res.writeHead(200, { 'content-type': mime[path.extname(file).toLowerCase()] || 'application/octet-stream', 'cache-control': 'no-store' });
    if (req.method === 'HEAD') return res.end();
    res.end(data);
  });
});

server.listen(port, '127.0.0.1', () => console.log(`LiveRoom: http://127.0.0.1:${port}`));
