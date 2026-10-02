import re
import json, os, shutil, time, uuid
from pathlib import Path
import requests
import subprocess
import threading

ROOT = Path(r"C:\AI")
SITE_ROOT = Path(__file__).resolve().parent
CHAR_ROOT = SITE_ROOT / "characters"
COMFY_URL = "http://127.0.0.1:8188"
CHAR_ROOT.mkdir(parents=True, exist_ok=True)

# --- Backend Functions ---
def safe_name(s):
    s = "".join(c if c.isalnum() or c in "_-" else "_" for c in (s or "Character_01"))
    return s.strip("_") or "Character_01"

def char_dir(name):
    p = CHAR_ROOT / safe_name(name)
    for x in ("source", "generated_dataset", "approved", "captions", "config", "samples", "output"):
        (p / x).mkdir(parents=True, exist_ok=True)
    return p

def comfy_online():
    try:
        return requests.get(COMFY_URL + "/system_stats", timeout=2).ok
    except Exception:
        return False

def object_info():
    r = requests.get(COMFY_URL + "/object_info", timeout=10)
    r.raise_for_status()
    return r.json()

def choose_name(info, node_type, preferred=None, strict=False):
    node = info.get(node_type, {})
    req = node.get("input", {}).get("required", {})
    for key in ("ckpt_name", "pulid_file", "model_name", "filename"):
        if key in req and isinstance(req[key], list) and req[key]:
            first = req[key][0]
            if isinstance(first, list):
                vals = first
                if preferred:
                    for v in vals:
                        if str(v).lower() == preferred.lower():
                            return v
                if strict:
                    return None
                return vals[0] if vals else preferred
    return None if strict else preferred

def upload_to_comfy(path):
    with open(path, "rb") as f:
        r = requests.post(
            COMFY_URL + "/upload/image",
            files={"image": (Path(path).name, f, "application/octet-stream")},
            data={"overwrite": "true", "type": "input"},
            timeout=60
        )
    r.raise_for_status()
    return r.json()["name"]

def build_workflow(image_name, prompt, negative, width, height, strength, seed):
    info = object_info()
    ckpt = choose_name(info, "CheckpointLoaderSimple", "RealVisXL_V5.0_fp16.safetensors", strict=True)
    pulid = choose_name(info, "PulidModelLoader", "pulid_v1.1.safetensors")
    if not ckpt:
        raise RuntimeError(r"ComfyUI не нашёл RealVisXL_V5.0_fp16.safetensors. Положите checkpoint в C:\AI\ComfyUI\models\checkpoints\ и перезапустите ComfyUI.")
    if not pulid:
        raise RuntimeError("ComfyUI не нашёл PuLID Model Loader или модель PuLID.")

    nodes = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}},
        "3": {"class_type": "PulidModelLoader", "inputs": {"pulid_file": pulid}},
        "4": {"class_type": "PulidEvaClipLoader", "inputs": {}},
        "5": {"class_type": "PulidInsightFaceLoader", "inputs": {"provider": "CUDA"}},
        "6": {"class_type": "ApplyPulid", "inputs": {
            "model": ["1", 0], "pulid": ["3", 0], "eva_clip": ["4", 0],
            "face_analysis": ["5", 0], "image": ["2", 0],
            "method": "fidelity", "weight": float(strength),
            "start_at": 0.0, "end_at": 1.0}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": int(width), "height": int(height), "batch_size": 1}},
        "10": {"class_type": "KSampler", "inputs": {
            "seed": int(seed), "steps": 30, "cfg": 6.5, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "denoise": 1.0, "model": ["6", 0],
            "positive": ["7", 0], "negative": ["8", 0], "latent_image": ["9", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {
            "filename_prefix": "CharacterLab/Character", "images": ["11", 0]}}
    }
    return {"prompt": nodes}

def submit(workflow):
    payload = dict(workflow)
    payload["client_id"] = str(uuid.uuid4())
    r = requests.post(COMFY_URL + "/prompt", json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI /prompt: {r.text}")
    return r.json()["prompt_id"]

def wait_history(prompt_id, timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(COMFY_URL + f"/history/{prompt_id}", timeout=15)
        if r.ok:
            data = r.json()
            if prompt_id in data:
                return data[prompt_id]
        time.sleep(1)
    raise TimeoutError("ComfyUI не завершил генерацию.")

def fetch_outputs(history, name):
    target = char_dir(name) / "generated_dataset"
    results = []
    for node in history.get("outputs", {}).values():
        for item in node.get("images", []):
            params = {
                "filename": item["filename"],
                "subfolder": item.get("subfolder", ""),
                "type": item.get("type", "output")
            }
            r = requests.get(COMFY_URL + "/view", params=params, timeout=60)
            r.raise_for_status()
            ext = Path(item["filename"]).suffix or ".png"
            fn = target / (Path(item["filename"]).stem + "_" + uuid.uuid4().hex[:6] + ext)
            fn.write_bytes(r.content)
            results.append(str(fn))
    return results

# ============================================================
# NORMAL CHARACTER GENERATION
# ============================================================
PROMPT_BASE = [
    "adult woman",
    "photorealistic photograph",
    "realistic human proportions",
    "natural skin texture",
    "realistic hands and anatomy",
    "individual hair strands",
    "natural realistic lighting",
    "high detail",
]

NEGATIVE_PROMPT = (
    "man, male, boy, masculine person, masculine body, second person, duplicate person, "
    "anime, cartoon, illustration, drawing, comic, manga, cel shading, 3d render, CGI, "
    "doll, figurine, painting, sketch, plastic skin, waxy skin, oversmoothed skin, "
    "deformed face, distorted face, asymmetrical face, bad eyes, malformed eyes, "
    "bad anatomy, extra limbs, extra fingers, missing fingers, duplicate body, "
    "blurry, low quality, oversaturated, text, watermark"
)


def make_prompt(clothing, scene, pose, extra="", prompt_base=None, variation=0):
    base = prompt_base if prompt_base is not None else PROMPT_BASE
    parts = [*base, str(clothing or "casual clothes"), str(scene or "studio"), str(pose or "portrait")]
    if extra:
        parts.append(str(extra).strip())
    parts.append("natural candid variation" if variation % 2 else "natural realistic pose")
    return ", ".join(x for x in parts if x)


def generate(name, source, count, strength, width, height, clothing="casual clothes", scene="studio",
             pose="portrait", extra="", progress=None, on_item=None, prompt_base=None, negative_prompt=None):
    if source is None:
        raise RuntimeError("Загрузите исходное фото.")
    if not comfy_online():
        raise RuntimeError("ComfyUI не запущен. Запустите его на 127.0.0.1:8188.")

    p = char_dir(name)
    original = p / "source" / ("original" + Path(source).suffix.lower())
    shutil.copy2(source, original)
    if progress:
        progress(0.02, "Загружаю исходное фото…")
    image_name = upload_to_comfy(str(original))
    negative = negative_prompt if negative_prompt is not None else NEGATIVE_PROMPT
    base = prompt_base if prompt_base is not None else PROMPT_BASE
    files = []
    prompts = []
    total = max(1, int(count))

    for i in range(total):
        prompt = make_prompt(clothing, scene, pose, extra, base, i)
        prompts.append(prompt)
        if progress:
            progress(0.05 + .90 * (i / max(total, 1)), f"Вариант {i + 1} из {total}…")
        seed = int(time.time() * 1000) % 2147483647 + i * 9973
        pid = submit(build_workflow(image_name, prompt, negative, width, height, strength, seed))
        outs = fetch_outputs(wait_history(pid), name)
        if not outs:
            raise RuntimeError(f"Вариант {i + 1}: ComfyUI не вернул изображение.")
        fn = Path(outs[0])
        fn = remove_background(fn)
        fn.with_suffix(".txt").write_text(prompt, encoding="utf-8")
        files.append(str(fn))
        if on_item:
            on_item(i, str(fn), prompt, total)

    if progress:
        progress(1.0, "Готово — персонажи без фона.")
    return files, prompts


def regenerate_slot(name, slot, prompt, strength, width, height, negative_prompt=None):
    p = char_dir(name)
    originals = image_files(p / "source")
    if not originals:
        raise RuntimeError("Исходное фото не найдено.")
    image_name = upload_to_comfy(str(originals[0]))
    negative = negative_prompt if negative_prompt is not None else NEGATIVE_PROMPT
    seed = int(time.time() * 1000) % 2147483647 + int(slot) * 7919 + 12345
    pid = submit(build_workflow(image_name, prompt, negative, width, height, strength, seed))
    outs = fetch_outputs(wait_history(pid), name)
    if not outs:
        raise RuntimeError("ComfyUI не вернул новое изображение.")
    fn = remove_background(Path(outs[0]))
    fn.with_suffix(".txt").write_text(prompt, encoding="utf-8")
    return str(fn)


_BG_SESSION = None

def remove_background(path):
    """Make the generated PNG transparent using rembg. No white/solid replacement background is added."""
    global _BG_SESSION
    try:
        from rembg import remove, new_session
    except Exception as e:
        raise RuntimeError(
            "Удаление фона не установлено. Запустите INSTALL_BACKGROUND_REMOVER.bat и повторите генерацию."
        ) from e
    if _BG_SESSION is None:
        _BG_SESSION = new_session("u2net")
    data = path.read_bytes()
    result = remove(data, session=_BG_SESSION)
    out = path.with_suffix(".png")
    out.write_bytes(result)
    if out != path and path.exists():
        try: path.unlink()
        except OSError: pass
    return out

def launch_comfy():
    py = ROOT / "ComfyUI" / "venv" / "Scripts" / "python.exe"
    main = ROOT / "ComfyUI" / "main.py"
    if comfy_online():
        return "● ComfyUI подключён"
    if not py.exists() or not main.exists():
        return "ComfyUI не найден"
    import subprocess
    subprocess.Popen([str(py), str(main)], cwd=str(ROOT / "ComfyUI"), creationflags=subprocess.CREATE_NEW_CONSOLE)
    return "◌ Запуск ComfyUI…"

def find_free_port(start=7860, end=7880):
    import socket
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass
    raise OSError(f"Нет свободного порта в диапазоне {start}-{end}.")





# ============================================================
# Dataset / Approved backend
# ============================================================

def image_files(folder):
    if not folder.exists(): return []
    exts={".png",".jpg",".jpeg",".webp",".bmp"}
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def caption_for(img):
    cap=img.with_suffix(".txt")
    return cap.read_text(encoding="utf-8",errors="replace").strip() if cap.exists() else ""

# Compatibility helpers used by the Dataset UI. Captions always live next to the image.
def image_caption(img):
    return caption_for(img)

def save_image_caption(img, text):
    img.with_suffix(".txt").write_text(str(text or "").strip(), encoding="utf-8")

def dataset_report(name):
    p=char_dir(name)
    approved=image_files(p/"approved"); generated=image_files(p/"generated_dataset"); source=image_files(p/"source")
    missing=[x.name for x in approved if not caption_for(x)]
    return {"approved":approved,"generated":generated,"source":source,"missing":missing}

def dataset_items(name,scope):
    r=dataset_report(name)
    if scope=="Approved": files=r["approved"]
    elif scope=="Generated dataset": files=r["generated"]
    elif scope=="Source": files=r["source"]
    else: files=r["source"]+r["generated"]+r["approved"]
    items=[]
    for f in files:
        tag="TRAIN" if f.parent.name=="approved" else f.parent.name.upper()
        cap=caption_for(f)
        items.append({"url":"/media/"+str(f.relative_to(CHAR_ROOT)).replace("\\","/"),
                      "name":f.name,"tag":tag,"caption":cap,
                      "caption_missing":not bool(cap) and f.parent.name=="approved"})
    return items,r

def make_caption_templates(name):
    p=char_dir(name); created=0
    for img in image_files(p/"approved"):
        cap=img.with_suffix(".txt")
        if not cap.exists(): cap.write_text("",encoding="utf-8"); created+=1
    return created

HTML = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Character Lab</title>
<style>
:root{
 --bg:#090b0f;--bg2:#0e1116;--card:#11151a;--card2:#171b20;--line:#292e35;
 --text:#f4f7fb;--muted:#7d858e;--muted2:#5c646d;--accent:#ff8754;
 --accent2:#ffab7c;--cyan:#55d8d1;--green:#65d39b
}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;background:
 var(--bg);
 color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
button,input,select,textarea{font:inherit}
button{cursor:pointer}
.app{width:min(1500px,calc(100vw - 36px));margin:0 auto;padding:18px 0 34px}
.layout{display:grid;grid-template-columns:76px 1fr;gap:15px}
.sidebar{height:calc(100vh - 36px);min-height:680px;position:sticky;top:18px;background:rgba(14,16,20,.97);
 border:1px solid #292e35;border-radius:22px;padding:9px;display:flex;flex-direction:column;align-items:center;
 box-shadow:0 30px 100px rgba(0,0,0,.35)}
.logo{width:42px;height:42px;border-radius:13px;display:grid;place-items:center;margin:2px 0 25px;
 background:linear-gradient(145deg,#ffab7d,#e76b3d);font-weight:900;box-shadow:0 8px 28px rgba(255,127,70,.18)}
.nav{display:flex;flex-direction:column;gap:5px;width:100%}
.nav button{width:56px;height:52px;border:0;border-radius:14px;background:transparent;color:#777d85;
 display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;font-size:8px}
.nav button .ico{font-size:17px;line-height:1}.nav button.active{background:#1b1e21;color:#fff}
.nav button.active .ico{color:var(--accent)}
.sidebar-bottom{margin-top:auto}.comfy{width:56px;height:54px;border:1px solid #292e35;border-radius:14px;
 background:#171b20;color:#9aa7b8;font-size:8px;line-height:1.3}
.main{min-width:0}.topbar{height:58px;display:flex;align-items:center;justify-content:space-between;margin-bottom:11px}
.brand h1{margin:0;font-size:18px;letter-spacing:-.04em}.brand p{margin:4px 0 0;color:#707780;font-size:9px}
.badges{display:flex;gap:7px}.badge{border:1px solid #292e35;background:#111419;color:#8b9198;border-radius:999px;padding:7px 10px;font-size:9px}
.badge.online:before{content:"";display:inline-block;width:5px;height:5px;border-radius:50%;background:var(--cyan);margin-right:6px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-bottom:10px}
.stat{border:1px solid #292e35;border-radius:15px;background:linear-gradient(145deg,#15181d,#101216);padding:12px 14px}
.stat label{display:block;color:#647287;text-transform:uppercase;letter-spacing:.13em;font-size:8px}
.stat strong{display:block;margin-top:6px;font-size:16px;letter-spacing:-.02em}.stat small{color:#566477;font-size:8px}
.panel{border:1px solid #292e35;border-radius:20px;background:rgba(13,15,19,.98);padding:15px;box-shadow:0 30px 90px rgba(0,0,0,.18)}
.panel-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.panel-head h2{font-size:13px;margin:0}.panel-head p{font-size:9px;color:#707780;margin:4px 0 0}
.steps{display:flex;gap:5px}.step{padding:6px 8px;border:1px solid #292e35;border-radius:7px;font-size:8px;color:#707780}
.step.active{background:#1b1e21;color:#eef2f6}
.namebar{display:flex;align-items:center;gap:10px;padding:8px 10px;background:#0e1115;border:1px solid #292e35;border-radius:11px;margin-bottom:10px}
.namebar label{font-size:8px;color:#707780;text-transform:uppercase;letter-spacing:.12em}.namebar input{flex:1}
input,select,textarea{width:100%;background:#171b20;color:#eef2f6;border:1px solid #292e35;border-radius:9px;outline:0}
input,select{height:37px;padding:0 10px}textarea{padding:9px;min-height:66px;resize:vertical}
input:focus,select:focus,textarea:focus{border-color:#4a5a70}
.workspace{display:grid;grid-template-columns:minmax(0,1.32fr) minmax(380px,.68fr);gap:10px}
.card{background:#0e1216;border:1px solid #292e35;border-radius:15px;padding:12px}
.card-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:9px}
.card-head b{font-size:10px}.card-head span{font-size:8px;color:#626970}
.drop{height:500px;border:1px dashed #394047;border-radius:12px;background:#0b0e12;display:flex;align-items:center;justify-content:center;overflow:hidden;position:relative}
.drop img{display:none;max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;object-position:center;flex:0 0 auto}.drop.loaded img{display:block}.drop.loaded .upload-copy{display:none}
.upload-copy{text-align:center;color:#78869a}.upload-copy .icon{font-size:32px;color:#596169;margin-bottom:9px}
.upload-copy b{display:block;color:#b5beca;font-size:12px}.upload-copy span{display:block;margin-top:5px;font-size:9px}
.controls{display:flex;flex-direction:column;gap:9px}.control{border:1px solid #292e35;background:#12161b;border-radius:13px;padding:11px}
.control-title{font-size:8px;color:#727980;text-transform:uppercase;letter-spacing:.13em;margin-bottom:9px}.prompt-control textarea{width:100%;box-sizing:border-box;background:#171b20;color:#c9d0d8;border:1px solid #292e35;border-radius:8px;font:9px/1.45 Arial,sans-serif}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}.field label{display:block;font-size:8px;color:#8b9299;margin:0 0 5px}
.range{display:grid;grid-template-columns:1fr 56px;gap:8px;align-items:center}.range output{height:34px;border:1px solid #292e35;border-radius:8px;background:#171b20;display:grid;place-items:center;font-size:9px}
input[type=range]{height:4px;padding:0;border:0;accent-color:var(--accent)}
.actions{display:grid;grid-template-columns:1fr 260px;gap:9px;margin:10px 0}
.primary{height:46px;border:0;border-radius:11px;color:#fff;font-weight:800;background:linear-gradient(135deg,var(--accent2),var(--accent));box-shadow:0 12px 35px rgba(255,128,76,.17)}
.status{height:46px;display:flex;align-items:center;padding:0 12px;border:1px solid #292e35;border-radius:11px;background:#13171c;color:#80878e;font-size:9px}
.results{border:1px solid #292e35;background:#0e1216;border-radius:15px;padding:12px}
.results-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}.results-head b{font-size:10px}.results-head span{font-size:8px;color:#626a72}
.gallery{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;min-height:170px}
.empty{grid-column:1/-1;min-height:170px;border:1px dashed #30363d;border-radius:11px;display:grid;place-items:center;color:#596168;font-size:9px}
.tile{position:relative;aspect-ratio:2/3;border-radius:10px;overflow:hidden;border:1px solid #30363d;background:#111823}
.tile img{width:100%;height:100%;object-fit:cover;display:block}.tile.selected{outline:2px solid var(--accent);outline-offset:2px}
.tile.waiting{border-style:dashed;background:#0f1318}.slot-placeholder{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;color:#59616a;background:linear-gradient(145deg,#101419,#151a20)}.slot-placeholder span{font-size:20px;font-weight:800;color:#3d454e}.slot-placeholder small{font-size:8px}.slot-index{position:absolute;left:7px;top:7px;padding:4px 6px;border-radius:6px;background:#0b0f13cc;color:#9ca5ae;font-size:8px;z-index:2}.tile.retrying img{filter:blur(16px);opacity:.38}.tile.retrying:after{content:"↻  Generating new variant…";position:absolute;inset:0;display:grid;place-items:center;background:rgba(9,11,15,.48);color:#fff;font-size:9px;font-weight:700}.retry-label{position:absolute;inset:0;display:grid;place-items:center;background:rgba(9,11,15,.48);color:#fff;font-size:9px;font-weight:700;z-index:3}.retry{position:absolute;left:6px;bottom:6px;height:24px;padding:0 8px;border:1px solid #3b4249;border-radius:7px;background:#11151aee;color:#f4f7fb;font-size:8px;z-index:2}.tile:hover .retry{display:block}.retry{display:block}
.check{display:none;position:absolute;right:7px;top:7px;width:21px;height:21px;border-radius:7px;background:var(--accent);place-items:center;color:#fff;font-size:11px}
.tile.selected .check{display:grid}
.result-actions{display:flex;gap:8px;margin-top:10px}.result-actions button{height:38px;border-radius:9px;padding:0 14px;font-size:9px}
.approve{border:0;background:var(--accent);color:#fff;font-weight:750}.clear{border:1px solid #292e35;background:#171b20;color:#c2cad4}
.toast{display:none;position:fixed;right:20px;bottom:20px;max-width:420px;padding:12px 14px;border:1px solid #33383e;border-radius:12px;background:#131d2b;color:#e2e8ef;box-shadow:0 20px 70px #0009;z-index:99;font-size:10px}
.footer{text-align:center;color:#404d60;font-size:8px;padding:10px}
.reference-fit{width:auto!important;height:auto!important;max-width:100%!important;max-height:100%!important;object-fit:contain!important}
@media(max-width:1050px){.workspace{grid-template-columns:1fr}.drop{height:420px}.gallery{grid-template-columns:repeat(4,1fr)}}
@media(max-width:720px){.app{width:calc(100vw - 14px)}.layout{grid-template-columns:1fr}.sidebar{position:static;height:auto;min-height:0;flex-direction:row}.logo{margin:0 8px 0 0}.nav{flex-direction:row}.sidebar-bottom{margin:0 0 0 auto}.stats{grid-template-columns:1fr 1fr}.actions{grid-template-columns:1fr}.gallery{grid-template-columns:repeat(3,1fr)}.steps{display:none}}
.lab-page{min-width:0}
.lab-page .card{min-height:0}
.page-title{display:flex;justify-content:space-between;align-items:center;margin:8px 0 11px;padding:0 3px}.page-title h2{margin:0;font-size:16px;letter-spacing:-.03em}.page-title p{margin:4px 0 0;color:#707780;font-size:9px}.page-title-badges{display:flex;gap:5px}.page-title-badges span{padding:6px 8px;border:1px solid #292e35;border-radius:7px;font-size:8px;color:#707780}.page-title-badges span:first-child{background:#1b1e21;color:#eef2f6}

</style>
</head>
<body>
<div class="app"><div class="layout">
<aside class="sidebar">
<div class="logo">C</div>
<nav class="nav">
<button class="active" data-page="generate"><span class="ico">✦</span>Generate</button>
<button data-page="dataset"><span class="ico">◇</span>Dataset</button>
<button data-page="settings"><span class="ico">⚙</span>Settings</button>
</nav>
<div class="sidebar-bottom"><button class="comfy" id="comfyBtn">▶<br>ComfyUI</button></div>
</aside>
<main class="main">
<header class="topbar"><div class="brand"><h1>Character Lab</h1><p>Identity workspace · local generation</p></div><div class="badges"><span class="badge online" id="onlineBadge">ComfyUI</span><span class="badge">RealVisXL · PuLID</span></div></header>
<div id="page-generate" class="lab-page">
<header class="topbar"><div class="brand"><h1>Generate</h1><p>Reference → normal character generation → approved images</p></div><div class="badges"><span class="badge">CHARACTER LAB</span><span class="badge">READY TO GENERATE</span></div></header>
<section class="stats">
<div class="stat"><label>Identity</label><strong id="statIdentity">68%</strong><small>PuLID fidelity</small></div>
<div class="stat"><label>Variants</label><strong id="statCount">20</strong><small>per generation</small></div>
<div class="stat"><label>Canvas</label><strong id="statCanvas">512 × 768</strong><small>output canvas</small></div>
<div class="stat"><label>Pipeline</label><strong id="statPipeline">Ready</strong><small id="statPipelineNote">local GPU</small></div>
</section>
<section class="panel">
<div class="panel-head"><div><h2>Create character</h2><p>One reference → normal generation → transparent PNG results.</p></div><div class="steps"><span class="step active">01 SOURCE</span><span class="step">02 GENERATE</span><span class="step">03 APPROVED</span></div></div>
<div class="namebar"><label>Character</label><input id="name" value="Character_01"></div>
<div class="workspace">
<div class="card">
<div class="card-head"><b>Reference image</b><span>identity source</span></div>
<div class="drop" id="drop"><input id="file" type="file" accept="image/*" hidden><img id="preview"><div class="upload-copy"><div class="icon">↑</div><b>Drop your reference here</b><span>or click to choose an image</span></div></div>
</div>
<div class="controls">
<div class="control"><div class="control-title">Reference strength</div><div class="range"><input id="strength" type="range" min=".55" max="1.00" step=".01" value=".68"><output id="strengthOut">0.68</output></div></div>
<div class="control"><div class="control-title">Appearance</div><div class="grid2">
<div class="field"><label>Clothing</label><select id="clothing"><option>casual clothes</option><option>evening dress</option><option>leather jacket</option><option>business outfit</option><option>sportswear</option><option>lingerie</option><option>swimwear</option></select></div>
<div class="field"><label>Scene</label><select id="scene"><option>studio</option><option>bedroom</option><option>city street</option><option>cafe</option><option>office</option><option>beach</option><option>forest</option></select></div>
<div class="field"><label>Pose / angle</label><select id="pose"><option>portrait</option><option>standing</option><option>sitting</option><option>full body</option><option>three-quarter view</option><option>profile</option></select></div>
<div class="field"><label>Extra prompt</label><input id="extra" placeholder="natural expression, soft light"></div>
</div></div>
<div class="control"><div class="control-title">Output</div><div class="grid2">
<div class="field"><label>Variants</label><select id="count"><option>4</option><option>8</option><option>12</option><option>16</option><option selected>20</option></select></div>
<div class="field"><label>Width</label><select id="width"><option selected>512</option><option>640</option><option>768</option></select></div>
<div class="field"><label>Height</label><select id="height"><option selected>768</option><option>896</option><option>1024</option></select></div>
</div></div>
</div></div>
<div class="actions"><button class="primary" id="generateBtn">✨ Generate characters</button><div class="status" id="status">Ready to generate</div></div>
<div class="results">
<div class="results-head"><b>Generated characters</b><span id="selectedText">0 selected</span></div>
<div class="gallery" id="gallery"><div class="empty">Generated variants will appear here</div></div>
<div class="result-actions"><button class="approve" id="saveDatasetBtn">✓ Save selected dataset</button><button class="clear" id="saveReadyBtn">✓ Save all ready</button><button class="clear" id="approveBtn">+ Add selected to Approved</button><button class="clear" id="clearBtn">Clear selection</button></div>
</div>
</section>
<div class="footer">Character Lab · local private workspace</div>
</div>
<div id="page-dataset" class="lab-page" style="display:none">
<header class="topbar"><div class="brand"><h1>Dataset</h1><p>Full character dataset · review · captions · approved images</p></div><div class="badges"><span class="badge">CHARACTER LAB</span><span class="badge">APPROVED READY</span></div></header>
<section class="panel">
<div class="panel-head"><div><h2>Dataset workspace</h2><p>Весь dataset персонажа в том же интерфейсе, что и Generate.</p></div><div class="steps"><span class="step active">01 SOURCE</span><span class="step active">02 DATASET</span><span class="step active">03 TRAIN</span></div></div>
<div class="namebar"><label>Character</label><input id="datasetCharacter" value="Character_01"></div>
<div class="stats"><div class="stat"><label>Approved</label><strong id="dsApproved">0</strong><small>approved images</small></div><div class="stat"><label>Generated</label><strong id="dsGenerated">0</strong><small>variants</small></div><div class="stat"><label>Source</label><strong id="dsSource">0</strong><small>references</small></div><div class="stat"><label>Captions</label><strong id="dsCaption">0</strong><small>ready captions</small></div></div>
<div class="control"><div class="control-title">Dataset view</div><div class="grid2"><div class="field"><label>Folder</label><select id="datasetScope"><option value="approved">Approved</option><option value="generated">Generated dataset</option><option value="source">Source</option><option value="all">All images</option></select></div><div class="field"><label>Status</label><div class="status" id="datasetStatus">Ready</div></div></div></div>
<div class="results"><div class="results-head"><b id="datasetTitle">Approved</b><span id="datasetCount">0 images</span></div><div class="gallery" id="datasetGallery"><div class="empty">Dataset is empty</div></div></div>
<div class="control" style="margin-top:10px"><div class="control-title">Caption editor</div><div class="field"><label>Selected image</label><input id="captionFile" readonly placeholder="Click an image"></div><div class="field" style="margin-top:8px"><label>Caption</label><textarea id="captionText" style="min-height:90px" placeholder="Write a caption for the selected image"></textarea></div><div class="result-actions"><button class="approve" id="saveCaption">Save caption</button><button class="clear" id="refreshDataset">Refresh</button></div></div>
</section><div class="footer">Character Lab · dataset manager</div>
</div>

<div id="page-settings" class="lab-page" style="display:none">
<header class="topbar"><div class="brand"><h1>Settings</h1><p>Local paths · runtime</p></div></header>
<section class="panel"><div class="card"><div class="card-head"><b>Character Lab paths</b><span>LOCAL</span></div><div class="control"><div class="field"><label>AI root</label><input value="C:\\AI" readonly></div><div class="field" style="margin-top:8px"><label>Characters</label><input value="Character Lab folder\characters" readonly></div></div></div></section><div class="footer">Character Lab · settings</div>
</div>

</main></div></div>
<div class="toast" id="toast"></div>
<script>

const labPages=["generate","dataset","settings"];
function showLabPage(page){
  labPages.forEach(p=>{const el=document.getElementById("page-"+p);if(el)el.style.display=(p===page?"block":"none")});
  document.querySelectorAll(".nav button[data-page]").forEach(b=>b.classList.toggle("active",b.dataset.page===page));
  if(page==="dataset")loadDataset();
  if(page==="settings")checkTrainer();
}
document.querySelectorAll(".nav button[data-page]").forEach(b=>b.onclick=()=>showLabPage(b.dataset.page));
showLabPage("generate");

async function loadDataset(){
  const name=(document.getElementById("datasetCharacter").value||"Character_01").trim();
  const scope=document.getElementById("datasetScope").value;
  try{
    const d=await (await fetch("/api/dataset?name="+encodeURIComponent(name)+"&scope="+encodeURIComponent(scope))).json();
    document.getElementById("dsApproved").textContent=d.summary.approved;
    document.getElementById("dsGenerated").textContent=d.summary.generated;
    document.getElementById("dsSource").textContent=d.summary.source;
    document.getElementById("dsCaption").textContent=d.summary.captions;
    document.getElementById("datasetCount").textContent=d.items.length+" images";
    document.getElementById("datasetStatus").textContent=d.summary.missing?"Missing captions: "+d.summary.missing:"Dataset ready";
    const g=document.getElementById("datasetGallery");g.innerHTML="";
    if(!d.items.length){g.innerHTML='<div class="empty">No images in this folder</div>';return}
    d.items.forEach(item=>{
      const tile=document.createElement("div");tile.className="tile";
      tile.innerHTML='<img src="'+item.url+'"><div class="check" style="display:grid;right:6px;top:6px;background:'+(item.caption_ok?"#65d39b":"#ff8754")+'">'+(item.caption_ok?"✓":"!")+"</div>";
      tile.onclick=()=>{
        document.getElementById("captionFile").value=item.name;
        document.getElementById("captionText").value=item.caption||"";
        document.getElementById("captionText").dataset.path=item.url;
        document.querySelectorAll("#datasetGallery .tile").forEach(x=>x.classList.remove("selected"));
        tile.classList.add("selected");
      };
      g.appendChild(tile);
    });
  }catch(e){document.getElementById("datasetStatus").textContent="Dataset read error"}
}
document.getElementById("datasetScope").onchange=loadDataset;
document.getElementById("datasetCharacter").onchange=loadDataset;
document.getElementById("refreshDataset").onclick=loadDataset;
document.getElementById("saveCaption").onclick=async()=>{
  const url=document.getElementById("captionText").dataset.path;
  if(!url)return toast("Выбери изображение.");
  try{
    const r=await fetch("/api/caption",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image_url:url,caption:document.getElementById("captionText").value})});
    const d=await r.json();if(!r.ok)throw Error(d.error||"Caption error");toast("Caption сохранён");loadDataset();
  }catch(e){toast(e.message)}
};

let sourceFile=null,generated=[],selected=new Set(),pollTimer=null;
const $=id=>document.getElementById(id);
function toast(msg){const t=$("toast");t.textContent=msg;t.style.display="block";clearTimeout(window.toastTimer);window.toastTimer=setTimeout(()=>t.style.display="none",4500)}
function refreshStats(){$("statIdentity").textContent=Math.round(parseFloat($("strength").value)*100)+"%";$ ("strengthOut").value=$("strength").value;$ ("statCount").textContent=$("count").value;$ ("statCanvas").textContent=$("width").value+" × "+$("height").value}
$("strength").oninput=refreshStats;$("count").onchange=refreshStats;$("width").onchange=refreshStats;$("height").onchange=refreshStats;
$("drop").onclick=()=>$("file").click();$("file").onchange=e=>loadSource(e.target.files[0]);$("drop").ondragover=e=>e.preventDefault();$("drop").ondrop=e=>{e.preventDefault();loadSource(e.dataTransfer.files[0])};
function loadSource(f){if(!f||!f.type.startsWith("image/"))return toast("Выберите изображение.");sourceFile=f;$("preview").src=URL.createObjectURL(f);$("preview").classList.add("reference-fit");$("drop").classList.add("loaded");$("status").textContent="Reference loaded"}
$("generateBtn").onclick=async()=>{
 if(!sourceFile)return toast("Сначала загрузите исходное фото.");
 selected.clear();generated=[];renderGallery();
 const fd=new FormData();fd.append("image",sourceFile);["name","count","strength","width","height","clothing","scene","pose","extra"].forEach(k=>fd.append(k,$(k).value));
 $("generateBtn").disabled=true;$("status").textContent="Generating characters…";
 try{const r=await fetch("/api/generate",{method:"POST",body:fd}),d=await r.json();if(!r.ok)throw Error(d.error||"Generation error");pollTimer=setInterval(poll,700)}catch(e){toast(e.message);$("generateBtn").disabled=false;$("status").textContent="Ready to generate"}
};
async function poll(){try{const d=await(await fetch("/api/job")).json();$("status").textContent=d.message+" · "+d.progress+"%";generated=d.items||[];renderGallery();if(d.state==="done" && !(d.retrying&&Object.keys(d.retrying).length)){clearInterval(pollTimer);pollTimer=null;$("generateBtn").disabled=false}if(d.state==="error"){clearInterval(pollTimer);pollTimer=null;$("generateBtn").disabled=false;$("status").textContent="Generation failed";toast(d.error||"Generation failed")}}catch(e){}}
function renderGallery(){
 const g=$("gallery"); g.innerHTML="";
 if(!generated.length){g.innerHTML='<div class="empty">Generated characters will appear here</div>';return}
 generated.forEach((item,i)=>{
   const tile=document.createElement("div");
   const waiting=item.state==="waiting", retrying=item.state==="retrying";
   tile.className="tile"+(selected.has(i)?" selected":"")+(retrying?" retrying":"")+(waiting?" waiting":"");
   const image=item.url ? '<img src="'+item.url+'?v='+Date.now()+'">' : '<div class="slot-placeholder"><span>'+String(i+1).padStart(2,"0")+'</span><small>'+ (waiting?"Waiting for generation…":"No image") +'</small></div>';
   const promptHint=(item.prompt||"").replace(/"/g,"&quot;");
   tile.innerHTML=image+'<div class="slot-index">'+String(i+1).padStart(2,"0")+'</div><div class="check">✓</div>'+(retrying?'<div class="retry-label">↻ Generating new image…</div>':'')+(waiting?'':'<button class="retry" type="button" title="Generate this variant again">↻ Retry</button>');
   tile.title=promptHint;
   tile.onclick=(e)=>{if(e.target.classList.contains("retry")||waiting||retrying)return;selected.has(i)?selected.delete(i):selected.add(i);renderGallery();updateSelection()};
   const rb=tile.querySelector(".retry"); if(rb) rb.onclick=(e)=>{e.stopPropagation();retryVariant(i)};
   g.appendChild(tile);
 });
 updateSelection();
}
function updateSelection(){$("selectedText").textContent=selected.size+" selected"}
async function retryVariant(i){
 const item=generated[i]; if(!item||item.state==="retrying"||item.state==="waiting")return;
 item.state="retrying"; selected.delete(i); renderGallery();
 try{
   const r=await fetch("/api/retry",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:$("name").value,index:i,strength:$("strength").value,width:$("width").value,height:$("height").value})});
   const d=await r.json(); if(!r.ok)throw Error(d.error||"Retry failed");
   if(!pollTimer)pollTimer=setInterval(poll,500);
 }catch(e){item.state="ready";renderGallery();toast(e.message)}
}
$("saveDatasetBtn").onclick=async()=>{if(!selected.size)return toast("Выберите изображения для Dataset.");try{const r=await fetch("/api/save-dataset",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:$("name").value,indices:[...selected]})});const d=await r.json();if(!r.ok)throw Error(d.error||"Save failed");toast(d.message);selected.clear();updateSelection()}catch(e){toast(e.message)}}
$("saveReadyBtn").onclick=async()=>{const ready=generated.map((x,i)=>x.state==="ready"?i:-1).filter(i=>i>=0);if(!ready.length)return toast("Пока нет готовых изображений.");try{const r=await fetch("/api/save-dataset",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:$("name").value,indices:ready})});const d=await r.json();if(!r.ok)throw Error(d.error||"Save failed");toast(d.message);selected.clear();renderGallery()}catch(e){toast(e.message)}}
$("approveBtn").onclick=async()=>{if(!selected.size)return toast("Выберите изображения.");try{const r=await fetch("/api/approve",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:$("name").value,indices:[...selected]})});const d=await r.json();if(!r.ok)throw Error(d.error);toast("Добавлено в Approved: "+d.saved);selected.clear();renderGallery()}catch(e){toast(e.message)}}
$("clearBtn").onclick=()=>{selected.clear();renderGallery()}

$("comfyBtn").onclick=async()=>{try{const d=await(await fetch("/api/launch-comfy",{method:"POST"})).json();toast(d.message);setTimeout(checkStatus,2500)}catch(e){toast("Не удалось запустить ComfyUI")}}
async function checkStatus(){try{const d=await(await fetch("/api/status")).json();$("statPipeline").textContent=d.online?"Online":"Offline";$("statPipelineNote").textContent=d.online?"ComfyUI connected":"ComfyUI required";$("onlineBadge").style.opacity=d.online?"1":".55"}catch(e){}}
refreshStats();checkStatus();setInterval(checkStatus,5000);
</script>
</body></html>
'''

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading
import socket
import webbrowser
import subprocess

JOB = {"id": None, "state": "idle", "progress": 0, "message": "Ready", "files": [], "items": [], "error": None, "retrying": {}}
JOB_LOCK = threading.Lock()

def set_job(**kwargs):
    with JOB_LOCK:
        JOB.update(kwargs)

def run_generation(params, source_bytes, source_filename):
    try:
        import tempfile
        total = max(1, int(params.get("count", 4)))
        base = PROMPT_BASE
        negative = NEGATIVE_PROMPT
        clothing = str(params.get("clothing", "casual clothes"))
        scene = str(params.get("scene", "studio"))
        pose = str(params.get("pose", "portrait"))
        extra = str(params.get("extra", "")).strip()
        prompts = [make_prompt(clothing, scene, pose, extra, base, i) for i in range(total)]
        items = [{"url": None, "prompt": prompts[i], "state": "waiting"} for i in range(total)]
        set_job(state="running", progress=1, message=f"Подготовка {total} вариантов…", files=[], items=items, retrying={}, error=None)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(source_filename).suffix or ".png") as tmp:
            tmp.write(source_bytes); tmp_path = tmp.name
        try:
            def on_item(i, filename, prompt, total_count):
                rel = "/media/" + str(Path(filename).relative_to(CHAR_ROOT)).replace("\\", "/")
                with JOB_LOCK:
                    current = list(JOB.get("items", []))
                    if i < len(current):
                        current[i] = {"url": rel, "prompt": prompt, "state": "ready"}
                    JOB["items"] = current
                    JOB["files"] = [x.get("url") for x in current if x.get("url")]
                    JOB["progress"] = round(5 + 90 * ((i + 1) / max(total_count, 1)), 1)
                    JOB["message"] = f"Вариант {i + 1} из {total_count} готов"

            set_job(progress=2, message="Загружаю исходное фото…")
            files, _ = generate(
                params["name"], tmp_path, total, float(params["strength"]),
                int(params["width"]), int(params["height"]),
                clothing=clothing, scene=scene, pose=pose, extra=extra,
                progress=lambda value, message: set_job(progress=round(float(value)*100,1), message=message),
                on_item=on_item, prompt_base=base, negative_prompt=negative
            )
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass
        with JOB_LOCK:
            final_items = list(JOB.get("items", []))
            JOB["state"] = "done"
            JOB["progress"] = 100
            JOB["message"] = f"{len(final_items)} variants ready · 100%"
            JOB["files"] = [x.get("url") for x in final_items if x.get("url")]
            JOB["items"] = final_items
            JOB["retrying"] = {}
            JOB["error"] = None
    except Exception as e:
        set_job(state="error", progress=0, message="Generation failed", error=str(e))

class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self.reply(200, HTML, "text/html; charset=utf-8")
        if path == "/api/job":
            with JOB_LOCK:
                return self.reply(200, json.dumps(JOB))
        if path == "/api/status":
            return self.reply(200, json.dumps({"online": comfy_online(), "site_root": str(SITE_ROOT), "characters_root": str(CHAR_ROOT)}))
        if path == "/api/dataset":
            q = dict(x.split("=",1) if "=" in x else (x,"") for x in urlparse(self.path).query.split("&") if x)
            items, data = dataset_items(q.get("name","Character_01"), q.get("scope","approved"))
            approved = data["approved"]
            captions = sum(1 for x in approved if image_caption(x))
            missing = len(approved) - captions
            return self.reply(200, json.dumps({"items":items,"summary":{"approved":len(approved),"generated":len(data["generated"]),"source":len(data["source"]),"captions":captions,"missing":missing}}, ensure_ascii=False))
        if path.startswith("/media/"):
            rel = path[len("/media/"):]
            target = (CHAR_ROOT / rel).resolve()
            root = CHAR_ROOT.resolve()
            if not str(target).startswith(str(root)) or not target.exists():
                return self.reply(404, "Not found", "text/plain")
            mime = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",".webp":"image/webp"}.get(target.suffix.lower(),"application/octet-stream")
            return self.reply(200, target.read_bytes(), mime)
        return self.reply(404, json.dumps({"error":"Not found"}))

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/generate":
            return self.handle_generate()
        if path == "/api/retry":
            return self.handle_retry()
        if path == "/api/save-dataset":
            return self.handle_save_dataset()
        if path == "/api/approve":
            return self.handle_approve()
        if path == "/api/launch-comfy":
            return self.handle_launch()
        if path == "/api/caption":
            return self.handle_caption()
        return self.reply(404, json.dumps({"error":"Not found"}))

    def handle_generate(self):
        ctype = self.headers.get("Content-Type","")
        if "multipart/form-data" not in ctype:
            return self.reply(400, json.dumps({"error":"Expected multipart/form-data"}))
        content_length = int(self.headers.get("Content-Length","0"))
        body = self.rfile.read(content_length)
        match = re.search(r'boundary=([^;]+)', ctype)
        if not match:
            return self.reply(400, json.dumps({"error":"Multipart boundary missing"}))
        boundary = match.group(1).strip().strip('"').encode()
        marker = b"--" + boundary
        parts = body.split(marker)

        fields = {}
        image_data = None
        image_filename = "source.png"

        for part in parts[1:]:
            part = part.lstrip(b"\r\n")
            if not part or part.startswith(b"--"):
                continue
            header_end = part.find(b"\r\n\r\n")
            if header_end < 0:
                continue
            raw_headers = part[:header_end].decode("utf-8", "replace")
            value = part[header_end + 4:]
            if value.endswith(b"\r\n"):
                value = value[:-2]

            name_match = re.search(r'name="([^"]+)"', raw_headers)
            if not name_match:
                continue
            field_name = name_match.group(1)
            filename_match = re.search(r'filename="([^"]*)"', raw_headers)

            if filename_match:
                image_filename = Path(filename_match.group(1)).name or "source.png"
                image_data = value
            else:
                fields[field_name] = value.decode("utf-8", "replace")

        if image_data is None:
            return self.reply(400, json.dumps({"error":"Reference image missing"}))

        with JOB_LOCK:
            if JOB["state"] == "running":
                return self.reply(409, json.dumps({"error":"Generation already running"}))

        params = {k: fields.get(k, "") for k in
                  ["name","count","strength","width","height","clothing","scene","pose","extra"]}
        set_job(id=str(uuid.uuid4()), state="queued", progress=0,
                message="Queued…", files=[], error=None)
        threading.Thread(
            target=run_generation,
            args=(params, image_data, image_filename),
            daemon=True
        ).start()
        return self.reply(200, json.dumps({"ok":True}))

    def handle_retry(self):
        try:
            n=int(self.headers.get("Content-Length","0")); payload=json.loads(self.rfile.read(n) or "{}")
            name=safe_name(payload.get("name","Character_01")); slot=int(payload.get("index",-1))
            strength=float(payload.get("strength",0.68)); width=int(payload.get("width",512)); height=int(payload.get("height",768)); negative_prompt=str(payload.get("negative_prompt","")).strip() or NEGATIVE_PROMPT
            with JOB_LOCK:
                items=list(JOB.get("items",[]))
                if slot < 0 or slot >= len(items): raise RuntimeError("Вариант не найден.")
                if items[slot].get("state")=="retrying": raise RuntimeError("Этот вариант уже генерируется.")
                prompt=items[slot].get("prompt","")
                old_url=items[slot].get("url","")
                items[slot]["state"]="retrying"
                JOB["items"]=items; JOB.setdefault("retrying",{})[str(slot)]=True; JOB["state"]="retrying"; JOB["message"]=f"Вариант {slot+1}: генерирую новую версию…"
            def worker():
                old_path=None
                try:
                    old_rel=old_url[len("/media/"):] if old_url.startswith("/media/") else ""
                    if old_rel: old_path=(CHAR_ROOT/old_rel).resolve()
                    new_file=regenerate_slot(name,slot,prompt,strength,width,height,negative_prompt)
                    new_url="/media/"+str(Path(new_file).relative_to(CHAR_ROOT)).replace("\\","/")
                    if old_path and old_path.exists():
                        try: old_path.unlink()
                        except OSError: pass
                        old_txt=old_path.with_suffix(".txt")
                        if old_txt.exists():
                            try: old_txt.unlink()
                            except OSError: pass
                    with JOB_LOCK:
                        JOB["items"][slot]={"url":new_url,"prompt":prompt,"state":"ready"}
                        JOB["files"]=[x.get("url") for x in JOB["items"] if x.get("url")]
                        JOB.setdefault("retrying",{}).pop(str(slot),None)
                        JOB["state"]="retrying" if JOB.get("retrying") else "done"
                        JOB["progress"]=100
                        JOB["message"]=f"Вариант {slot+1} обновлён" if JOB["state"]=="done" else f"Вариант {slot+1} обновлён · ещё есть генерация"
                except Exception as e:
                    with JOB_LOCK:
                        JOB["items"][slot]["state"]="ready"
                        JOB.setdefault("retrying",{}).pop(str(slot),None)
                        JOB["state"]="retrying" if JOB.get("retrying") else "done"
                        JOB["error"]=str(e); JOB["message"]=f"Ошибка повтора варианта {slot+1}: {e}"
            threading.Thread(target=worker,daemon=True).start()
            return self.reply(200,json.dumps({"ok":True}))
        except Exception as e:
            return self.reply(400,json.dumps({"error":str(e)}))

    def handle_save_dataset(self):
        try:
            n=int(self.headers.get("Content-Length","0")); payload=json.loads(self.rfile.read(n) or "{}")
            name=safe_name(payload.get("name","Character_01")); indices=[int(x) for x in payload.get("indices",[])]
            with JOB_LOCK: items=list(JOB.get("items",[]))
            if not indices: raise RuntimeError("Выберите изображения для сохранения.")
            target=char_dir(name)/"approved"; saved=0
            for idx in indices:
                if 0 <= idx < len(items) and items[idx].get("state") == "ready" and items[idx].get("url"):
                    rel=items[idx].get("url","")[len("/media/"):]
                    src=(CHAR_ROOT/rel).resolve()
                    if not src.exists() or not str(src).startswith(str(CHAR_ROOT.resolve())): continue
                    dst=target/src.name; shutil.copy2(src,dst)
                    prompt=items[idx].get("prompt","")
                    dst.with_suffix(".txt").write_text(prompt,encoding="utf-8")
                    saved+=1
            return self.reply(200,json.dumps({"saved":saved,"message":f"Сохранено в Dataset: {saved}"},ensure_ascii=False))
        except Exception as e:
            return self.reply(400,json.dumps({"error":str(e)},ensure_ascii=False))

    def handle_approve(self):
        try:
            n = int(self.headers.get("Content-Length","0"))
            payload = json.loads(self.rfile.read(n) or "{}")
            name = safe_name(payload.get("name","Character_01"))
            indices = payload.get("indices",[])
            with JOB_LOCK:
                files = list(JOB.get("files",[]))
                items = list(JOB.get("items",[]))
            target = char_dir(name) / "approved"
            saved = 0
            for idx in indices:
                idx = int(idx)
                if 0 <= idx < len(files):
                    rel = files[idx][len("/media/"):]
                    src = (CHAR_ROOT / rel).resolve()
                    if src.exists() and str(src).startswith(str(CHAR_ROOT.resolve())):
                        dst = target / src.name
                        shutil.copy2(src, dst)
                        prompt = items[idx].get("prompt", "")
                        dst.with_suffix(".txt").write_text(prompt, encoding="utf-8")
                        saved += 1
            return self.reply(200, json.dumps({"saved":saved}))
        except Exception as e:
            return self.reply(400, json.dumps({"error":str(e)}))


    def handle_caption(self):
        try:
            n=int(self.headers.get("Content-Length","0"))
            payload=json.loads(self.rfile.read(n) or "{}")
            image_url=payload.get("image_url","")
            rel=image_url[len("/media/"):] if image_url.startswith("/media/") else ""
            target=(CHAR_ROOT/rel).resolve()
            root=CHAR_ROOT.resolve()
            if not rel or not target.exists() or not str(target).startswith(str(root)):
                raise RuntimeError("Изображение не найдено.")
            save_image_caption(target,payload.get("caption",""))
            return self.reply(200,json.dumps({"ok":True}))
        except Exception as e:
            return self.reply(400,json.dumps({"error":str(e)}))

    def handle_launch(self):
        if comfy_online():
            return self.reply(200, json.dumps({"message":"ComfyUI уже запущен."}))
        py = ROOT / "ComfyUI" / "venv" / "Scripts" / "python.exe"
        main = ROOT / "ComfyUI" / "main.py"
        if not py.exists() or not main.exists():
            return self.reply(404, json.dumps({"message":"ComfyUI не найден в C:\\AI\\ComfyUI."}))
        subprocess.Popen([str(py),str(main)], cwd=str(ROOT/"ComfyUI"), creationflags=subprocess.CREATE_NEW_CONSOLE)
        return self.reply(200, json.dumps({"message":"Запускаю ComfyUI…"}))

    def log_message(self, fmt, *args):
        print("[Character Lab]", fmt % args)

def find_free_port(start=7860, end=7880):
    for port in range(start,end+1):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1",port))
                return port
            except OSError:
                pass
    raise OSError(f"Нет свободного порта {start}-{end}.")

if __name__ == "__main__":
    port = find_free_port()
    print(f"Character Lab: http://127.0.0.1:{port}")
    server = ThreadingHTTPServer(("127.0.0.1",port), Handler)
    threading.Timer(.7, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    server.serve_forever()