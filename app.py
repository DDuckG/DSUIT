# app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FULLUS — Desktop UI (16:9) at http://127.0.0.1:8888
(đã rút gọn mô tả)
"""
import os
import sys
import shutil
import time
import threading
import subprocess
from collections import deque
from datetime import datetime
from queue import Queue, Empty
from flask import Flask, request, send_from_directory, Response, jsonify, render_template_string, abort

try:
    from werkzeug.utils import secure_filename
except Exception:
    def secure_filename(name):
        return ''.join(c for c in name if c.isalnum() or c in (' ','-','_','.')).strip().replace(' ','_')

try:
    import yaml
except Exception:
    yaml = None

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw_videos")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
CONF_DIR = os.path.join(BASE_DIR, "configs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
for p in (DATA_DIR, OUT_DIR, CONF_DIR, STATIC_DIR, os.path.join(STATIC_DIR, "snd")):
    os.makedirs(p, exist_ok=True)

DEFAULT_CONFIG = os.path.join(CONF_DIR, "config.yaml")

# Bật static để phục vụ mp3: /static/...
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
LOGS = deque(maxlen=5000)
CLIENTS: list[Queue] = []
JOBS: dict[str, dict] = {}   # {"a1.mp4": {"ready": bool, "output_url": str, "web_url": Optional[str], "events_url": Optional[str]}}

def log(msg: str):
    line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
    LOGS.append(line)
    for q in list(CLIENTS):
        try: q.put_nowait(line)
        except Exception: pass

def write_runtime_config(base_cfg_path, fps, width, height, cam_h, sound_on):
    os.makedirs(os.path.join(BASE_DIR, "runtime_configs"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "runtime_configs", f"run_{int(time.time())}.yaml")
    if yaml is not None:
        try:
            with open(base_cfg_path, "r", encoding="utf-8") as f:
                base = yaml.safe_load(f) or {}
        except Exception:
            base = {}
        base.setdefault("io", {})
        base.setdefault("camera", {})
        base.setdefault("audio", {})
        base["io"]["output_fps"]  = int(fps)
        base["io"]["output_size"] = [int(width), int(height)]
        base["camera"]["height_m"] = float(cam_h)
        base["audio"]["enabled"] = bool(sound_on)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(base, f, sort_keys=False)
    else:
        cfg_txt = (
            f"io:\n  output_fps: {int(fps)}\n  output_size: [{int(width)}, {int(height)}]\n"
            f"camera:\n  height_m: {float(cam_h)}\n"
            f"audio:\n  enabled: {str(bool(sound_on)).lower()}\n"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cfg_txt)
    return out_path

def transcode_to_web_safe(src_mp4: str) -> tuple[bool, str]:
    ffmpeg = shutil.which("ffmpeg")
    web_path = os.path.splitext(src_mp4)[0] + "_web.mp4"

    # Sanity checks
    if not os.path.exists(src_mp4):
        log(f"[UI] transcode skipped: input not found: {src_mp4}")
        return False, web_path
    try:
        size = os.path.getsize(src_mp4)
    except Exception:
        size = 0
    if size <= 0:
        log(f"[UI] transcode skipped: input is empty: {src_mp4}")
        return False, web_path

    if not ffmpeg:
        log("[UI] ffmpeg not found; serving original file (codec may be unsupported).")
        return False, web_path

    try:
        cmd = [
            ffmpeg, "-y", "-i", src_mp4,
            "-movflags", "+faststart",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline", "-level", "3.1",
            "-preset", "veryfast", "-tune", "fastdecode",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-an",
            web_path
        ]
        log("[UI] ffmpeg: " + " ".join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = "\n".join(proc.stdout.splitlines()[-8:])
        for ln in tail.splitlines():
            LOGS.append(ln)
            for q in list(CLIENTS):
                try: q.put_nowait(ln)
                except Exception: pass

        if proc.returncode == 0 and os.path.exists(web_path) and os.path.getsize(web_path) > 0:
            log(f"[UI] Transcoded to web-safe: {web_path}")
            return True, web_path
        else:
            log(f"[UI] ffmpeg failed with code {proc.returncode}, serving original.")
            return False, web_path
    except Exception as e:
        log(f"[UI] ffmpeg error: {e}")
        return False, web_path
    
def _wait_for_file_ready(path: str, timeout_sec: float = 3.0, min_size: int = 8_192) -> bool:
    """
    Wait (briefly) until `path` exists and is at least `min_size` bytes.
    Returns True if ready, False if timed out.
    """
    t0 = time.time()
    last_size = -1
    while time.time() - t0 < timeout_sec:
        if os.path.exists(path):
            try:
                sz = os.path.getsize(path)
            except Exception:
                sz = 0
            if sz >= min_size:
                return True
            # If size is growing, small sleep and keep waiting
            if sz != last_size:
                last_size = sz
        time.sleep(0.12)
    return os.path.exists(path) and os.path.getsize(path) >= min_size


def run_pipeline_async(job_name: str, src_path: str, out_path: str, cfg_path: str):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-u", "-m", "src.main", "--src", src_path, "--out", out_path, "--config", cfg_path]
    log("cmd: " + " ".join(cmd))
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, env=env) as p:
            for line in p.stdout:
                if not line: 
                    continue
                line = line.rstrip("\n")
                LOGS.append(line)
                for q in list(CLIENTS):
                    try: q.put_nowait(line)
                    except Exception: pass
            rc = p.wait()

        if rc == 0:
            # Double-check the output file actually exists and has non-trivial size
            if not _wait_for_file_ready(out_path, timeout_sec=3.0, min_size=8_192):
                log(f"[UI] ERROR: pipeline returned rc=0 but output not found or empty: {out_path}")
                try:
                    # Helpful diagnostics
                    out_dir = os.path.dirname(out_path) or "."
                    listing = ", ".join(os.listdir(out_dir))
                    log(f"[UI] outputs dir listing: {out_dir}: {listing}")
                except Exception as e:
                    log(f"[UI] could not list outputs dir: {e}")
                JOBS.setdefault(job_name, {}).update({"ready": False})
                return

            JOBS.setdefault(job_name, {}).update({"ready": True})
            log(f"OUTPUT_READY {out_path}")

            ok, web_path = transcode_to_web_safe(out_path)
            if ok:
                rel = os.path.basename(web_path)
                JOBS[job_name]["web_url"] = f"/video/output_web/{rel}"
                log(f"OUTPUT_READY_WEB {web_path}")
            else:
                JOBS[job_name]["web_url"] = None
        else:
            log(f"ERROR pipeline returned code {rc}")
            JOBS.setdefault(job_name, {}).update({"ready": False})

    except FileNotFoundError:
        log("ERROR: Could not launch 'python -m src.main'. Ensure your environment is set up.")
        JOBS.setdefault(job_name, {}).update({"ready": False})
    except Exception as e:
        log(f"ERROR: {e}")
        JOBS.setdefault(job_name, {}).update({"ready": False})
        
@app.route("/")
def index():
    return render_template_string(INDEX_HTML, default_fps=30, default_w=1280, default_h=720, default_cam_h=1.35)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    fps = int(request.form.get("fps", 30))
    width = int(request.form.get("width", 1280))
    height = int(request.form.get("height", 720))
    cam_h = float(request.form.get("cam_h", 1.35))
    sound_on = request.form.get("sound_on", "true").lower() == "true"
    base_cfg = request.form.get("config", DEFAULT_CONFIG)

    if not file or file.filename == "": return jsonify({"ok": False, "error": "No file provided."}), 400
    name = secure_filename(file.filename)
    if not name.lower().endswith(".mp4"): return jsonify({"ok": False, "error": "Only .mp4 files are accepted."}), 400

    in_path = os.path.join(DATA_DIR, name)
    out_path = os.path.join(OUT_DIR, name)
    os.makedirs(os.path.dirname(in_path), exist_ok=True)
    file.save(in_path)
    log(f"[UI] received file -> {in_path}")
    log(f"[UI] sound={'on' if sound_on else 'off'} fps={fps} size={width}x{height} cam_h={cam_h}")

    JOBS[name] = {"ready": False, "output_url": f"/video/output/{name}", "web_url": None, "events_url": None}
    cfg_path = write_runtime_config(base_cfg, fps, width, height, cam_h, sound_on)
    threading.Thread(target=run_pipeline_async, args=(name, in_path, out_path, cfg_path), daemon=True).start()
    return jsonify({"ok": True, "job": name, "input_url": f"/video/input/{name}", "output_url": JOBS[name]["output_url"]})

@app.route("/status/<job>")
def job_status(job: str):
    info = JOBS.get(job)
    if not info: return jsonify({"exists": False, "ready": False})
    return jsonify({
        "exists": True,
        "ready": bool(info.get("ready", False)),
        "output_url": info.get("output_url"),
        "web_url": info.get("web_url"),
        "events_url": info.get("events_url"),
    })

@app.route("/video/input/<path:fname>")
def video_input(fname): return send_from_directory(DATA_DIR, fname, as_attachment=False)

@app.route("/video/output/<path:fname>")
def video_output(fname):
    full = os.path.join(OUT_DIR, fname)
    if not os.path.exists(full): abort(404)
    return send_from_directory(OUT_DIR, fname, as_attachment=False)

@app.route("/video/output_web/<path:fname>")
def video_output_web(fname):
    full = os.path.join(OUT_DIR, fname)
    if not os.path.exists(full): abort(404)
    return send_from_directory(OUT_DIR, fname, as_attachment=False)

@app.route("/logs")
def sse_logs():
    q = Queue(); CLIENTS.append(q)
    def stream():
        for ln in list(LOGS)[-80:]: yield f"data: {ln}\n\n"
        try:
            last = time.time()
            while True:
                try:
                    ln = q.get(timeout=0.5); yield f"data: {ln}\n\n"
                except Empty:
                    if time.time()-last>15: yield ": keepalive\n\n"; last=time.time()
        except GeneratorExit: pass
        finally:
            try: CLIENTS.remove(q)
            except ValueError: pass
    return Response(stream(), mimetype="text/event-stream")

INDEX_HTML = r"""
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FULLUS — Obstacle Warning System</title>
<style>
  :root{
    --ink:#31445B; --muted:#506483; --border:#D2DCE8;
    --card:#FFFFFF; --mint:#A8E6CF; --blue:#A1C2F2;
    --mintLight:#E2F4EC; --lavLight:#F6F6FA;
    --gutter:24px; --radius:24px; --thin:2px; --container: 1280px;
  }
  html,body{height:100%; margin:0; font-family: 'Open Sans', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:var(--ink);}
  body{
    background: radial-gradient(1200px 800px at 10% 5%, rgba(220,236,255,0.9), transparent 60%),
                radial-gradient(1000px 700px at 95% 8%, rgba(246,236,252,0.85), transparent 60%),
                radial-gradient(1000px 700px at 85% 85%, rgba(226,244,236,0.8), transparent 60%),
                linear-gradient(135deg, #E0ECFF, #E8F3F0 40%, #F6ECFC);
  }
  .wrap{max-width: var(--container); margin: 32px auto 48px auto; padding: 0 24px;}
  header{ text-align:center; margin-bottom: 32px;}
  header .title{ font-family: Poppins, system-ui, sans-serif; font-weight: 800; letter-spacing: 2px; font-size: 64px; }
  header .subtitle{ margin-top: 6px; font-family: Poppins, system-ui, sans-serif; font-weight: 500; letter-spacing: 1px; font-size: 22px; }
  .grid{ display:grid; grid-template-columns: repeat(12, 1fr); gap: var(--gutter);}
  .card{ background: var(--card); border: var(--thin) solid var(--border); border-radius: var(--radius); box-shadow: 0 10px 30px rgba(28,49,74,0.08); }
  .video-card{ position: relative; padding: 16px; }
  .video-inner{ position: relative; width: 100%; aspect-ratio: 16/9; border-radius: 18px; border: var(--thin) solid #BECDDA; overflow: hidden; background: #000; }
  .video-inner.mint{ background: #000; }
  .video-label{ position:absolute; left: 24px; bottom: 20px; font-weight: 700; font-size: 20px; color:#a9b7c6;}
  .video-inner video{ width:100%; height:100%; object-fit: cover; display:block; background:#000; }
  #uploadArea{ cursor: pointer; }
  #outputArea video{ pointer-events: none; }
  #outputArea{ cursor: not-allowed; }
  .ready-overlay{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; pointer-events:auto; }
  .ready-overlay.hidden{ display:none; }
  .bigplay{ width: 96px; height: 96px; border-radius: 999px; background: #ffffffcc; border: 1px solid var(--border); display:flex; align-items:center; justify-content:center; box-shadow: 0 12px 30px rgba(28,49,74,.18); cursor: pointer; font-weight: 800; }
  .bigplay:active{ transform: translateY(1px); }
  .output-ctl{ position:absolute; right:16px; top:16px; display:flex; gap:8px; pointer-events:auto; }
  .btn{ appearance:none; border:1px solid var(--border); background:#F5F8FD; padding:8px 12px; border-radius:12px; font-weight:700; color:var(--ink); cursor:pointer; box-shadow: 0 2px 8px rgba(28,49,74,.06); }
  .btn:active{ transform: translateY(1px); }
  .parameters.card{ padding: 20px; }
  .pill-row{ display:flex; gap:12px; margin: 12px 0 6px 0; flex-wrap:wrap; }
  .pill{ display:flex; align-items:center; gap:12px; padding: 12px 14px; border:1px solid var(--border); border-radius:14px; background:#F5F8FD; }
  .pill input{ width: 72px; padding:8px 10px; border:1px solid var(--border); border-radius:10px; font-weight:600; color:var(--ink); background:#fff; }
  .pill .w{ width: 88px; } .pill .h{ width: 88px; }
  .toggle.card{ padding: 20px;}
  .toggle-row{ display:flex; align-items:center; justify-content:space-between; gap:16px; }
  .switch{ position:relative; width: 86px; height: 44px; background:#D2DCE8; border-radius: 22px; transition: all .18s ease; border: 1px solid var(--border); }
  .switch .knob{ position:absolute; top:3px; left:3px; width:38px; height:38px; background:#fff; border-radius: 50%; transition: all .18s ease; box-shadow: 0 2px 4px rgba(0,0,0,.15); }
  .switch.on{ background:#A8E6CF; } .switch.on .knob{ left: 45px; }
  .log.card{ padding: 20px; background: #F6F6FA; }
  #logbox{ height: 220px; overflow:auto; font-family: ui-monospace, SFMono, Menlo, Consolas, monospace; font-size: 13px; }
  #leftCol{ grid-column: span 8; } #rightCol{ grid-column: span 4; }
  @media (max-width: 1200px){ #leftCol{ grid-column: span 12; } #rightCol{ grid-column: span 12; } }
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="title">FULLUS</div>
      <div class="subtitle">OBSTACLE WARNING SYSTEM FOR THE VISUALLY IMPAIRED</div>
    </header>
    <main class="grid">
      <section id="leftCol">
        <div id="uploadArea" class="card video-card">
          <div class="video-inner" id="uploadInner">
            <video id="uploadVideo" controls muted playsinline></video>
          </div>
          <div class="video-label">upload</div>
          <input id="fileInput" type="file" accept="video/mp4" style="display:none"/>
        </div>
        <div id="outputArea" class="card video-card" style="margin-top: 24px">
          <div class="video-inner mint">
            <video id="outputVideo" preload="auto" playsinline muted></video>
            <div id="readyOverlay" class="ready-overlay hidden">
              <div id="bigPlay" class="bigplay" title="Play">▶</div>
            </div>
            <div class="output-ctl">
              <button id="playBtn" class="btn">Play</button>
              <button id="pauseBtn" class="btn">Pause</button>
            </div>
          </div>
          <div class="video-label">output</div>
        </div>
      </section>
      <aside id="rightCol" class="grid" style="grid-template-columns: 1fr; gap: 24px">
        <div class="parameters card">
          <div style="font-weight:800; font-size:22px; margin-bottom:4px">Parameters</div>
          <div class="pill-row">
            <div class="pill">fps <input id="fps" type="number" min="1" step="1" value="{{ default_fps }}"/></div>
            <div class="pill">resolution <input id="width" class="w" type="number" min="160" step="2" value="{{ default_w }}"/> × <input id="height" class="h" type="number" min="90" step="2" value="{{ default_h }}"/></div>
            <div class="pill">camera height <input id="cam_h" type="number" step="0.01" value="{{ default_cam_h }}"/></div>
          </div>
        </div>
        <div class="toggle card">
          <div class="toggle-row">
            <div style="font-weight:800; font-size:22px;">Sound</div>
            <div id="soundSwitch" class="switch on" role="switch" aria-checked="true" tabindex="0" onclick="toggleSound()"><div class="knob"></div></div>
          </div>
          <div style="color:#506483; margin-top:8px">Toggle warning sound</div>
        </div>
        <div class="log card">
          <div style="font-weight:800; font-size:22px; margin-bottom:6px">log</div>
          <div id="logbox"></div>
        </div>
      </aside>
    </main>
  </div>
<script>
  // --- Sound toggle ---
  let SOUND_ON = true;
  function toggleSound(){ SOUND_ON=!SOUND_ON; const el=document.getElementById('soundSwitch'); el.classList.toggle('on',SOUND_ON); el.setAttribute('aria-checked',SOUND_ON?'true':'false'); }

  const uploadArea=document.getElementById('uploadArea');
  const fileInput=document.getElementById('fileInput');
  const uploadVideo=document.getElementById('uploadVideo');
  const outputVideo=document.getElementById('outputVideo');
  const logbox=document.getElementById('logbox');
  const playBtn=document.getElementById('playBtn');
  const pauseBtn=document.getElementById('pauseBtn');
  const readyOverlay=document.getElementById('readyOverlay');
  const bigPlay=document.getElementById('bigPlay');

  // Track current URLs to switch/fallback on error
  let OUTPUT_URL=null, WEB_URL=null, CURRENT_SRC=null, ATTACHED=false;

  // --- Alert events ---
  let EVENTS=[];      // {t, level, message}
  let EV_IDX=0;
  let EVENTS_URL=null;

  // Preload audio
  const warnAudio = new Audio('/static/snd/canthan.mp3');
  const dangerAudio = new Audio('/static/snd/nguyhiem.mp3');

  uploadArea.addEventListener('click', ()=>fileInput.click());

  fileInput.addEventListener('change', async (ev)=>{
    const f=ev.target.files[0]; if(!f) return;
    if(!f.name.toLowerCase().endsWith('.mp4')){ alert('Only .mp4 files are accepted'); return; }
    const url=URL.createObjectURL(f); uploadVideo.src=url; uploadVideo.play().catch(()=>{});
    const fd=new FormData();
    fd.append('file', f);
    fd.append('fps', document.getElementById('fps').value||30);
    fd.append('width', document.getElementById('width').value||1280);
    fd.append('height', document.getElementById('height').value||720);
    fd.append('cam_h', document.getElementById('cam_h').value||1.35);
    fd.append('sound_on', SOUND_ON?'true':'false');
    const r=await fetch('/upload',{method:'POST',body:fd});
    const j=await r.json();
    if(!j.ok){ alert('Upload failed: '+(j.error||'unknown')); return; }
    window.__job=j.job; OUTPUT_URL=j.output_url; WEB_URL=null; EVENTS_URL=null; EVENTS=[]; EV_IDX=0; ATTACHED=false;
    pollJobUntilReady(j.job);
  });

  // --- Logs via SSE ---
  function appendLog(line){
    const pre=document.createElement('div'); pre.textContent=line; logbox.appendChild(pre);
    logbox.scrollTop=logbox.scrollHeight;

    // Prefer web-ready signal
    if(line.startsWith('OUTPUT_READY_WEB')){
      if(window.__job){ fetch('/status/'+encodeURIComponent(window.__job)).then(r=>r.json()).then(s=>{
        if(s.web_url){ WEB_URL=s.web_url; tryAttachPreferWeb(); }
      }); }
      return;
    }
    if(line.startsWith('OUTPUT_READY ')){
      setTimeout(()=>{ if(!ATTACHED) tryAttachPreferWeb(true); }, 3000);
    }
    if(line.startsWith('ALERTS_READY ')){
      if(window.__job){ fetch('/status/'+encodeURIComponent(window.__job)).then(r=>r.json()).then(s=>{
        if(s.events_url){ EVENTS_URL=s.events_url; fetchEvents(); }
      }); }
    }
  }
  try{ const es=new EventSource('/logs'); es.onmessage=(e)=>appendLog(e.data); }catch(e){ console.warn('SSE failed',e); }

  // --- Status poller ---
  async function pollJobUntilReady(job){
    let seenReady=false, waited=0;
    while(true){
      const r=await fetch('/status/'+encodeURIComponent(job));
      if(!r.ok) break;
      const s=await r.json();
      if(s.exists && s.ready){
        seenReady=true;
        if(s.events_url && !EVENTS_URL){ EVENTS_URL=s.events_url; fetchEvents(); }
        if(s.web_url){ WEB_URL=s.web_url; tryAttachPreferWeb(); return; }
        OUTPUT_URL=s.output_url || OUTPUT_URL;
        waited+=2;
        if(waited>=10){ tryAttachPreferWeb(true); return; }
      }
      await new Promise(res=>setTimeout(res, 2000));
    }
  }

  async function fetchEvents(){
    try{
      const r=await fetch(EVENTS_URL+'?v='+Date.now());
      if(!r.ok) return;
      const j=await r.json();
      EVENTS = Array.isArray(j.events) ? j.events.slice().sort((a,b)=>a.t-b.t) : [];
      EV_IDX=0;
      // Log meta
      if(j.meta){
        appendLog('[ALERTS] loaded: events='+EVENTS.length+' roi=['+(j.meta.roi_frac||[]).join(',')+']');
      }else{
        appendLog('[ALERTS] loaded: events='+EVENTS.length);
      }
    }catch(e){ console.warn('fetch events error',e); }
  }

  function urlWithBust(u){ if(!u) return u; const v=Date.now(); return u+(u.includes('?')?'&':'?')+'v='+v; }

  function hardAttach(url){
    if(!url) return;
    ATTACHED=true; CURRENT_SRC=url;
    outputVideo.pause();
    outputVideo.removeAttribute('src');
    outputVideo.load();
    outputVideo.src=urlWithBust(url);
    outputVideo.load();

    readyOverlay.classList.remove('hidden');

    const onCanPlay=()=>{ outputVideo.play().catch(()=>{}); };
    const onPlay=()=>{ readyOverlay.classList.add('hidden'); };

    outputVideo.addEventListener('canplay', onCanPlay, {once:true});
    outputVideo.addEventListener('play', onPlay, {once:true});
  }

  function tryAttachPreferWeb(allowFallback=false){
    if(WEB_URL){ hardAttach(WEB_URL); return; }
    if(allowFallback && OUTPUT_URL){ hardAttach(OUTPUT_URL); }
  }

  // Fallback when <video> fires error
  outputVideo.addEventListener('error', ()=>{
    const msg=document.createElement('div'); msg.textContent='[UI] video element error'; logbox.appendChild(msg);
    if(CURRENT_SRC && WEB_URL && CURRENT_SRC!==WEB_URL){ hardAttach(WEB_URL); return; }
    if(CURRENT_SRC && OUTPUT_URL && CURRENT_SRC!==OUTPUT_URL){ hardAttach(OUTPUT_URL); return; }
  });

  // External controls
  document.getElementById('playBtn').addEventListener('click', ()=>{ outputVideo.play().catch(()=>{}); });
  document.getElementById('pauseBtn').addEventListener('click', ()=>{ try{ outputVideo.pause(); }catch(e){} });
  document.getElementById('bigPlay').addEventListener('click', ()=>{ outputVideo.play().catch(()=>{}); });

  // === Phát cảnh báo đúng mốc thời gian (từ alerts.json) ===
  const EPS = 0.05;
  outputVideo.addEventListener('timeupdate', ()=>{
    if(!EVENTS || EVENTS.length===0) return;
    const t = outputVideo.currentTime || 0;
    while(EV_IDX < EVENTS.length && EVENTS[EV_IDX].t <= t + EPS){
      const ev = EVENTS[EV_IDX++];
      const msg = (ev && ev.message) ? ev.message : (ev.level==='danger'?'Danger':'Be careful');
      // log ra UI
      appendLog('[ALERT] t='+(ev.t||0).toFixed(2)+' '+(ev.level||'warn')+' -> '+msg);
      // phát âm thanh nếu bật
      if(SOUND_ON){
        try{
          if(ev.level==='danger'){ dangerAudio.currentTime=0; dangerAudio.play().catch(()=>{}); }
          else if(ev.level==='warn'){ warnAudio.currentTime=0; warnAudio.play().catch(()=>{}); }
        }catch(e){}
      }
    }
  });

  // Khi tua: reposition chỉ số EV_IDX tới event đầu >= currentTime - EPS
  function resetEventCursorTo(t){
    if(!EVENTS || EVENTS.length===0){ EV_IDX=0; return; }
    let i=0;
    while(i<EVENTS.length && EVENTS[i].t < t - EPS) i++;
    EV_IDX=i;
  }
  outputVideo.addEventListener('seeked', ()=>{ resetEventCursorTo(outputVideo.currentTime||0); });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting FULLUS UI on http://127.0.0.1:8888")
    app.run(host="127.0.0.1", port=8888, debug=False, threaded=True)
