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
from werkzeug.utils import secure_filename
import yaml

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw_videos")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CONF_DIR = os.path.join(BASE_DIR, "configs")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
for p in (DATA_DIR, OUT_DIR, CONF_DIR, ASSETS_DIR):
    os.makedirs(p, exist_ok = True)

DEFAULT_CONFIG = os.path.join(CONF_DIR, "config.yaml")

app = Flask(__name__)
LOGS = deque(maxlen = 5000)
CLIENTS: list[Queue] = []
JOBS: dict[str, dict] = {}

def log(msg: str):
    line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
    LOGS.append(line)
    for q in list(CLIENTS):
        q.put_nowait(line)


def write_runtime_config(base_cfg_path, fps, width, height, cam_h, sound_on):
    os.makedirs(os.path.join(BASE_DIR, "runtime_configs"), exist_ok = True)
    out_path = os.path.join(BASE_DIR, "runtime_configs", f"run_{int(time.time())}.yaml")
    if yaml is not None:
        try:
            with open(base_cfg_path, "r", encoding = "utf-8") as f:
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
        with open(out_path, "w", encoding = "utf-8") as f:
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

def run_pipeline_async(job_name: str, src_path: str, out_video_path: str, cfg_path: str):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-u", "-m", "src.main", "--src", src_path, "--out", out_video_path, "--config", cfg_path]
    log("cmd: " + " ".join(cmd))
    try:
        rc = 255
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, env=env) as p:
            for line in p.stdout:
                if not line: continue
                line = line.rstrip("\n")
                LOGS.append(line)
                for q in list(CLIENTS):
                    try: q.put_nowait(line)
                    except Exception: pass
            rc = p.wait()

        folder = os.path.dirname(out_video_path)
        base = os.path.splitext(os.path.basename(out_video_path))[0]
        alerts_path = os.path.join(folder, f"{base}_alerts.json")

        if rc == 0 and os.path.exists(out_video_path) and os.path.getsize(out_video_path) > 0:
            JOBS.setdefault(job_name, {}).update({
                "ready": True, "folder": folder,
                "video": f"/video/output/{job_name}/{os.path.basename(out_video_path)}",
                "alerts": (f"/video/output/{job_name}/{os.path.basename(alerts_path)}" if os.path.exists(alerts_path) else None),
                "video_web": None
            })
            log(f"OUTPUT_READY {out_video_path}")
            ok, web_path = transcode_to_web_safe(out_video_path)
            if ok:
                JOBS[job_name]["video_web"] = f"/video/output_web/{job_name}/{os.path.basename(web_path)}"
                log(f"OUTPUT_READY_WEB {web_path}")
        else:
            if rc == 0:
                log(f"[UI] ERROR: pipeline returned rc=0 but output not found or empty: {out_video_path}")
                try:
                    listing = ", ".join(sorted(os.listdir(folder)))
                    log(f"[UI] outputs dir listing: {folder}: {listing}")
                except Exception:
                    pass
            JOBS.setdefault(job_name, {}).update({"ready": False, "folder": folder})
    except FileNotFoundError:
        log("ERROR: Could not launch 'python -m src.main'. Ensure your environment is set up.")
        JOBS.setdefault(job_name, {}).update({"ready": False})
    except Exception as e:
        log(f"ERROR: {e}")
        JOBS.setdefault(job_name, {}).update({"ready": False})

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, default_fps=30, default_w=1280, default_h=720, default_cam_h=1.35)

@app.route("/assets/<path:subpath>")
def assets(subpath):
    full = os.path.join(ASSETS_DIR, subpath)
    if not os.path.isfile(full): abort(404)
    resp = send_from_directory(ASSETS_DIR, subpath, as_attachment=False)
    resp.headers["Accept-Ranges"] = "bytes"
    return resp

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    fps = int(request.form.get("fps", 30))
    width = int(request.form.get("width", 1280))
    height = int(request.form.get("height", 720))
    cam_h = float(request.form.get("cam_h", 1.35))
    sound_on = request.form.get("sound_on", "true").lower() == "true"
    base_cfg = request.form.get("config", DEFAULT_CONFIG)

    if not file or file.filename == "": 
        return jsonify({"ok": False, "error": "No file provided."}), 400
    name = secure_filename(file.filename)
    if not name.lower().endswith(".mp4"): 
        return jsonify({"ok": False, "error": "Only .mp4 files are accepted."}), 400

    in_path = os.path.join(DATA_DIR, name)
    os.makedirs(os.path.dirname(in_path), exist_ok=True)
    file.save(in_path)
    log(f"[UI] received file -> {in_path}")
    log(f"[UI] sound={'on' if sound_on else 'off'} fps={fps} size={width}x{height} cam_h={cam_h}")

    stem = os.path.splitext(name)[0]
    job_folder = os.path.join(OUT_DIR, stem)
    os.makedirs(job_folder, exist_ok=True)
    out_path = os.path.join(job_folder, f"{stem}.mp4")

    JOBS[stem] = {"ready": False, "folder": job_folder,
                  "video": f"/video/output/{stem}/{stem}.mp4",
                  "video_web": None,
                  "alerts": f"/video/output/{stem}/{stem}_alerts.json"}
    cfg_path = write_runtime_config(base_cfg, fps, width, height, cam_h, sound_on)
    threading.Thread(target=run_pipeline_async, args=(stem, in_path, out_path, cfg_path), daemon=True).start()

    return jsonify({"ok": True, "job": stem,
                    "input_url": f"/video/input/{name}",
                    "status_url": f"/status/{stem}"})

@app.route("/status/<job>")
def job_status(job: str):
    info = JOBS.get(job)
    if not info: 
        return jsonify({"exists": False, "ready": False})
    return jsonify({
        "exists": True, "ready": bool(info.get("ready", False)),
        "video_url": info.get("video"),
        "video_web_url": info.get("video_web"),
        "alerts_url": info.get("alerts"),
        "folder": info.get("folder"),
    })

@app.route("/video/input/<path:fname>")
def video_input(fname): 
    return send_from_directory(DATA_DIR, fname, as_attachment=False)

@app.route("/video/output/<job>/<path:fname>")
def video_output(job, fname):
    folder = os.path.join(OUT_DIR, job)
    full = os.path.join(folder, fname)
    if not os.path.exists(full): abort(404)
    return send_from_directory(folder, fname, as_attachment=False)

@app.route("/video/output_web/<job>/<path:fname>")
def video_output_web(job, fname):
    folder = os.path.join(OUT_DIR, job)
    full = os.path.join(folder, fname)
    if not os.path.exists(full): abort(404)
    return send_from_directory(folder, fname, as_attachment=False)

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

  /* --- Upload Placeholder --- */
  .upload-placeholder{
    position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:16px; background: #EDF2F8; color:#6E86A6; text-align:center; user-select:none; cursor:pointer;
  }
  .upload-placeholder .icon{ width: 108px; height:108px; opacity:0.8; }
  .upload-placeholder .hint{ font-weight:800; font-size:22px; }

  /* hide video until set */
  #uploadVideo{ display:none; width:100%; height:100%; object-fit:cover; background:#000; }
  /* shown after file chosen */
  .has-upload #uploadVideo{ display:block; }
  .has-upload .upload-placeholder{ display:none; }

  .video-inner video{ width:100%; height:100%; object-fit: cover; display:block; background:#000; }

  .video-label{ position:absolute; left: 24px; bottom: 20px; font-weight: 700; font-size: 20px; color:#a9b7c6;}

  /* Clear button “trồi ra ngoài” */
  .clear-btn{
    position:absolute; top:-12px; left:-22px; width:56px; height:56px; border-radius:16px;
    background:#fff; border:2px solid var(--border); box-shadow:0 8px 24px rgba(28,49,74,.15);
    display:none; align-items:center; justify-content:center; cursor:pointer;
  }
  .clear-btn img{ width:28px; height:28px; }
  .has-upload + .clear-btn{ display:flex; }

  /* Output area controls */
  .ready-overlay{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; pointer-events:auto; }
  .ready-overlay.hidden{ display:none; }
  .bigplay{ width: 96px; height: 96px; border-radius: 999px; background: #ffffffcc; border: 1px solid var(--border); display:flex; align-items:center; justify-content:center; box-shadow: 0 12px 30px rgba(28,49,74,.18); cursor: pointer; font-weight: 800; }
  .bigplay:active{ transform: translateY(1px); }
  .output-ctl{ position:absolute; right:16px; top:16px; display:flex; gap:8px; pointer-events:auto; }
  .btn{ appearance:none; border:1px solid var(--border); background:#F5F8FD; padding:8px 12px; border-radius:12px; font-weight:700; color:var(--ink); cursor:pointer; box-shadow: 0 2px 8px rgba(28,49,74,.06); }
  .btn:active{ transform: translateY(1px); }

  /* Sidebar */
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
        <!-- Upload card -->
        <div id="uploadArea" class="card video-card">
          <div id="uploadInner" class="video-inner">
            <div id="uploadPlaceholder" class="upload-placeholder" role="button" tabindex="0" aria-label="Upload a .mp4">
              <img class="icon" alt="upload" src="/assets/svg/upload.svg"/>
              <div class="hint">Click to upload .mp4</div>
            </div>
            <video id="uploadVideo" controls muted playsinline></video>
          </div>
          <div id="clearBtn" class="clear-btn" title="Clear & reset">
            <img src="/assets/svg/red_x.svg" alt="clear"/>
          </div>
          <div class="video-label">upload</div>
          <input id="fileInput" type="file" accept="video/mp4" style="display:none"/>
        </div>

        <!-- Output card -->
        <div id="outputArea" class="card video-card" style="margin-top: 24px">
          <div class="video-inner">
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
            <div id="soundSwitch" class="switch on" role="switch" aria-checked="true" tabindex="0"><div class="knob"></div></div>
          </div>
          <div style="color:#506483; margin-top:8px">Toggle warning sound</div>
        </div>

        <div class="log card">
          <div style="font-weight:800; font-size:22px; margin-bottom:6px">Log</div>
          <div id="logbox"></div>
        </div>
      </aside>
    </main>
  </div>

<script>
  // --- State ---
  let SOUND_ON = true;
  let activeJob = null;
  let CANCEL_TOKEN = {cancel:false};
  let uploadObjectURL = null;
  let OUTPUT_URL=null, WEB_URL=null, ALERTS_URL=null, CURRENT_SRC=null, ATTACHED=false;

  // --- Elements ---
  const uploadArea=document.getElementById('uploadArea');
  const uploadInner=document.getElementById('uploadInner');
  const uploadPlaceholder=document.getElementById('uploadPlaceholder');
  const uploadVideo=document.getElementById('uploadVideo');
  const clearBtn=document.getElementById('clearBtn');

  const fileInput=document.getElementById('fileInput');
  const outputVideo=document.getElementById('outputVideo');
  const logbox=document.getElementById('logbox');
  const playBtn=document.getElementById('playBtn');
  const pauseBtn=document.getElementById('pauseBtn');
  const readyOverlay=document.getElementById('readyOverlay');
  const bigPlay=document.getElementById('bigPlay');
  const soundSwitch=document.getElementById('soundSwitch');

  // --- Audio ---
  let audioYellow = new Audio('/assets/audio/be_careful.mp3');
  let audioRed = new Audio('/assets/audio/danger.mp3');
  audioYellow.preload = 'auto'; audioRed.preload = 'auto';

  function logLine(s){
    const pre=document.createElement('div'); pre.textContent=s; logbox.appendChild(pre);
    logbox.scrollTop=logbox.scrollHeight;
  }

  // --- UI helpers ---
  function setUploadHasVideo(has){
    uploadInner.classList.toggle('has-upload', !!has);
  }

  function resetUpload(){
    CANCEL_TOKEN.cancel = true;  // stop poller
    activeJob = null;
    OUTPUT_URL = WEB_URL = ALERTS_URL = CURRENT_SRC = null; ATTACHED = false;

    // stop & clear videos
    try{ uploadVideo.pause(); }catch(e){}
    try{ outputVideo.pause(); }catch(e){}
    try{ outputVideo.removeAttribute('src'); outputVideo.load(); }catch(e){}
    if(uploadObjectURL){ URL.revokeObjectURL(uploadObjectURL); uploadObjectURL=null; }
    uploadVideo.removeAttribute('src'); uploadVideo.load();

    setUploadHasVideo(false);
    readyOverlay.classList.remove('hidden');
    logLine('[UI] reset upload');
  }

  function attachOutput(url){
    if(!url) return;
    ATTACHED=true; CURRENT_SRC=url;
    outputVideo.pause();
    outputVideo.removeAttribute('src');
    outputVideo.load();
    outputVideo.src=url + (url.includes('?')?'&':'?') + 'v=' + Date.now();
    outputVideo.load();

    readyOverlay.classList.remove('hidden');

    const onCanPlay=()=>{ outputVideo.play().catch(()=>{}); };
    const onPlay=()=>{ readyOverlay.classList.add('hidden'); };

    outputVideo.addEventListener('canplay', onCanPlay, {once:true});
    outputVideo.addEventListener('play', onPlay, {once:true});
  }

  // --- Upload interactions ---
  uploadPlaceholder.addEventListener('click', ()=>fileInput.click());
  uploadPlaceholder.addEventListener('keypress', (e)=>{ if(e.key==='Enter') fileInput.click(); });
  clearBtn.addEventListener('click', resetUpload);

  fileInput.addEventListener('change', async (ev)=>{
    const f=ev.target.files[0]; if(!f) return;
    if(!f.name.toLowerCase().endsWith('.mp4')){ alert('Only .mp4 files are accepted'); return; }
    if(uploadObjectURL){ URL.revokeObjectURL(uploadObjectURL); }
    uploadObjectURL = URL.createObjectURL(f);
    uploadVideo.src = uploadObjectURL;
    setUploadHasVideo(true);
    try{ await uploadVideo.play(); }catch(_){}

    // prepare form
    const fd=new FormData();
    fd.append('file', f);
    fd.append('fps', document.getElementById('fps').value||30);
    fd.append('width', document.getElementById('width').value||1280);
    fd.append('height', document.getElementById('height').value||720);
    fd.append('cam_h', document.getElementById('cam_h').value||1.35);
    fd.append('sound_on', SOUND_ON?'true':'false');

    CANCEL_TOKEN = {cancel:false};
    const r=await fetch('/upload',{method:'POST',body:fd});
    const j=await r.json();
    if(!j.ok){ alert('Upload failed: '+(j.error||'unknown')); resetUpload(); return; }
    activeJob=j.job; OUTPUT_URL=null; WEB_URL=null; ALERTS_URL=null; ATTACHED=false;
    pollJobUntilReady(j.status_url, CANCEL_TOKEN);
  });

  // --- Poll status & attach output + alerts ---
  async function pollJobUntilReady(statusUrl, token){
    let waited=0;
    while(!token.cancel){
      const r=await fetch(statusUrl);
      if(!r.ok) break;
      const s=await r.json();
      if(s.exists && s.ready){
        if(s.video_web_url){ WEB_URL=s.video_web_url; }
        OUTPUT_URL = s.video_url || OUTPUT_URL;
        ALERTS_URL = s.alerts_url || ALERTS_URL;

        // attach video
        if(WEB_URL){ attachOutput(WEB_URL); } else if(OUTPUT_URL){ attachOutput(OUTPUT_URL); }

        // fetch alerts (once)
        if(ALERTS_URL){ fetchAlerts(ALERTS_URL); }
        return;
      }
      await new Promise(res=>setTimeout(res, 1200));
      waited+=1;
    }
  }

  // Alerts schedule & playback
  let schedule = [];
  let nextIdx = 0;
  function clearSchedule(){ schedule=[]; nextIdx=0; }
  function fetchAlerts(url){
    logLine('[UI] fetching alerts: '+url);
    fetch(url).then(r=>{
      if(!r.ok) throw new Error('HTTP '+r.status+' for '+url);
      return r.json();
    }).then(js=>{
      schedule = (Array.isArray(js)?js:[]);
      schedule.sort((a,b)=>(a.t||0)-(b.t||0));
      nextIdx = 0;
      logLine('[UI] loaded '+schedule.length+' alerts from '+url);
    }).catch(e=>{
      logLine('[UI] alerts load failed: '+e);
    });
  }

  // Audio toggle
  soundSwitch.addEventListener('click', ()=>{
    SOUND_ON = !SOUND_ON;
    soundSwitch.classList.toggle('on', SOUND_ON);
    soundSwitch.setAttribute('aria-checked', SOUND_ON?'true':'false');
  });

  // Hook playback timeupdate to fire alerts
  let lastPlayhead = 0;
  outputVideo.addEventListener('timeupdate', ()=>{
    const t = outputVideo.currentTime || 0;
    // replay/seek backward -> reset index
    if(t + 0.05 < lastPlayhead){ nextIdx = 0; }
    lastPlayhead = t;

    while(nextIdx < schedule.length && t >= (schedule[nextIdx].t || 0)){
      const ev = schedule[nextIdx++];
      const msg = (ev.message || (ev.level==='red'?'Danger':'Be careful'));
      logLine(`[ALERT] ${t.toFixed(2)}s ${msg}`);
      if(SOUND_ON){
        try{
          if((ev.level||'').toLowerCase()==='red'){ audioRed.currentTime=0; audioRed.play().catch(()=>{}); }
          else { audioYellow.currentTime=0; audioYellow.play().catch(()=>{}); }
        }catch(_){}
      }
    }
  });

  // Output controls
  document.getElementById('playBtn').addEventListener('click', ()=>{ outputVideo.play().catch(()=>{}); });
  document.getElementById('pauseBtn').addEventListener('click', ()=>{ try{ outputVideo.pause(); }catch(e){} });
  document.getElementById('bigPlay').addEventListener('click', ()=>{ outputVideo.play().catch(()=>{}); });
  outputVideo.addEventListener('error', ()=>{
    logLine('[UI] video element error');
    if(CURRENT_SRC && WEB_URL && CURRENT_SRC!==WEB_URL){ attachOutput(WEB_URL); return; }
    if(CURRENT_SRC && OUTPUT_URL && CURRENT_SRC!==OUTPUT_URL){ attachOutput(OUTPUT_URL); return; }
  });

  // SSE logs (read-only)
  try{ const es=new EventSource('/logs'); es.onmessage=(e)=>{ const line=e.data; const node=document.createElement('div'); node.textContent=line; logbox.appendChild(node); logbox.scrollTop=logbox.scrollHeight; }; }catch(e){ console.warn('SSE failed',e); }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting FULLUS UI on http://127.0.0.1:8888")
    app.run(host="127.0.0.1", port = 8888, debug = False, threaded = True)
