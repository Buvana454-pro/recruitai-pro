import json, re, time, uuid, os, sys, threading, webbrowser
from datetime import datetime, timezone
from collections import deque
from typing import Optional
import base64

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("\n❌  Run this first:\n")
    print("    pip install fastapi uvicorn groq\n")
    sys.exit(1)

try:
    from groq import Groq
except ImportError:
    print("\n❌  Run this first:\n")
    print("    pip install groq\n")
    sys.exit(1)

# ── Get API Key ───────────────────────────────────────────────
API_KEY = os.environ.get("GROQ_API_KEY", "")

if not API_KEY:
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   Get your FREE Groq API Key (takes 1 min)  ║")
    print("  ║                                              ║")
    print("  ║   1. Go to  https://console.groq.com        ║")
    print("  ║   2. Sign up (free, no credit card)         ║")
    print("  ║   3. Click API Keys → Create API Key        ║")
    print("  ║   4. Copy the key and paste it below        ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    API_KEY = input("GROQ_API_KEY: ").strip()
    if not API_KEY:
        print("\n❌  No key entered. Exiting.\n")
        sys.exit(1)
    print()

# ── Init Groq client ─────────────────────────────────────────
client = Groq(api_key=API_KEY)

# ══════════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════════
class AppLogger:
    def __init__(self):
        self._logs = deque(maxlen=500)

    def _emit(self, level, event, payload):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            "service": "recruitai-pro",
            "log_id": uuid.uuid4().hex[:8],
            **payload,
        }
        self._logs.appendleft(entry)
        print(json.dumps(entry))
        return entry

    def info(self, e, p={}):  return self._emit("INFO",  e, p)
    def warn(self, e, p={}):  return self._emit("WARN",  e, p)
    def error(self, e, p={}): return self._emit("ERROR", e, p)

    def get_logs(self, limit=100, level=None):
        logs = list(self._logs)
        if level and level.upper() != "ALL":
            logs = [l for l in logs if l["level"] == level.upper()]
        return logs[:limit]

    def clear(self): self._logs.clear()

# ══════════════════════════════════════════════════════════════
#  AI SCREENING  (Groq — llama3-70b, ultra fast & free)
# ══════════════════════════════════════════════════════════════
SYSTEM_MSG = """You are an expert technical recruiter AI. Evaluate resumes against job descriptions with precision.
Respond ONLY with valid JSON — no markdown fences, no explanation, just the raw JSON object."""

PROMPT_TEMPLATE = """Evaluate this resume against the job description below.

JOB TITLE: {title}
JOB DESCRIPTION: {description}
REQUIRED SKILLS: {required_skills}
NICE TO HAVE: {nice_to_have}

RESUME:
{resume_text}

Return ONLY this JSON (no markdown, no backticks):
{{
  "score": <integer 0-100>,
  "match_pct": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "verdict": "<STRONG_MATCH|GOOD_MATCH|PARTIAL_MATCH|WEAK_MATCH|NO_MATCH>",
  "summary": "<2-3 sentence summary of candidate strengths and fit>",
  "skills_matched": ["skill1", "skill2"],
  "skills_missing": ["skill1", "skill2"],
  "skills_bonus": ["nice-to-have skills the candidate has"],
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "concerns": ["concern 1", "concern 2"],
  "experience_years": <integer>,
  "recommendation": "<one clear action sentence>",
  "interview_questions": ["question 1", "question 2", "question 3"]
}}"""


def run_screening(resume_text, title, desc, req_skills, nice_skills):
    prompt = PROMPT_TEMPLATE.format(
        title=title,
        description=desc,
        required_skills=", ".join(req_skills),
        nice_to_have=", ".join(nice_skills) if nice_skills else "None specified",
        resume_text=resume_text[:6000],
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # free, fast, excellent quality
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    raw = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else 0

    # Strip any accidental markdown fences
    clean = re.sub(r"```json|```", "", raw).strip()
    result = json.loads(clean)
    result["tokens_used"] = tokens_used
    return result

# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════
app = FastAPI(title="RecruitAI Pro")
logger = AppLogger()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class JobDescription(BaseModel):
    title: str
    description: str
    required_skills: list[str]
    nice_to_have: Optional[list[str]] = []

class ScreenRequest(BaseModel):
    job_description: JobDescription
    resume_text: str
    candidate_name: Optional[str] = "Anonymous"


@app.post("/api/screen")
async def screen(req: ScreenRequest):
    trace_id  = uuid.uuid4().hex[:8]
    screen_id = f"scr_{uuid.uuid4().hex[:6]}"

    logger.info("screening.started", {
        "screen_id": screen_id,
        "trace_id": trace_id,
        "jd_title": req.job_description.title,
        "resume_length": len(req.resume_text),
    })

    start = time.time()
    try:
        result = run_screening(
            req.resume_text,
            req.job_description.title,
            req.job_description.description,
            req.job_description.required_skills,
            req.job_description.nice_to_have,
        )
        latency_ms = int((time.time() - start) * 1000)

        logger.info("screening.completed", {
            "screen_id": screen_id,
            "trace_id": trace_id,
            "score": result["score"],
            "latency_ms": latency_ms,
            "tokens_used": result.get("tokens_used", 0),
            "skills_matched": result.get("skills_matched", []),
            "skills_missing": result.get("skills_missing", []),
        })

        if result.get("confidence", 1.0) < 0.5:
            logger.warn("screening.low_confidence", {
                "screen_id": screen_id,
                "confidence": result["confidence"],
            })

        return {"screen_id": screen_id, "trace_id": trace_id, "latency_ms": latency_ms, **result}

    except Exception as e:
        logger.error("screening.ai_error", {"screen_id": screen_id, "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
def get_logs(limit: int = 100, level: Optional[str] = None):
    return {"logs": logger.get_logs(limit, level)}

@app.delete("/api/logs")
def clear_logs():
    logger.clear()
    return {"message": "cleared"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ── App Settings ──────────────────────────────────────────────
_app_settings = {
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.2,
    "max_tokens": 1500,
    "score_threshold_strong": 75,
    "score_threshold_good": 50,
    "auto_flag_low_confidence": True,
    "notifications": True,
    "theme": "dark",
}

class SettingsReq(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    score_threshold_strong: Optional[int] = None
    score_threshold_good: Optional[int] = None
    auto_flag_low_confidence: Optional[bool] = None
    notifications: Optional[bool] = None
    theme: Optional[str] = None

@app.get("/api/settings")
def get_settings():
    return _app_settings

@app.post("/api/settings")
def save_settings(req: SettingsReq):
    for k, v in req.dict(exclude_none=True).items():
        if k in _app_settings:
            _app_settings[k] = v
    logger.info("settings.updated", {"changes": req.dict(exclude_none=True)})
    return {"ok": True, "settings": _app_settings}

# ── Notes ─────────────────────────────────────────────────────
_notes: list = []

class NoteReq(BaseModel):
    title: str
    content: str
    tags: Optional[list[str]] = []

@app.get("/api/notes")
def get_notes():
    return {"notes": _notes}

@app.post("/api/notes")
def add_note(req: NoteReq):
    note = {"id": uuid.uuid4().hex[:8], "title": req.title, "content": req.content, "tags": req.tags or [], "created_at": datetime.now(timezone.utc).isoformat()}
    _notes.insert(0, note)
    return note

@app.delete("/api/notes/{note_id}")
def delete_note(note_id: str):
    global _notes
    _notes = [n for n in _notes if n["id"] != note_id]
    return {"ok": True}

# ── Job Templates ─────────────────────────────────────────────
_templates: list = []

class TemplateReq(BaseModel):
    name: str
    title: str
    description: str
    required_skills: list[str]
    nice_to_have: Optional[list[str]] = []

@app.get("/api/templates")
def get_templates():
    return {"templates": _templates}

@app.post("/api/templates")
def save_template(req: TemplateReq):
    t = {"id": uuid.uuid4().hex[:8], "name": req.name, "title": req.title, "description": req.description, "required_skills": req.required_skills, "nice_to_have": req.nice_to_have or [], "created_at": datetime.now(timezone.utc).isoformat()}
    _templates.insert(0, t)
    return t

@app.delete("/api/templates/{tid}")
def delete_template(tid: str):
    global _templates
    _templates = [t for t in _templates if t["id"] != tid]
    return {"ok": True}

# ── Export results as JSON ────────────────────────────────────
from fastapi.responses import Response as FastResponse

_screen_history: list = []

@app.post("/api/history")
async def save_to_history(req: dict):
    _screen_history.insert(0, {**req, "saved_at": datetime.now(timezone.utc).isoformat()})
    if len(_screen_history) > 200:
        _screen_history.pop()
    return {"ok": True}

@app.get("/api/export")
def export_results():
    data = json.dumps({"exported_at": datetime.now(timezone.utc).isoformat(), "results": _screen_history}, indent=2)
    return FastResponse(content=data, media_type="application/json", headers={"Content-Disposition": "attachment; filename=recruitai_results.json"})

# ══════════════════════════════════════════════════════════════
#  FULL FRONTEND HTML
# ══════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RecruitAI Pro</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><rect width='64' height='64' rx='14' fill='url(%23g)'/><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='%238b5cf6'/><stop offset='1' stop-color='%2300e5ff'/></linearGradient></defs><text x='50%25' y='54%25' dominant-baseline='middle' text-anchor='middle' font-size='34' font-family='serif' fill='white'>✦</text></svg>">
<link href="https://fonts.googleapis.com/css2?family=Clash+Display:wght@400;500;600;700&family=Cabinet+Grotesk:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#030308;--bg1:#08080f;--bg2:#0d0d1a;--bg3:#12121f;--bg4:#16162a;
  --glass:rgba(255,255,255,0.03);--glass2:rgba(255,255,255,0.06);--glass3:rgba(255,255,255,0.09);
  --border:rgba(255,255,255,0.06);--border2:rgba(255,255,255,0.12);--border3:rgba(255,255,255,0.18);
  --text:#f8f8ff;--text2:#a0a0c0;--text3:#505070;
  --cyan:#00e5ff;--violet:#8b5cf6;--rose:#ff4d8d;--emerald:#00f5a0;--amber:#ffb340;
  --r:16px;
}
html{scroll-behavior:smooth}
body{font-family:'Cabinet Grotesk',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}

/* AURORA */
.aurora{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none}
.orb{position:absolute;border-radius:50%;filter:blur(90px);animation:drift 22s ease-in-out infinite}
.o1{width:700px;height:700px;background:radial-gradient(circle,rgba(139,92,246,0.18),transparent);top:-200px;left:-200px;animation-delay:0s}
.o2{width:600px;height:600px;background:radial-gradient(circle,rgba(0,229,255,0.15),transparent);top:35%;right:-180px;animation-delay:-8s}
.o3{width:500px;height:500px;background:radial-gradient(circle,rgba(255,77,141,0.12),transparent);bottom:-100px;left:35%;animation-delay:-15s}
.o4{width:350px;height:350px;background:radial-gradient(circle,rgba(0,245,160,0.1),transparent);top:55%;left:8%;animation-delay:-11s}
@keyframes drift{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(50px,-35px) scale(1.08)}66%{transform:translate(-35px,50px) scale(0.94)}}

/* GRID + NOISE */
.grid-bg{position:fixed;inset:0;z-index:0;pointer-events:none;background-image:linear-gradient(rgba(255,255,255,0.012) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.012) 1px,transparent 1px);background-size:56px 56px}
.noise{position:fixed;inset:0;z-index:1;pointer-events:none;opacity:0.022;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");background-size:256px}

/* APP */
.app{position:relative;z-index:2;display:flex;flex-direction:column;min-height:100vh}

/* HEADER */
.header{display:flex;align-items:center;padding:0 28px;height:62px;background:rgba(3,3,8,0.75);backdrop-filter:blur(24px);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100}
.logo-wrap{display:flex;align-items:center;gap:12px}
.logo-gem{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,var(--violet),var(--cyan));display:flex;align-items:center;justify-content:center;font-size:17px;box-shadow:0 0 18px rgba(139,92,246,0.45)}
.logo-name{font-family:'Clash Display',sans-serif;font-size:19px;font-weight:700;letter-spacing:-0.5px}
.logo-name b{background:linear-gradient(135deg,var(--cyan),var(--violet));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.pill{font-size:10px;padding:3px 9px;border-radius:20px;font-family:'Clash Display',sans-serif;letter-spacing:0.04em}
.pill-green{background:rgba(0,245,160,0.1);color:var(--emerald);border:1px solid rgba(0,245,160,0.25)}
.pill-groq{background:rgba(255,179,64,0.1);color:var(--amber);border:1px solid rgba(255,179,64,0.25)}
.header-nav{display:flex;gap:2px;margin-left:auto;background:var(--glass);border:1px solid var(--border);border-radius:11px;padding:4px}
.nav-btn{padding:7px 16px;border-radius:8px;border:none;background:none;color:var(--text2);font-size:13px;font-family:'Cabinet Grotesk',sans-serif;font-weight:500;cursor:pointer;transition:all 0.2s;white-space:nowrap}
.nav-btn:hover{color:var(--text);background:var(--glass2)}
.nav-btn.active{background:linear-gradient(135deg,rgba(139,92,246,0.22),rgba(0,229,255,0.1));color:var(--text);border:1px solid rgba(139,92,246,0.3)}

/* BODY */
.body{display:flex;flex:1}

/* SIDEBAR */
.sidebar{width:215px;flex-shrink:0;padding:22px 10px;border-right:1px solid var(--border);background:rgba(3,3,8,0.55);backdrop-filter:blur(14px);display:flex;flex-direction:column;gap:2px}
.s-sec{font-size:10px;text-transform:uppercase;letter-spacing:0.12em;color:var(--text3);padding:10px 12px 5px;font-family:'Clash Display',sans-serif}
.s-item{display:flex;align-items:center;gap:9px;padding:8px 12px;border-radius:10px;cursor:pointer;font-size:13px;color:var(--text2);border:1px solid transparent;background:none;width:100%;text-align:left;transition:all 0.18s;font-family:'Cabinet Grotesk',sans-serif;font-weight:500;position:relative}
.s-item:hover{background:var(--glass2);color:var(--text)}
.s-item.active{background:linear-gradient(135deg,rgba(139,92,246,0.14),rgba(0,229,255,0.07));color:var(--cyan);border-color:rgba(0,229,255,0.15)}
.s-item.active::before{content:'';position:absolute;left:0;top:18%;height:64%;width:2.5px;background:linear-gradient(180deg,var(--violet),var(--cyan));border-radius:0 3px 3px 0}
.s-icon{font-size:15px;width:20px;text-align:center;flex-shrink:0}

/* MAIN */
.main{flex:1;overflow-y:auto;padding:30px 34px;min-width:0}

/* PANELS */
.panel{display:none;animation:fadeUp 0.38s ease}
.panel.active{display:block}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

/* HERO */
.hero{text-align:center;padding:16px 0 36px;position:relative}
.hero-chip{display:inline-flex;align-items:center;gap:7px;font-size:11px;padding:5px 14px;border-radius:20px;background:rgba(0,229,255,0.07);color:var(--cyan);border:1px solid rgba(0,229,255,0.2);margin-bottom:18px;font-family:'Clash Display',sans-serif;letter-spacing:0.04em}
.live-dot{width:6px;height:6px;background:var(--cyan);border-radius:50%;animation:ldot 2s infinite}
@keyframes ldot{0%,100%{opacity:1;box-shadow:0 0 7px var(--cyan)}50%{opacity:0.4;box-shadow:none}}
.hero h1{font-family:'Clash Display',sans-serif;font-size:44px;font-weight:700;letter-spacing:-1.5px;line-height:1.08;margin-bottom:14px}
.grad-text{background:linear-gradient(135deg,var(--cyan) 0%,var(--violet) 45%,var(--rose) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero p{font-size:15px;color:var(--text2);max-width:480px;margin:0 auto 26px;line-height:1.65}
.hero-meta{display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;margin-top:4px}
.hero-meta-item{font-size:12px;color:var(--text3);display:flex;align-items:center;gap:5px}
.hero-meta-item span{color:var(--emerald)}

/* CARDS */
.card{background:rgba(13,13,26,0.65);backdrop-filter:blur(18px);border:1px solid var(--border);border-radius:var(--r);padding:22px;transition:border-color 0.3s}
.card:hover{border-color:var(--border2)}
.card-accent{background:rgba(13,13,26,0.65);backdrop-filter:blur(18px);border:1px solid rgba(139,92,246,0.2);border-radius:var(--r);padding:22px;box-shadow:0 0 36px rgba(139,92,246,0.06),inset 0 1px 0 rgba(255,255,255,0.04)}
.card-sm{background:rgba(18,18,31,0.7);border:1px solid var(--border);border-radius:12px;padding:15px 18px}

/* FORM */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
.fgroup{display:flex;flex-direction:column;gap:6px}
.flabel{font-size:11.5px;color:var(--text2);font-weight:500;letter-spacing:0.02em}
.flabel small{color:var(--text3);font-weight:400}
.finput,.ftarea{width:100%;background:rgba(18,18,31,0.9);border:1px solid var(--border2);border-radius:10px;padding:10px 14px;color:var(--text);font-size:13.5px;font-family:'Cabinet Grotesk',sans-serif;outline:none;transition:all 0.2s;line-height:1.6}
.ftarea{resize:vertical}
.finput:focus,.ftarea:focus{border-color:rgba(139,92,246,0.5);background:rgba(18,18,31,1);box-shadow:0 0 0 3px rgba(139,92,246,0.07)}
.finput::placeholder,.ftarea::placeholder{color:var(--text3)}

/* SECTION LABEL */
.sec-lbl{font-size:10.5px;text-transform:uppercase;letter-spacing:0.1em;color:var(--text3);font-family:'Clash Display',sans-serif;display:flex;align-items:center;gap:8px;margin-bottom:14px}
.sec-lbl::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}

/* BUTTONS */
.btn-main{display:inline-flex;align-items:center;gap:8px;padding:12px 26px;border-radius:11px;border:none;cursor:pointer;font-size:14px;font-family:'Clash Display',sans-serif;font-weight:600;background:linear-gradient(135deg,var(--violet),var(--cyan));color:#fff;transition:all 0.22s;box-shadow:0 4px 22px rgba(139,92,246,0.32);position:relative;overflow:hidden}
.btn-main::after{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(255,255,255,0.1),transparent);opacity:0;transition:opacity 0.2s}
.btn-main:hover::after{opacity:1}
.btn-main:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(139,92,246,0.42)}
.btn-main:disabled{opacity:0.5;cursor:not-allowed;transform:none}
.btn-sec{display:inline-flex;align-items:center;gap:7px;padding:10px 20px;border-radius:10px;border:1px solid var(--border2);background:var(--glass);color:var(--text2);font-size:13px;font-family:'Cabinet Grotesk',sans-serif;font-weight:500;cursor:pointer;transition:all 0.18s;backdrop-filter:blur(6px)}
.btn-sec:hover{border-color:var(--border3);color:var(--text);background:var(--glass2)}
.btn-ico{width:34px;height:34px;border-radius:8px;border:1px solid var(--border2);background:var(--glass);color:var(--text2);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:15px;transition:all 0.18s}
.btn-ico:hover{border-color:var(--cyan);color:var(--cyan)}

/* BADGES */
.badge{display:inline-flex;align-items:center;gap:3px;font-size:11px;padding:3px 9px;border-radius:20px;font-weight:600;font-family:'Clash Display',sans-serif;letter-spacing:0.02em}
.bc{background:rgba(0,229,255,0.09);color:var(--cyan);border:1px solid rgba(0,229,255,0.2)}
.bv{background:rgba(139,92,246,0.1);color:#a78bfa;border:1px solid rgba(139,92,246,0.2)}
.br{background:rgba(255,77,141,0.1);color:var(--rose);border:1px solid rgba(255,77,141,0.2)}
.be{background:rgba(0,245,160,0.1);color:var(--emerald);border:1px solid rgba(0,245,160,0.2)}
.ba{background:rgba(255,179,64,0.1);color:var(--amber);border:1px solid rgba(255,179,64,0.2)}
.bg{background:rgba(255,255,255,0.04);color:var(--text3);border:1px solid var(--border)}

/* TAGS */
.tw{display:flex;flex-wrap:wrap;gap:5px}
.tag{font-size:11px;padding:3px 9px;border-radius:6px;font-weight:500}
.tag-m{background:rgba(0,245,160,0.09);color:var(--emerald);border:1px solid rgba(0,245,160,0.2)}
.tag-x{background:rgba(255,77,141,0.09);color:var(--rose);border:1px solid rgba(255,77,141,0.2)}
.tag-b{background:rgba(255,179,64,0.09);color:var(--amber);border:1px solid rgba(255,179,64,0.2)}

/* SCORE RING */
.sr-wrap{position:relative;width:126px;height:126px;flex-shrink:0}
.sr-inner{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.sr-num{font-family:'Clash Display',sans-serif;font-size:32px;font-weight:700;line-height:1}
.sr-den{font-size:10px;color:var(--text3);margin-top:2px}

/* METRIC CARDS */
.mgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin-bottom:22px}
.metric{background:rgba(13,13,26,0.85);border:1px solid var(--border);border-radius:13px;padding:17px 20px;position:relative;overflow:hidden}
.metric::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent)}
.m-lbl{font-size:10.5px;color:var(--text3);text-transform:uppercase;letter-spacing:0.09em;font-family:'Clash Display',sans-serif;margin-bottom:7px}
.m-val{font-size:29px;font-weight:700;font-family:'Clash Display',sans-serif;line-height:1}
.m-sub{font-size:11px;color:var(--text3);margin-top:4px}

/* BAR */
.bar-row{margin-bottom:9px}
.bar-top{display:flex;justify-content:space-between;margin-bottom:4px;font-size:12px}
.bar-lbl{color:var(--text2)}
.bar-v{font-family:'Clash Display',sans-serif;font-weight:600}
.bar-tr{background:rgba(255,255,255,0.04);border-radius:4px;height:5px;overflow:hidden}
.bar-fi{height:100%;border-radius:4px;transition:width 0.9s cubic-bezier(.34,1.56,.64,1)}

/* RESULT CARD */
.rc{background:rgba(13,13,26,0.75);backdrop-filter:blur(18px);border:1px solid var(--border);border-radius:17px;margin-bottom:14px;overflow:hidden;transition:border-color 0.25s}
.rc:hover{border-color:var(--border2)}
.rc-hdr{display:flex;align-items:center;justify-content:space-between;padding:19px 22px;cursor:pointer}
.rc-body{padding:0 22px 22px;border-top:1px solid var(--border)}

/* LOGS */
.log-row{display:grid;grid-template-columns:68px 48px 185px 1fr;gap:9px;padding:9px 5px;border-bottom:1px solid var(--border);font-size:11.5px;cursor:pointer;border-radius:6px;align-items:start;transition:background 0.12s}
.log-row:hover{background:var(--glass2)}
.log-time{font-family:monospace;color:var(--text3);font-size:10px}
.log-detail{display:none;padding:10px 14px;margin:2px 0 5px;background:rgba(3,3,8,0.85);border-radius:8px;font-family:monospace;font-size:11px;color:var(--text2);white-space:pre-wrap;word-break:break-all;border:1px solid var(--border)}

/* SPINNER / LOADING */
.spin{width:18px;height:18px;border:2px solid rgba(255,255,255,0.1);border-top-color:var(--cyan);border-radius:50%;animation:sp 0.7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.lov{position:fixed;inset:0;z-index:999;background:rgba(3,3,8,0.88);backdrop-filter:blur(14px);display:none;flex-direction:column;align-items:center;justify-content:center;gap:18px}
.lov.show{display:flex}
.big-ring{width:68px;height:68px;border-radius:50%;border:2.5px solid rgba(255,255,255,0.06);border-top-color:var(--cyan);border-right-color:var(--violet);animation:sp 0.95s linear infinite}
.lov-title{font-family:'Clash Display',sans-serif;font-size:17px;color:var(--text2)}
.lov-sub{font-size:12px;color:var(--text3)}
.lov-dots::after{content:'';animation:ld 1.4s infinite}
@keyframes ld{0%{content:''}25%{content:'.'}50%{content:'..'}75%,100%{content:'...'}}

/* ERR */
.err-box{background:rgba(255,77,141,0.07);border:1px solid rgba(255,77,141,0.2);border-radius:10px;padding:11px 15px;color:var(--rose);font-size:13px;display:none;margin-top:10px}

/* GRID */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:13px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:13px}
.row{display:flex;align-items:center}
.col{display:flex;flex-direction:column}
.between{display:flex;align-items:center;justify-content:space-between}
.gap4{gap:4px}.gap8{gap:8px}.gap10{gap:10px}.gap12{gap:12px}.gap14{gap:14px}.gap16{gap:16px}.gap18{gap:18px}.gap20{gap:20px}.gap24{gap:24px}
.mb4{margin-bottom:4px}.mb8{margin-bottom:8px}.mb12{margin-bottom:12px}.mb14{margin-bottom:14px}.mb16{margin-bottom:16px}.mb20{margin-bottom:20px}.mb24{margin-bottom:24px}.mb28{margin-bottom:28px}.mb32{margin-bottom:32px}
.mt8{margin-top:8px}.mt10{margin-top:10px}.mt12{margin-top:12px}.mt14{margin-top:14px}.mt16{margin-top:16px}.mt20{margin-top:20px}
.empty{text-align:center;padding:72px 20px;color:var(--text3)}
.empty-i{font-size:50px;margin-bottom:14px;opacity:0.28}
.empty-t{font-family:'Clash Display',sans-serif;font-size:21px;color:var(--text2);margin-bottom:8px}
@media(max-width:900px){.body{flex-direction:column}.sidebar{display:none}.form-grid,.g2,.g3{grid-template-columns:1fr}.mgrid{grid-template-columns:1fr 1fr}.main{padding:18px 16px}}
</style>
</head>
<body>

<!-- BACKGROUNDS -->
<div class="aurora"><div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div><div class="orb o4"></div></div>
<div class="grid-bg"></div>
<div class="noise"></div>

<!-- LOADING OVERLAY -->
<div class="lov" id="lov">
  <div class="big-ring"></div>
  <div class="lov-title">Analyzing with Groq AI<span class="lov-dots"></span></div>
  <div class="lov-sub">Llama 3 · Usually under 3 seconds ⚡</div>
</div>

<div class="app">

<!-- HEADER -->
<header class="header">
  <div class="logo-wrap">
    <div class="logo-gem">✦</div>
    <div class="logo-name">Recruit<b>AI</b> Pro</div>
    <span class="pill pill-groq">⚡ Groq</span>
  </div>
  <nav class="header-nav">
    <button class="nav-btn active" onclick="go('screen',this)">✦ Screen</button>
    <button class="nav-btn" onclick="go('results',this)">Results</button>
    <button class="nav-btn" onclick="go('dashboard',this)">Dashboard</button>
    <button class="nav-btn" onclick="go('logs',this)">Logs</button>
  </nav>
  <div style="margin-left:12px;display:flex;align-items:center;gap:8px">
    <div id="hdr-user" style="display:none;align-items:center;gap:8px">
      <div id="hdr-avatar" onclick="toggleProfile()" style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--violet),var(--cyan));display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;cursor:pointer;font-family:'Clash Display',sans-serif;box-shadow:0 0 12px rgba(139,92,246,0.4)"></div>
      <span id="hdr-name" style="font-size:13px;color:var(--text2);font-weight:500;cursor:pointer" onclick="toggleProfile()"></span>
    </div>
    <button id="hdr-login-btn" class="btn-sec" onclick="showAuthModal('login')" style="font-size:12px;padding:6px 14px">Login</button>
  </div>
</header>

<div class="body">

<!-- SIDEBAR -->
<aside class="sidebar">
  <div class="s-sec">Menu</div>
  <button class="s-item active" onclick="go('screen',null,this)"><span class="s-icon">✦</span>Screen Resume</button>
  <button class="s-item" onclick="go('results',null,this)"><span class="s-icon">📄</span>Results</button>
  <button class="s-item" onclick="go('compare',null,this)"><span class="s-icon">⚖️</span>Compare</button>
  <button class="s-item" onclick="go('dashboard',null,this)"><span class="s-icon">📊</span>Dashboard</button>
  <button class="s-item" onclick="go('notes',null,this)"><span class="s-icon">📝</span>Notes</button>
  <button class="s-item" onclick="go('templates',null,this)"><span class="s-icon">🗂️</span>Templates</button>
  <button class="s-item" onclick="go('logs',null,this)"><span class="s-icon">📋</span>Logs</button>
  <button class="s-item" onclick="go('settings',null,this)"><span class="s-icon">⚙️</span>Settings</button>
  <div class="s-sec mt20">Recent</div>
  <div id="sb-hist"></div>
  <div style="margin-top:auto;padding:12px;border-top:1px solid var(--border)">
    <div style="font-size:11px;color:var(--text3);line-height:1.6">
      <div style="color:var(--emerald);margin-bottom:4px;font-family:'Clash Display',sans-serif;font-size:10px;letter-spacing:0.08em;text-transform:uppercase">Groq Free Tier</div>
      ✓ 30 req/min<br>✓ 14,400 req/day<br>✓ Llama 3.3 70B<br>✓ No credit card
    </div>
  </div>
</aside>

<!-- MAIN -->
<main class="main">

<!-- ═══ SCREEN PAGE ═══ -->
<div id="page-screen" class="panel active">
  <div class="hero">
    <div class="hero-chip"><span class="live-dot"></span>Powered by Groq · Llama 3.3 70B</div>
    <h1>AI Resume Screening<br><span class="grad-text">Done in Seconds</span></h1>
    <p>Paste a job description and any resume. Our AI scores the match, identifies gaps, and delivers recruiter-grade insights instantly.</p>
    <div class="row gap12" style="justify-content:center;flex-wrap:wrap">
      <button class="btn-sec" onclick="loadSample()">⚡ Load Sample Data</button>
    </div>
    <div class="hero-meta mt16">
      <div class="hero-meta-item"><span>✓</span> 100% Free</div>
      <div class="hero-meta-item"><span>✓</span> No credit card</div>
      <div class="hero-meta-item"><span>✓</span> ~2s response time</div>
      <div class="hero-meta-item"><span>✓</span> 14,400 screens/day</div>
    </div>
  </div>

  <div class="form-grid">
    <!-- JD -->
    <div class="card-accent col gap14">
      <div class="sec-lbl">Job Description</div>
      <div class="fgroup"><label class="flabel">Job Title <small>*</small></label><input class="finput" id="jd-title" placeholder="e.g. Senior Backend Engineer"></div>
      <div class="fgroup"><label class="flabel">Role Description</label><textarea class="ftarea" id="jd-desc" rows="5" placeholder="Describe the role, responsibilities, team..."></textarea></div>
      <div class="fgroup"><label class="flabel">Required Skills <small>(comma separated)</small></label><input class="finput" id="jd-req" placeholder="Python, AWS, PostgreSQL, Docker"></div>
      <div class="fgroup"><label class="flabel">Nice to Have <small>(comma separated)</small></label><input class="finput" id="jd-nice" placeholder="Redis, Kubernetes, GraphQL"></div>
    </div>
    <!-- RESUME -->
    <div class="card-accent col gap14">
      <div class="sec-lbl">Candidate Resume</div>
      <div class="fgroup"><label class="flabel">Candidate Name <small>(optional)</small></label><input class="finput" id="cv-name" placeholder="Anonymous"></div>
      <!-- UPLOAD BOX -->
      <div class="fgroup">
        <label class="flabel">Upload Resume <small>(PDF or TXT — auto-fills text below)</small></label>
        <label id="upload-zone" style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;border:1.5px dashed var(--border2);border-radius:10px;padding:18px 12px;cursor:pointer;transition:all 0.2s;background:rgba(18,18,31,0.6);min-height:80px" onmouseover="this.style.borderColor='rgba(139,92,246,0.5)'" onmouseout="this.style.borderColor='var(--border2)'">
          <input type="file" id="cv-file" accept=".pdf,.txt,.md" style="display:none" onchange="handleFileUpload(this)">
          <span style="font-size:22px">📄</span>
          <span style="font-size:12px;color:var(--text2);text-align:center">Drop PDF or TXT here, or <span style="color:var(--cyan)">click to browse</span></span>
          <span id="upload-status" style="font-size:11px;color:var(--text3)">Supports PDF, TXT, MD</span>
        </label>
      </div>
      <div class="fgroup" style="flex:1"><label class="flabel">Resume Text <small>* (paste directly or upload above)</small></label><textarea class="ftarea" id="cv-text" rows="14" placeholder="Paste full resume here, or upload a file above..." style="font-family:monospace;font-size:12.5px;line-height:1.7"></textarea></div>
    </div>
  </div>

  <div class="err-box" id="serr"></div>
  <div class="between mt16">
    <div style="font-size:12px;color:var(--text3)">Groq · Llama 3.3 70B · Free tier · No billing required</div>
    <button class="btn-main" id="sbtn" onclick="doScreen()"><span>✦</span>Analyze Resume</button>
  </div>
</div>

<!-- ═══ RESULTS PAGE ═══ -->
<div id="page-results" class="panel">
  <div class="between mb32">
    <div>
      <h2 style="font-family:'Clash Display';font-size:27px;font-weight:700;letter-spacing:-0.5px">Screening Results</h2>
      <p style="color:var(--text2);font-size:13px;margin-top:4px" id="rcount">0 candidates screened</p>
    </div>
    <button class="btn-main" onclick="go('screen')"><span>+</span>Screen Another</button>
  </div>
  <div id="rlist">
    <div class="empty"><div class="empty-i">✦</div><div class="empty-t">No results yet</div><p style="font-size:14px;margin-bottom:22px">Screen your first resume to see AI insights</p><button class="btn-main" onclick="go('screen')">Start Screening</button></div>
  </div>
</div>

<!-- ═══ DASHBOARD PAGE ═══ -->
<div id="page-dashboard" class="panel">
  <div class="mb32">
    <h2 style="font-family:'Clash Display';font-size:27px;font-weight:700;letter-spacing:-0.5px">Analytics Dashboard</h2>
    <p style="color:var(--text2);font-size:13px;margin-top:4px" id="dsub">Screen resumes to see analytics</p>
  </div>
  <div id="dcont">
    <div class="empty"><div class="empty-i">📊</div><div class="empty-t">No data yet</div><p style="font-size:14px">Analytics appear after you screen resumes</p></div>
  </div>
</div>

<!-- ═══ LOGS PAGE ═══ -->
<div id="page-logs" class="panel">
  <div class="between mb20">
    <div>
      <h2 style="font-family:'Clash Display';font-size:27px;font-weight:700;letter-spacing:-0.5px">Live Log Stream</h2>
      <p style="color:var(--text2);font-size:13px;margin-top:4px">Structured JSON logs from the screening pipeline</p>
    </div>
    <div class="row gap8">
      <label class="row gap6" style="font-size:12px;color:var(--text2);cursor:pointer;gap:6px">
        <input type="checkbox" id="arf" checked onchange="setAuto(this.checked)">Auto-refresh
      </label>
      <button class="btn-ico" onclick="fetchLogs()" title="Refresh">↻</button>
      <button class="btn-ico" style="color:var(--rose)" onclick="clearLogs()" title="Clear">✕</button>
    </div>
  </div>
  <div class="mgrid mb20">
    <div class="metric"><div class="m-lbl">Total Logs</div><div class="m-val" id="ls-tot">0</div></div>
    <div class="metric"><div class="m-lbl">Errors / Warnings</div><div class="m-val"><span style="color:var(--rose)" id="ls-err">0</span><span style="color:var(--text3);font-size:18px;margin:0 6px">/</span><span style="color:var(--amber)" id="ls-warn">0</span></div></div>
    <div class="metric"><div class="m-lbl">Avg AI Score</div><div class="m-val" style="color:var(--cyan)" id="ls-sc">—</div></div>
  </div>
  <div class="row gap8 mb14" style="flex-wrap:wrap">
    <button class="btn-sec" id="lf-ALL"   onclick="setLF('ALL',this)"   style="font-size:11.5px;padding:5px 13px;border-color:rgba(139,92,246,0.4);color:#a78bfa">All</button>
    <button class="btn-sec" id="lf-INFO"  onclick="setLF('INFO',this)"  style="font-size:11.5px;padding:5px 13px">INFO</button>
    <button class="btn-sec" id="lf-OK"    onclick="setLF('OK',this)"    style="font-size:11.5px;padding:5px 13px">OK</button>
    <button class="btn-sec" id="lf-WARN"  onclick="setLF('WARN',this)"  style="font-size:11.5px;padding:5px 13px">WARN</button>
    <button class="btn-sec" id="lf-ERROR" onclick="setLF('ERROR',this)" style="font-size:11.5px;padding:5px 13px">ERROR</button>
    <input class="finput" id="lsrch" placeholder="🔍 Search..." oninput="renderLogs()" style="width:170px;padding:5px 12px;font-size:12px">
    <span style="font-size:11px;color:var(--text3);margin-left:auto;align-self:center" id="lcnt"></span>
  </div>
  <div class="card" style="padding:7px 14px;min-height:180px"><div id="lstream"></div></div>
</div>

<!-- ═══ COMPARE PAGE ═══ -->
<div id="page-compare" class="panel">
  <div class="between mb28">
    <div>
      <h2 style="font-family:'Clash Display';font-size:27px;font-weight:700;letter-spacing:-0.5px">Compare Candidates</h2>
      <p style="color:var(--text2);font-size:13px;margin-top:4px">Side-by-side comparison of screened candidates</p>
    </div>
    <button class="btn-sec" onclick="go('screen')">+ Screen More</button>
  </div>
  <div id="compare-picker" style="margin-bottom:20px">
    <div class="card-sm mb14">
      <div class="sec-lbl mb10" style="font-size:9px">Select up to 5 candidates to compare</div>
      <div id="compare-checkboxes" style="display:flex;flex-wrap:wrap;gap:8px"></div>
      <button class="btn-main mt14" onclick="renderCompare()" style="margin-top:14px">Compare Selected</button>
    </div>
  </div>
  <div id="compare-table"></div>
</div>

<!-- ═══ NOTES PAGE ═══ -->
<div id="page-notes" class="panel">
  <div class="between mb28">
    <div>
      <h2 style="font-family:'Clash Display';font-size:27px;font-weight:700;letter-spacing:-0.5px">Recruiter Notes</h2>
      <p style="color:var(--text2);font-size:13px;margin-top:4px">Keep notes on candidates, roles, or anything else</p>
    </div>
    <button class="btn-main" onclick="showNoteForm()">+ New Note</button>
  </div>
  <div id="note-form" style="display:none;margin-bottom:20px">
    <div class="card-accent col gap12">
      <div class="sec-lbl">New Note</div>
      <div class="fgroup"><label class="flabel">Title</label><input class="finput" id="note-title" placeholder="e.g. Interview feedback — John Smith"></div>
      <div class="fgroup"><label class="flabel">Content</label><textarea class="ftarea" id="note-content" rows="5" placeholder="Write your note here..."></textarea></div>
      <div class="fgroup"><label class="flabel">Tags <small>(comma separated)</small></label><input class="finput" id="note-tags" placeholder="interview, python, backend"></div>
      <div class="row gap8">
        <button class="btn-main" onclick="saveNote()">Save Note</button>
        <button class="btn-sec" onclick="document.getElementById('note-form').style.display='none'">Cancel</button>
      </div>
    </div>
  </div>
  <div id="notes-list">
    <div class="empty"><div class="empty-i">📝</div><div class="empty-t">No notes yet</div><p style="font-size:14px">Click "+ New Note" to add your first note</p></div>
  </div>
</div>

<!-- ═══ TEMPLATES PAGE ═══ -->
<div id="page-templates" class="panel">
  <div class="between mb28">
    <div>
      <h2 style="font-family:'Clash Display';font-size:27px;font-weight:700;letter-spacing:-0.5px">Job Templates</h2>
      <p style="color:var(--text2);font-size:13px;margin-top:4px">Save job descriptions and reuse them instantly</p>
    </div>
    <button class="btn-main" onclick="saveCurrentAsTemplate()">💾 Save Current JD</button>
  </div>
  <div id="templates-list">
    <div class="empty"><div class="empty-i">🗂️</div><div class="empty-t">No templates yet</div><p style="font-size:14px;margin-bottom:20px">Fill in a Job Description and click "Save Current JD"</p></div>
  </div>
</div>

<!-- ═══ SETTINGS PAGE ═══ -->
<div id="page-settings" class="panel">
  <div class="mb28">
    <h2 style="font-family:'Clash Display';font-size:27px;font-weight:700;letter-spacing:-0.5px">Settings</h2>
    <p style="color:var(--text2);font-size:13px;margin-top:4px">Configure AI model, scoring thresholds, and preferences</p>
  </div>

  <div class="col gap14">
    <!-- AI Model Settings -->
    <div class="card">
      <div class="sec-lbl mb16">🤖 AI Model Settings</div>
      <div class="g2 gap14" style="margin-bottom:14px">
        <div class="fgroup">
          <label class="flabel">Model</label>
          <select class="finput" id="set-model">
            <option value="llama-3.3-70b-versatile">Llama 3.3 70B Versatile (recommended)</option>
            <option value="llama-3.1-70b-versatile">Llama 3.1 70B Versatile</option>
            <option value="llama-3.1-8b-instant">Llama 3.1 8B Instant (faster)</option>
            <option value="mixtral-8x7b-32768">Mixtral 8x7B</option>
            <option value="gemma2-9b-it">Gemma 2 9B</option>
          </select>
        </div>
        <div class="fgroup">
          <label class="flabel">Temperature <small id="temp-val">(0.2)</small></label>
          <input type="range" id="set-temp" min="0" max="1" step="0.1" value="0.2" class="finput" style="padding:6px 0;cursor:pointer" oninput="document.getElementById('temp-val').textContent='('+this.value+')'">
        </div>
      </div>
      <div class="fgroup" style="max-width:300px">
        <label class="flabel">Max Output Tokens</label>
        <select class="finput" id="set-tokens">
          <option value="1000">1000 (faster)</option>
          <option value="1500" selected>1500 (recommended)</option>
          <option value="2000">2000 (more detail)</option>
        </select>
      </div>
    </div>

    <!-- Scoring Thresholds -->
    <div class="card">
      <div class="sec-lbl mb16">🎯 Scoring Thresholds</div>
      <div class="g2 gap14">
        <div class="fgroup">
          <label class="flabel">Strong Match threshold <small id="strong-val">(75)</small></label>
          <input type="range" id="set-strong" min="50" max="95" step="5" value="75" class="finput" style="padding:6px 0;cursor:pointer" oninput="document.getElementById('strong-val').textContent='('+this.value+')'">
          <span style="font-size:11px;color:var(--emerald)">Scores ≥ this = Strong Match</span>
        </div>
        <div class="fgroup">
          <label class="flabel">Good Match threshold <small id="good-val">(50)</small></label>
          <input type="range" id="set-good" min="20" max="74" step="5" value="50" class="finput" style="padding:6px 0;cursor:pointer" oninput="document.getElementById('good-val').textContent='('+this.value+')'">
          <span style="font-size:11px;color:var(--amber)">Scores ≥ this = Good Match</span>
        </div>
      </div>
    </div>

    <!-- Preferences -->
    <div class="card">
      <div class="sec-lbl mb16">🔧 Preferences</div>
      <div class="col gap12">
        <label style="display:flex;align-items:center;gap:12px;cursor:pointer;padding:10px 0;border-bottom:1px solid var(--border)">
          <input type="checkbox" id="set-flag" checked style="width:16px;height:16px;cursor:pointer;accent-color:var(--violet)">
          <div><div style="font-size:13px;font-weight:500">Auto-flag low confidence results</div><div style="font-size:11px;color:var(--text3);margin-top:2px">Show warning when AI confidence is below 50%</div></div>
        </label>
        <label style="display:flex;align-items:center;gap:12px;cursor:pointer;padding:10px 0;border-bottom:1px solid var(--border)">
          <input type="checkbox" id="set-notif" checked style="width:16px;height:16px;cursor:pointer;accent-color:var(--violet)">
          <div><div style="font-size:13px;font-weight:500">Show toast notifications</div><div style="font-size:11px;color:var(--text3);margin-top:2px">Display success/error notifications after screening</div></div>
        </label>
        <label style="display:flex;align-items:center;gap:12px;cursor:pointer;padding:10px 0">
          <input type="checkbox" id="set-autosave" checked style="width:16px;height:16px;cursor:pointer;accent-color:var(--violet)">
          <div><div style="font-size:13px;font-weight:500">Auto-save screening results</div><div style="font-size:11px;color:var(--text3);margin-top:2px">Automatically save results to history for export</div></div>
        </label>
      </div>
    </div>

    <!-- Data Management -->
    <div class="card">
      <div class="sec-lbl mb16">📦 Data Management</div>
      <div class="row gap10" style="flex-wrap:wrap">
        <button class="btn-main" onclick="exportResults()" style="background:linear-gradient(135deg,#0f5,#0ae)">⬇ Export Results (JSON)</button>
        <button class="btn-sec" onclick="if(confirm('Clear all screening results?')){results=[];updateSB();renderResults();renderDash();showToast('Results cleared','ok')}">🗑 Clear Results</button>
        <button class="btn-sec" onclick="if(confirm('Clear all logs?')){clearLogs();showToast('Logs cleared','ok')}">🗑 Clear Logs</button>
      </div>
    </div>

    <!-- About -->
    <div class="card">
      <div class="sec-lbl mb16">ℹ️ About</div>
      <div style="font-size:13px;color:var(--text2);line-height:1.8">
        <div class="between mb8"><span>App</span><span style="color:var(--text);font-family:'Clash Display',sans-serif">RecruitAI Pro</span></div>
        <div class="between mb8"><span>AI Provider</span><span style="color:var(--emerald)">Groq (Free Tier)</span></div>
        <div class="between mb8"><span>Model</span><span style="color:var(--cyan)" id="about-model">Llama 3.3 70B Versatile</span></div>
        <div class="between mb8"><span>Version</span><span style="color:var(--text)">2.0.0</span></div>
        <div class="between"><span>Free daily limit</span><span style="color:var(--amber)">14,400 requests / day</span></div>
      </div>
    </div>

    <button class="btn-main" onclick="saveSettings()" style="align-self:flex-start">💾 Save Settings</button>
  </div>
</div>

</main>
</div>
</div>

<!-- AUTH MODAL -->
<div id="auth-modal" style="display:none;position:fixed;inset:0;z-index:998;background:rgba(3,3,8,0.85);backdrop-filter:blur(14px);align-items:center;justify-content:center">
  <div style="background:rgba(13,13,26,0.98);border:1px solid rgba(139,92,246,0.25);border-radius:20px;padding:36px 32px;width:100%;max-width:420px;position:relative;box-shadow:0 0 60px rgba(139,92,246,0.15)">
    <button onclick="closeAuthModal()" style="position:absolute;top:16px;right:16px;background:none;border:none;color:var(--text3);font-size:20px;cursor:pointer;line-height:1">✕</button>
    <div style="text-align:center;margin-bottom:24px">
      <div style="width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,var(--violet),var(--cyan));display:flex;align-items:center;justify-content:center;font-size:20px;margin:0 auto 12px;box-shadow:0 0 20px rgba(139,92,246,0.4)">✦</div>
      <h2 id="auth-title" style="font-family:'Clash Display',sans-serif;font-size:22px;font-weight:700">Welcome Back</h2>
      <p id="auth-sub" style="font-size:13px;color:var(--text2);margin-top:4px">Login to your RecruitAI account</p>
    </div>
    <div id="auth-err" style="background:rgba(255,77,141,0.08);border:1px solid rgba(255,77,141,0.2);border-radius:8px;padding:10px 14px;color:var(--rose);font-size:13px;display:none;margin-bottom:14px"></div>
    <div id="signup-name-wrap" style="display:none;margin-bottom:12px">
      <label class="flabel">Full Name</label>
      <input class="finput" id="auth-name" placeholder="Your full name">
    </div>
    <div id="signup-email-wrap" style="display:none;margin-bottom:12px">
      <label class="flabel">Email</label>
      <input class="finput" id="auth-email" placeholder="you@example.com" type="email">
    </div>
    <div style="margin-bottom:12px">
      <label class="flabel">Username</label>
      <input class="finput" id="auth-username" placeholder="username">
    </div>
    <div style="margin-bottom:20px">
      <label class="flabel">Password</label>
      <input class="finput" id="auth-password" placeholder="••••••••" type="password" onkeydown="if(event.key==='Enter')submitAuth()">
    </div>
    <button class="btn-main" onclick="submitAuth()" id="auth-submit-btn" style="width:100%;justify-content:center">Login</button>
    <p style="text-align:center;font-size:13px;color:var(--text2);margin-top:16px">
      <span id="auth-switch-text">Don't have an account?</span>
      <span id="auth-switch-btn" onclick="switchAuthMode()" style="color:var(--cyan);cursor:pointer;margin-left:4px;font-weight:500">Sign up</span>
    </p>
  </div>
</div>

<!-- PROFILE DROPDOWN -->
<div id="profile-dropdown" style="display:none;position:fixed;top:70px;right:16px;z-index:500;background:rgba(13,13,26,0.98);border:1px solid rgba(139,92,246,0.2);border-radius:14px;padding:16px;min-width:220px;box-shadow:0 8px 32px rgba(0,0,0,0.4)">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;padding-bottom:14px;border-bottom:1px solid var(--border)">
    <div id="pd-avatar" style="width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,var(--violet),var(--cyan));display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;font-family:'Clash Display',sans-serif"></div>
    <div>
      <div id="pd-name" style="font-weight:600;font-size:14px;font-family:'Clash Display',sans-serif"></div>
      <div id="pd-email" style="font-size:11px;color:var(--text3)"></div>
    </div>
  </div>
  <div id="pd-screens" style="font-size:12px;color:var(--text2);margin-bottom:14px;padding:8px 10px;background:rgba(139,92,246,0.08);border-radius:8px;border:1px solid rgba(139,92,246,0.15)"></div>
  <button onclick="logout()" style="width:100%;background:rgba(255,77,141,0.08);border:1px solid rgba(255,77,141,0.2);color:var(--rose);border-radius:8px;padding:8px;font-size:13px;cursor:pointer;font-family:'Cabinet Grotesk',sans-serif;transition:all 0.2s" onmouseover="this.style.background='rgba(255,77,141,0.15)'" onmouseout="this.style.background='rgba(255,77,141,0.08)'">Sign out</button>
</div>

<script>
let results=[],allLogs=[],lf='ALL',atimer=null,appSettings={score_threshold_strong:75,score_threshold_good:50,notifications:true,autosave:true};
const VC={STRONG_MATCH:'#00f5a0',GOOD_MATCH:'#4ade80',PARTIAL_MATCH:'#ffb340',WEAK_MATCH:'#fb923c',NO_MATCH:'#ff4d8d'};
const VG={STRONG_MATCH:'rgba(0,245,160,0.12)',GOOD_MATCH:'rgba(74,222,128,0.12)',PARTIAL_MATCH:'rgba(255,179,64,0.12)',WEAK_MATCH:'rgba(251,146,60,0.12)',NO_MATCH:'rgba(255,77,141,0.12)'};
const LB={INFO:'bc',WARN:'ba',ERROR:'br',OK:'be',DEBUG:'bv'};

// ── TOAST ─────────────────────────────────────────────────────
function showToast(msg, type='ok'){
  if(!appSettings.notifications) return;
  const t=document.createElement('div');
  const colors={ok:'rgba(0,245,160,0.15)',err:'rgba(255,77,141,0.15)',info:'rgba(0,229,255,0.15)'};
  const borders={ok:'rgba(0,245,160,0.3)',err:'rgba(255,77,141,0.3)',info:'rgba(0,229,255,0.3)'};
  const icons={ok:'✓',err:'⚠',info:'ℹ'};
  t.style.cssText=`position:fixed;bottom:24px;right:24px;z-index:9999;padding:12px 18px;border-radius:10px;background:${colors[type]};border:1px solid ${borders[type]};color:var(--text);font-size:13px;backdrop-filter:blur(12px);display:flex;align-items:center;gap:8px;animation:fadeUp 0.3s ease;max-width:320px;box-shadow:0 8px 24px rgba(0,0,0,0.3)`;
  t.innerHTML=`<span style="font-size:15px">${icons[type]}</span><span>${msg}</span>`;
  document.body.appendChild(t);
  setTimeout(()=>{t.style.opacity='0';t.style.transition='opacity 0.4s';setTimeout(()=>t.remove(),400)},3000);
}

// ── NAVIGATION ────────────────────────────────────────────────
function go(page,nb,sb){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  const pg=document.getElementById('page-'+page);
  if(pg) pg.classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.s-item').forEach(b=>b.classList.remove('active'));
  if(nb) nb.classList.add('active');
  if(sb) sb.classList.add('active');
  // sync sidebar by index from button order
  const sis=document.querySelectorAll('.s-item');
  const sidemap={screen:0,results:1,compare:2,dashboard:3,notes:4,templates:5,logs:6,settings:7};
  if(sidemap[page]!==undefined) sis[sidemap[page]]?.classList.add('active');
  if(page==='logs') fetchLogs();
  if(page==='dashboard') renderDash();
  if(page==='results') renderResults();
  if(page==='compare') renderComparePicker();
  if(page==='notes') fetchNotes();
  if(page==='templates') fetchTemplates();
  if(page==='settings') loadSettingsUI();
}

// ── SETTINGS ─────────────────────────────────────────────────
async function loadSettingsUI(){
  try{
    const r=await fetch('/api/settings');
    if(!r.ok) return;
    const s=await r.json();
    document.getElementById('set-model').value=s.model||'llama-3.3-70b-versatile';
    document.getElementById('set-temp').value=s.temperature||0.2;
    document.getElementById('temp-val').textContent='('+s.temperature+')';
    document.getElementById('set-tokens').value=s.max_tokens||1500;
    document.getElementById('set-strong').value=s.score_threshold_strong||75;
    document.getElementById('strong-val').textContent='('+s.score_threshold_strong+')';
    document.getElementById('set-good').value=s.score_threshold_good||50;
    document.getElementById('good-val').textContent='('+s.score_threshold_good+')';
    document.getElementById('set-flag').checked=s.auto_flag_low_confidence!==false;
    document.getElementById('set-notif').checked=s.notifications!==false;
    document.getElementById('about-model').textContent=s.model||'llama-3.3-70b-versatile';
    appSettings.score_threshold_strong=s.score_threshold_strong||75;
    appSettings.score_threshold_good=s.score_threshold_good||50;
    appSettings.notifications=s.notifications!==false;
  }catch(e){}
}

async function saveSettings(){
  const payload={
    model: document.getElementById('set-model').value,
    temperature: parseFloat(document.getElementById('set-temp').value),
    max_tokens: parseInt(document.getElementById('set-tokens').value),
    score_threshold_strong: parseInt(document.getElementById('set-strong').value),
    score_threshold_good: parseInt(document.getElementById('set-good').value),
    auto_flag_low_confidence: document.getElementById('set-flag').checked,
    notifications: document.getElementById('set-notif').checked,
  };
  try{
    const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if(r.ok){
      appSettings={...appSettings,...payload};
      document.getElementById('about-model').textContent=payload.model;
      showToast('Settings saved successfully!','ok');
    }
  }catch(e){showToast('Failed to save settings','err');}
}

// ── NOTES ─────────────────────────────────────────────────────
function showNoteForm(){
  const f=document.getElementById('note-form');
  f.style.display=f.style.display==='none'?'block':'none';
  if(f.style.display==='block') document.getElementById('note-title').focus();
}

async function fetchNotes(){
  try{
    const r=await fetch('/api/notes');
    if(!r.ok) return;
    const d=await r.json();
    renderNotes(d.notes||[]);
  }catch(e){}
}

function renderNotes(notes){
  const el=document.getElementById('notes-list');
  if(!notes.length){el.innerHTML='<div class="empty"><div class="empty-i">📝</div><div class="empty-t">No notes yet</div><p style="font-size:14px">Click "+ New Note" to add your first note</p></div>';return;}
  el.innerHTML=notes.map(n=>`
    <div class="card mb12" style="position:relative">
      <div class="between mb8">
        <div style="font-family:\'Clash Display\';font-size:15px;font-weight:600">${n.title}</div>
        <div class="row gap8">
          <span style="font-size:11px;color:var(--text3)">${n.created_at.slice(0,10)}</span>
          <button onclick="deleteNote('${n.id}')" style="background:none;border:none;color:var(--text3);cursor:pointer;font-size:16px;line-height:1" title="Delete">✕</button>
        </div>
      </div>
      <p style="font-size:13px;color:var(--text2);line-height:1.7;white-space:pre-wrap;margin-bottom:10px">${n.content}</p>
      ${n.tags&&n.tags.length?`<div class="row gap4" style="flex-wrap:wrap">${n.tags.map(t=>`<span class="tag tag-m" style="font-size:10px">${t}</span>`).join('')}</div>`:''}
    </div>`).join('');
}

async function saveNote(){
  const title=document.getElementById('note-title').value.trim();
  const content=document.getElementById('note-content').value.trim();
  if(!title||!content){showToast('Title and content are required','err');return;}
  const tags=document.getElementById('note-tags').value.split(',').map(t=>t.trim()).filter(Boolean);
  try{
    const r=await fetch('/api/notes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title,content,tags})});
    if(r.ok){
      document.getElementById('note-title').value='';
      document.getElementById('note-content').value='';
      document.getElementById('note-tags').value='';
      document.getElementById('note-form').style.display='none';
      fetchNotes();
      showToast('Note saved!','ok');
    }
  }catch(e){showToast('Failed to save note','err');}
}

async function deleteNote(id){
  if(!confirm('Delete this note?')) return;
  await fetch('/api/notes/'+id,{method:'DELETE'});
  fetchNotes();
  showToast('Note deleted','info');
}

// ── TEMPLATES ─────────────────────────────────────────────────
async function fetchTemplates(){
  try{
    const r=await fetch('/api/templates');
    if(!r.ok) return;
    const d=await r.json();
    renderTemplates(d.templates||[]);
  }catch(e){}
}

function renderTemplates(templates){
  const el=document.getElementById('templates-list');
  if(!templates.length){el.innerHTML='<div class="empty"><div class="empty-i">🗂️</div><div class="empty-t">No templates yet</div><p style="font-size:14px;margin-bottom:20px">Fill in a Job Description and click "Save Current JD"</p></div>';return;}
  el.innerHTML=templates.map(t=>`
    <div class="card mb12">
      <div class="between mb10">
        <div>
          <div style="font-family:\'Clash Display\';font-size:15px;font-weight:600">${t.name}</div>
          <div style="font-size:12px;color:var(--text3);margin-top:2px">${t.title}</div>
        </div>
        <div class="row gap8">
          <button class="btn-sec" onclick="loadTemplate('${t.id}')" style="font-size:11px;padding:5px 12px">↩ Load</button>
          <button onclick="deleteTemplate('${t.id}')" style="background:none;border:none;color:var(--text3);cursor:pointer;font-size:16px" title="Delete">✕</button>
        </div>
      </div>
      <div class="row gap6" style="flex-wrap:wrap">
        ${t.required_skills.map(s=>`<span class="tag tag-m" style="font-size:10px">${s}</span>`).join('')}
        ${(t.nice_to_have||[]).map(s=>`<span class="tag tag-b" style="font-size:10px">${s}</span>`).join('')}
      </div>
    </div>`).join('');
}

async function saveCurrentAsTemplate(){
  const title=document.getElementById('jd-title').value.trim();
  if(!title){showToast('Fill in a Job Title first on the Screen page','err');go('screen');return;}
  const name=prompt('Template name (e.g. "Backend Engineer Role"):',title);
  if(!name) return;
  const payload={
    name,title,
    description:document.getElementById('jd-desc').value,
    required_skills:document.getElementById('jd-req').value.split(',').map(s=>s.trim()).filter(Boolean),
    nice_to_have:document.getElementById('jd-nice').value.split(',').map(s=>s.trim()).filter(Boolean),
  };
  try{
    const r=await fetch('/api/templates',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if(r.ok){showToast('Template saved!','ok');fetchTemplates();}
  }catch(e){showToast('Failed to save template','err');}
}

async function loadTemplate(id){
  const r=await fetch('/api/templates');
  const d=await r.json();
  const t=d.templates.find(x=>x.id===id);
  if(!t) return;
  document.getElementById('jd-title').value=t.title;
  document.getElementById('jd-desc').value=t.description;
  document.getElementById('jd-req').value=t.required_skills.join(', ');
  document.getElementById('jd-nice').value=(t.nice_to_have||[]).join(', ');
  go('screen');
  showToast('Template loaded — ready to screen!','ok');
}

async function deleteTemplate(id){
  if(!confirm('Delete this template?')) return;
  await fetch('/api/templates/'+id,{method:'DELETE'});
  fetchTemplates();
  showToast('Template deleted','info');
}

// ── COMPARE ───────────────────────────────────────────────────
let compareSelected=[];

function renderComparePicker(){
  const el=document.getElementById('compare-checkboxes');
  if(!results.length){el.innerHTML='<span style="color:var(--text3);font-size:13px">No results yet — screen some resumes first.</span>';return;}
  el.innerHTML=results.map((r,i)=>{
    const vc=VC[r.verdict]||'var(--text3)';
    return `<label style="display:flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid var(--border);border-radius:8px;cursor:pointer;transition:all 0.2s;font-size:13px" onmouseover="this.style.borderColor=\'var(--border2)\'" onmouseout="this.style.borderColor=\'var(--border)\'">
      <input type="checkbox" value="${i}" onchange="toggleCompare(${i},this.checked)" style="accent-color:var(--violet);width:14px;height:14px">
      <span style="font-weight:500">${r.candidateName}</span>
      <span style="color:var(--text3)">·</span>
      <span style="font-family:\'Clash Display\';font-weight:700;color:${vc}">${r.score}</span>
      <span class="badge" style="background:${(VC[r.verdict]||'#888')}18;color:${vc};border-color:${vc}35;font-size:9px">${(r.verdict||'').replace(/_/g,' ')}</span>
    </label>`;
  }).join('');
}

function toggleCompare(i,checked){
  if(checked){if(compareSelected.length>=5){showToast('Max 5 candidates for comparison','err');event.target.checked=false;return;}compareSelected.push(i);}
  else{compareSelected=compareSelected.filter(x=>x!==i);}
}

function renderCompare(){
  if(compareSelected.length<2){showToast('Select at least 2 candidates to compare','err');return;}
  const sel=compareSelected.map(i=>results[i]);
  const fields=[
    ['Score','score',r=>r.score],
    ['Match %','match',r=>Math.round((r.match_pct||0)*100)+'%'],
    ['Confidence','conf',r=>Math.round((r.confidence||0)*100)+'%'],
    ['Verdict','verdict',r=>(r.verdict||'').replace(/_/g,' ')],
    ['Experience','exp',r=>(r.experience_years||0)+' years'],
    ['Matched Skills','sm',r=>(r.skills_matched||[]).join(', ')||'None'],
    ['Missing Skills','sx',r=>(r.skills_missing||[]).join(', ')||'None'],
    ['Latency','lat',r=>r.latency_ms+'ms'],
  ];
  const el=document.getElementById('compare-table');
  el.innerHTML=`<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead><tr style="border-bottom:1px solid var(--border)">
      <th style="text-align:left;padding:10px 12px;color:var(--text3);font-size:10px;text-transform:uppercase;letter-spacing:0.08em;font-family:\'Clash Display\';width:140px">Criteria</th>
      ${sel.map(r=>`<th style="text-align:left;padding:10px 12px;font-family:\'Clash Display\';font-size:13px;color:var(--text)">${r.candidateName}</th>`).join('')}
    </tr></thead>
    <tbody>
      ${fields.map(([label,,fn])=>`<tr style="border-bottom:1px solid var(--border)">
        <td style="padding:10px 12px;color:var(--text3);font-size:12px;font-weight:500">${label}</td>
        ${sel.map(r=>{
          const v=fn(r);
          let color='var(--text)';
          if(label==='Score'){const s=parseInt(v);color=s>=appSettings.score_threshold_strong?'var(--emerald)':s>=appSettings.score_threshold_good?'var(--amber)':'var(--rose)';}
          if(label==='Verdict'){color=VC[r.verdict]||'var(--text)';}
          if(label==='Missing Skills'&&v!=='None') color='var(--rose)';
          if(label==='Matched Skills'&&v!=='None') color='var(--emerald)';
          return `<td style="padding:10px 12px;color:${color};font-weight:${label==='Score'?'700':'400'};font-family:${label==='Score'?'\'Clash Display\'':'inherit'};font-size:${label==='Score'?'16px':'13px'}">${v}</td>`;
        }).join('')}
      </tr>`).join('')}
    </tbody>
  </table></div>
  <div style="margin-top:16px;font-size:11px;color:var(--text3)">Comparing ${sel.length} candidates · Scores use current threshold settings</div>`;
}

// ── EXPORT ────────────────────────────────────────────────────
async function exportResults(){
  if(!results.length){showToast('No results to export yet','err');return;}
  // Save to backend then download
  for(const r of results){
    try{ await fetch('/api/history',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(r)}); }catch(e){}
  }
  // Trigger download
  const a=document.createElement('a');
  a.href='/api/export';
  a.download='recruitai_results.json';
  a.click();
  showToast('Results exported!','ok');
}

function loadSample(){
  document.getElementById('jd-title').value='Senior Python Backend Engineer';
  document.getElementById('jd-desc').value='We are building next-generation developer tools. Looking for a senior backend engineer to design and scale our core APIs serving millions of developers worldwide. You\'ll work closely with product, data science, and frontend teams.';
  document.getElementById('jd-req').value='Python, FastAPI, PostgreSQL, AWS, Docker';
  document.getElementById('jd-nice').value='Redis, Kubernetes, GraphQL';
  document.getElementById('cv-name').value='Priya Sharma';
  document.getElementById('cv-text').value=`Priya Sharma
priya.sharma@dev.io | linkedin.com/in/priyasharma | github.com/priyasharma

SUMMARY
Staff backend engineer with 7 years building high-throughput distributed systems. Led teams of 4–6 engineers, shipped products at 5M+ users scale.

EXPERIENCE

Staff Software Engineer — DevTools Inc (2022–Present)
• Architected FastAPI microservices handling 8M+ daily API calls with 99.98% uptime
• Redesigned PostgreSQL schema and indexing — cut P99 latency from 400ms to 35ms
• Built multi-region AWS ECS deployment with zero-downtime blue/green deploys
• Introduced Celery + Redis async task queue — reduced endpoint latency 55%
• Mentored 3 junior engineers, drove backend chapter OKRs

Senior Engineer — CloudBase (2019–2022)
• Real-time data pipeline: 2TB/day with Python + Kafka + Postgres
• Containerized 12 legacy services with Docker, CI/CD via GitHub Actions
• GraphQL federation layer — reduced client round-trips by 40%

Software Engineer — FinEdge (2017–2019)
• REST API development in Flask + SQLAlchemy
• Payment API integrations (Stripe, Razorpay)

SKILLS
Python, FastAPI, Flask, PostgreSQL, Redis, AWS (EC2/ECS/S3/RDS/Lambda), Docker, Kubernetes, Kafka, GraphQL, Terraform, Git, Linux

EDUCATION
B.Tech Computer Engineering — IIT Bombay, 2017 · GPA 9.1/10`;
}

async function doScreen(){
  const title=document.getElementById('jd-title').value.trim();
  const resume=document.getElementById('cv-text').value.trim();
  const err=document.getElementById('serr');
  if(!title||!resume){err.textContent='⚠ Please fill in Job Title and Resume Text.';err.style.display='block';return;}
  err.style.display='none';
  document.getElementById('lov').classList.add('show');
  document.getElementById('sbtn').disabled=true;
  try{
    const r=await fetch('/api/screen',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        job_description:{title,description:document.getElementById('jd-desc').value,required_skills:document.getElementById('jd-req').value.split(',').map(s=>s.trim()).filter(Boolean),nice_to_have:document.getElementById('jd-nice').value.split(',').map(s=>s.trim()).filter(Boolean)},
        resume_text:resume,
        candidate_name:document.getElementById('cv-name').value||'Anonymous',
      }),
    });
    if(!r.ok){const e=await r.json();throw new Error(e.detail||'Screening failed');}
    const d=await r.json();
    d.candidateName=document.getElementById('cv-name').value||'Anonymous';
    d.jobTitle=title;
    results.unshift(d);
    updateSB();
    showToast(`Score: ${d.score} · ${(d.verdict||'').replace(/_/g,' ')}`, d.score>=75?'ok':d.score>=50?'info':'err');
    if(appSettings.autosave){try{await fetch('/api/history',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});}catch(e){}}
    go('results');
  }catch(e){
    err.textContent='⚠ '+e.message;err.style.display='block';
  }finally{
    document.getElementById('lov').classList.remove('show');
    document.getElementById('sbtn').disabled=false;
  }
}

function ring(score){
  const c=score>=75?'var(--emerald)':score>=50?'var(--amber)':'var(--rose)';
  const R=50,ci=2*Math.PI*R,fi=(score/100)*ci;
  return `<div class="sr-wrap">
    <svg width="126" height="126" style="transform:rotate(-90deg)">
      <circle cx="63" cy="63" r="${R}" fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="8"/>
      <circle cx="63" cy="63" r="${R}" fill="none" stroke="${c}" stroke-width="8" stroke-dasharray="${fi} ${ci}" stroke-linecap="round" style="filter:drop-shadow(0 0 7px ${c})"/>
    </svg>
    <div class="sr-inner"><div class="sr-num" style="color:${c}">${score}</div><div class="sr-den">/ 100</div></div>
  </div>`;
}

function renderResults(){
  document.getElementById('rcount').textContent=results.length+' candidate'+(results.length!==1?'s':'')+' screened';
  if(!results.length){document.getElementById('rlist').innerHTML='<div class="empty"><div class="empty-i">✦</div><div class="empty-t">No results yet</div><p style="font-size:14px;margin-bottom:22px">Screen a resume to see AI insights here</p><button class="btn-main" onclick="go(\'screen\')">Start Screening</button></div>';return;}
  document.getElementById('rlist').innerHTML=results.map((r,i)=>{
    const vc=VC[r.verdict]||'var(--text3)',vg=VG[r.verdict]||'transparent';
    return `<div class="rc" style="${i===0?'border-color:rgba(139,92,246,0.3)':''}">
      <div class="rc-hdr" onclick="tog(${i})">
        <div class="row gap10" style="flex-wrap:wrap">
          <span style="font-family:'Clash Display';font-size:14px;font-weight:600">${r.candidateName}</span>
          <span style="color:var(--text3);font-size:12px">→</span>
          <span style="color:var(--text2);font-size:13px">${r.jobTitle}</span>
          <span class="badge" style="background:${vg};color:${vc};border-color:${vc}40">${(r.verdict||'').replace(/_/g,' ')}</span>
        </div>
        <div class="row gap10">
          <span style="font-family:'Clash Display';font-size:23px;font-weight:700;color:${vc}">${r.score}</span>
          <span id="arr-${i}" style="color:var(--text3)">${i===0?'▲':'▼'}</span>
        </div>
      </div>
      <div id="rb-${i}" style="display:${i===0?'block':'none'}" class="rc-body">
        <div class="row gap18 mb16 mt20" style="align-items:flex-start">
          ${ring(r.score)}
          <div style="flex:1">
            <div class="sec-lbl mb8" style="font-size:9.5px">AI Summary</div>
            <p style="font-size:13.5px;line-height:1.75;color:var(--text2)">${r.summary||''}</p>
            <div class="row gap8 mt10" style="flex-wrap:wrap">
              <span class="badge bc">Match ${Math.round((r.match_pct||0)*100)}%</span>
              <span class="badge bv">Confidence ${Math.round((r.confidence||0)*100)}%</span>
              <span class="badge bc">~${r.experience_years||0} yrs exp</span>
              <span class="badge bg">${r.latency_ms}ms · Groq ⚡</span>
              <span class="badge bg">${(r.tokens_used||0).toLocaleString()} tokens</span>
            </div>
          </div>
        </div>
        <div class="g3 mb14">
          <div class="card-sm"><div class="sec-lbl mb8" style="font-size:9px">✅ Skills Matched</div><div class="tw">${(r.skills_matched||[]).map(s=>`<span class="tag tag-m">${s}</span>`).join('')||'<span style="color:var(--text3);font-size:12px">None</span>'}</div></div>
          <div class="card-sm"><div class="sec-lbl mb8" style="font-size:9px">❌ Skills Missing</div><div class="tw">${(r.skills_missing||[]).length?(r.skills_missing||[]).map(s=>`<span class="tag tag-x">${s}</span>`).join(''):'<span style="color:var(--emerald);font-size:12px">None! 🎉</span>'}</div></div>
          <div class="card-sm"><div class="sec-lbl mb8" style="font-size:9px">⭐ Bonus Skills</div><div class="tw">${(r.skills_bonus||[]).length?(r.skills_bonus||[]).map(s=>`<span class="tag tag-b">${s}</span>`).join(''):'<span style="color:var(--text3);font-size:12px">None</span>'}</div></div>
        </div>
        <div class="g2 mb14">
          <div class="card-sm">
            <div class="sec-lbl mb8" style="font-size:9px">💪 Strengths</div>
            ${(r.strengths||[]).map(s=>`<div class="row gap8 mb8" style="font-size:13px;align-items:flex-start"><span style="color:var(--emerald);flex-shrink:0;margin-top:2px">▸</span><span style="color:var(--text2);line-height:1.5">${s}</span></div>`).join('')}
          </div>
          <div class="card-sm">
            <div class="sec-lbl mb8" style="font-size:9px">⚠ Concerns</div>
            ${(r.concerns||[]).length?(r.concerns||[]).map(c=>`<div class="row gap8 mb8" style="font-size:13px;align-items:flex-start"><span style="color:var(--amber);flex-shrink:0;margin-top:2px">▸</span><span style="color:var(--text2);line-height:1.5">${c}</span></div>`).join(''):'<span style="color:var(--text3);font-size:13px">No major concerns</span>'}
          </div>
        </div>
        <div style="background:linear-gradient(135deg,rgba(139,92,246,0.08),rgba(0,229,255,0.04));border:1px solid rgba(139,92,246,0.18);border-radius:12px;padding:15px 18px;margin-bottom:14px">
          <div class="sec-lbl mb6" style="font-size:9px">🎯 Recommendation</div>
          <p style="font-size:13.5px;color:var(--text);line-height:1.6">${r.recommendation||''}</p>
        </div>
        ${(r.interview_questions||[]).length?`<div class="card-sm mb14">
          <div class="sec-lbl mb8" style="font-size:9px">💬 Suggested Interview Questions</div>
          <ol style="padding-left:18px">${(r.interview_questions||[]).map(q=>`<li style="font-size:13px;color:var(--text2);padding:4px 0;line-height:1.55">${q}</li>`).join('')}</ol>
        </div>`:''}
        <div style="font-size:10px;color:var(--text3);font-family:monospace">trace_id: ${r.trace_id} · screen_id: ${r.screen_id}</div>
      </div>
    </div>`;
  }).join('');
}

function tog(i){const b=document.getElementById('rb-'+i),a=document.getElementById('arr-'+i),o=b.style.display==='block';b.style.display=o?'none':'block';a.textContent=o?'▼':'▲';}

function renderDash(){
  const el=document.getElementById('dcont');
  if(!results.length){el.innerHTML='<div class="empty"><div class="empty-i">📊</div><div class="empty-t">No data yet</div><p style="font-size:14px">Screen some resumes first</p></div>';return;}
  const avg=Math.round(results.reduce((a,b)=>a+b.score,0)/results.length);
  const strong=results.filter(r=>r.score>=75).length;
  const avgLat=Math.round(results.reduce((a,b)=>a+(b.latency_ms||0),0)/results.length);
  const totTok=results.reduce((a,b)=>a+(b.tokens_used||0),0);
  const bk={'90–100':0,'75–89':0,'50–74':0,'25–49':0,'0–24':0};
  results.forEach(r=>{if(r.score>=90)bk['90–100']++;else if(r.score>=75)bk['75–89']++;else if(r.score>=50)bk['50–74']++;else if(r.score>=25)bk['25–49']++;else bk['0–24']++;});
  const vc={};results.forEach(r=>{if(r.verdict)vc[r.verdict]=(vc[r.verdict]||0)+1;});
  const sf={};results.forEach(r=>(r.skills_matched||[]).forEach(s=>{sf[s]=(sf[s]||0)+1;}));
  const top=Object.entries(sf).sort((a,b)=>b[1]-a[1]).slice(0,6);
  const bc=k=>k.startsWith('9')||k.startsWith('7')?'var(--emerald)':k.startsWith('5')?'var(--amber)':'var(--rose)';
  const bar=(l,v,mx,c)=>{const p=mx>0?Math.round(v/mx*100):0;return `<div class="bar-row"><div class="bar-top"><span class="bar-lbl">${l}</span><span class="bar-v">${v}</span></div><div class="bar-tr"><div class="bar-fi" style="width:${p}%;background:${c}"></div></div></div>`;};
  document.getElementById('dsub').textContent=`Analytics · ${results.length} screening${results.length!==1?'s':''}`;
  el.innerHTML=`
  <div class="mgrid mb24">
    <div class="metric"><div class="m-lbl">Avg Match Score</div><div class="m-val" style="color:${avg>=75?'var(--emerald)':avg>=50?'var(--amber)':'var(--rose)'}">${avg}</div><div class="m-sub">out of 100</div></div>
    <div class="metric"><div class="m-lbl">Strong Matches ≥75</div><div class="m-val" style="color:var(--emerald)">${strong}<span style="font-size:16px;color:var(--text3);margin-left:8px">${Math.round(strong/results.length*100)}%</span></div><div class="m-sub">of candidates</div></div>
    <div class="metric"><div class="m-lbl">Avg Groq Latency</div><div class="m-val" style="color:var(--cyan)">${avgLat}<span style="font-size:14px;margin-left:4px">ms</span></div><div class="m-sub">${totTok.toLocaleString()} total tokens</div></div>
  </div>
  <div class="g2 mb18">
    <div class="card"><div class="sec-lbl mb14">Score Distribution</div>${Object.entries(bk).map(([k,v])=>bar(k,v,results.length,bc(k))).join('')}</div>
    <div class="card"><div class="sec-lbl mb14">Verdict Breakdown</div>${Object.entries(vc).map(([v,c])=>bar(v.replace(/_/g,' '),c,results.length,VC[v]||'var(--violet)')).join('')}</div>
  </div>
  ${top.length?`<div class="card mb18"><div class="sec-lbl mb14">Top Matched Skills</div><div class="g2">${top.map(([s,c])=>bar(s,c,results.length,'var(--violet)')).join('')}</div></div>`:''}
  <div class="card">
    <div class="sec-lbl mb14">Screening History</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr style="border-bottom:1px solid var(--border)">${['Candidate','Role','Score','Verdict','Time','Tokens'].map(h=>`<th style="text-align:left;padding:8px 0;color:var(--text3);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:'Clash Display'">${h}</th>`).join('')}</tr></thead>
      <tbody>${results.map(r=>`<tr style="border-bottom:1px solid var(--border)">
        <td style="padding:10px 0;font-weight:500">${r.candidateName||'—'}</td>
        <td style="color:var(--text2);max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:10px 0">${r.jobTitle}</td>
        <td style="font-family:'Clash Display';font-weight:700;color:${r.score>=75?'var(--emerald)':r.score>=50?'var(--amber)':'var(--rose)'};padding:10px 0">${r.score}</td>
        <td style="padding:10px 0"><span class="badge" style="background:${(VC[r.verdict]||'#888')}18;color:${VC[r.verdict]||'var(--text3)'};border-color:${(VC[r.verdict]||'#888')}35;font-size:9px">${(r.verdict||'').replace(/_/g,' ')}</span></td>
        <td style="color:var(--text3);padding:10px 0">${r.latency_ms}ms</td>
        <td style="color:var(--text3);padding:10px 0">${(r.tokens_used||0).toLocaleString()}</td>
      </tr>`).join('')}</tbody>
    </table>
  </div>`;
}

function setLF(level,btn){lf=level;document.querySelectorAll('[id^="lf-"]').forEach(b=>{b.style.borderColor='';b.style.color='';});btn.style.borderColor='rgba(139,92,246,0.45)';btn.style.color='#a78bfa';renderLogs();}
function renderLogs(){
  const q=(document.getElementById('lsrch')?.value||'').toLowerCase();
  const fil=allLogs.filter(l=>{if(lf!=='ALL'&&l.level!==lf)return false;if(q&&!JSON.stringify(l).toLowerCase().includes(q))return false;return true;});
  document.getElementById('lcnt').textContent=fil.length+' entries';
  const el=document.getElementById('lstream');
  if(!fil.length){el.innerHTML='<div style="text-align:center;padding:30px;color:var(--text3);font-size:13px">No logs yet — screen a resume to generate logs.</div>';return;}
  el.innerHTML=fil.map(l=>{
    const t=(l.ts||'').slice(11,19);
    const{ts,event,level,service,log_id,...rest}=l;
    const prev=Object.entries(rest).slice(0,3).map(([k,v])=>`${k}=${JSON.stringify(v)}`).join(' · ');
    return `<div class="log-row" onclick="this.nextSibling.style.display=this.nextSibling.style.display==='block'?'none':'block'">
      <span class="log-time">${t}</span>
      <span class="badge ${LB[level]||'bv'}">${level}</span>
      <span style="font-family:monospace;font-size:11px;color:var(--text)">${event}</span>
      <span style="font-size:10.5px;color:var(--text3);font-family:monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${prev}</span>
    </div><div class="log-detail">${JSON.stringify(l,null,2)}</div>`;
  }).join('');
  document.getElementById('ls-tot').textContent=allLogs.length;
  document.getElementById('ls-err').textContent=allLogs.filter(l=>l.level==='ERROR').length;
  document.getElementById('ls-warn').textContent=allLogs.filter(l=>l.level==='WARN').length;
  const sc=allLogs.filter(l=>l.score).map(l=>l.score);
  document.getElementById('ls-sc').textContent=sc.length?Math.round(sc.reduce((a,b)=>a+b,0)/sc.length):'—';
}
async function fetchLogs(){try{const r=await fetch('/api/logs?limit=200');if(r.ok){const d=await r.json();allLogs=d.logs||[];renderLogs();}}catch(e){}}
function setAuto(on){clearInterval(atimer);if(on)atimer=setInterval(()=>{if(document.getElementById('page-logs').classList.contains('active'))fetchLogs();},3000);}
async function clearLogs(){await fetch('/api/logs',{method:'DELETE'});allLogs=[];renderLogs();}
function updateSB(){document.getElementById('sb-hist').innerHTML=results.slice(0,5).map(r=>`<button class="s-item" onclick="go('results')" style="font-size:11px"><span>${r.score>=75?'🟢':r.score>=50?'🟡':'🔴'}</span><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${r.candidateName} · ${r.score}</span></button>`).join('');}
// ── FILE UPLOAD ───────────────────────────────────────────────
async function handleFileUpload(input) {
  const file = input.files[0];
  if (!file) return;
  const status = document.getElementById('upload-status');
  const zone = document.getElementById('upload-zone');
  status.textContent = '⏳ Reading file...';
  status.style.color = 'var(--amber)';
  zone.style.borderColor = 'rgba(139,92,246,0.5)';
  try {
    const form = new FormData();
    form.append('file', file);
    const resp = await fetch('/api/upload-resume', { method: 'POST', body: form });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || 'Upload failed'); }
    const data = await resp.json();
    document.getElementById('cv-text').value = data.text;
    // Try to extract name from filename
    if (!document.getElementById('cv-name').value) {
      const namePart = file.name.replace(/\.(pdf|txt|md)$/i, '').replace(/[_\-]/g, ' ').replace(/resume|cv/gi, '').trim();
      if (namePart) document.getElementById('cv-name').value = namePart;
    }
    status.textContent = `✓ ${file.name} loaded (${data.text.length} chars)`;
    status.style.color = 'var(--emerald)';
    zone.style.borderColor = 'rgba(0,245,160,0.4)';
  } catch(e) {
    status.textContent = '⚠ ' + e.message;
    status.style.color = 'var(--rose)';
    zone.style.borderColor = 'rgba(255,77,141,0.4)';
  }
}

// Make upload zone work as click-to-browse
document.addEventListener('DOMContentLoaded', () => {
  const zone = document.getElementById('upload-zone');
  if (zone) {
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor='rgba(139,92,246,0.6)'; });
    zone.addEventListener('dragleave', () => { zone.style.borderColor='var(--border2)'; });
    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.style.borderColor='var(--border2)';
      const f = e.dataTransfer.files[0];
      if (f) { document.getElementById('cv-file').files = e.dataTransfer.files; handleFileUpload(document.getElementById('cv-file')); }
    });
  }
});

// ── AUTH ──────────────────────────────────────────────────────
let authMode = 'login';   // 'login' or 'signup'
let currentUser = null;

function showAuthModal(mode) {
  authMode = mode || 'login';
  document.getElementById('auth-modal').style.display = 'flex';
  document.getElementById('auth-err').style.display = 'none';
  document.getElementById('auth-username').value = '';
  document.getElementById('auth-password').value = '';
  document.getElementById('auth-name').value = '';
  document.getElementById('auth-email').value = '';
  updateAuthUI();
}

function updateAuthUI() {
  const isLogin = authMode === 'login';
  document.getElementById('auth-title').textContent = isLogin ? 'Welcome Back' : 'Create Account';
  document.getElementById('auth-sub').textContent = isLogin ? 'Login to your RecruitAI account' : 'Sign up — it\'s free!';
  document.getElementById('auth-submit-btn').textContent = isLogin ? 'Login' : 'Create Account';
  document.getElementById('auth-switch-text').textContent = isLogin ? "Don't have an account?" : 'Already have an account?';
  document.getElementById('auth-switch-btn').textContent = isLogin ? 'Sign up' : 'Login';
  document.getElementById('signup-name-wrap').style.display = isLogin ? 'none' : 'block';
  document.getElementById('signup-email-wrap').style.display = isLogin ? 'none' : 'block';
}

function switchAuthMode() {
  authMode = authMode === 'login' ? 'signup' : 'login';
  document.getElementById('auth-err').style.display = 'none';
  updateAuthUI();
}

function closeAuthModal() {
  document.getElementById('auth-modal').style.display = 'none';
}

async function submitAuth() {
  const username = document.getElementById('auth-username').value.trim();
  const password = document.getElementById('auth-password').value;
  const errEl = document.getElementById('auth-err');
  if (!username || !password) { errEl.textContent = '⚠ Username and password are required.'; errEl.style.display = 'block'; return; }
  const btn = document.getElementById('auth-submit-btn');
  btn.disabled = true; btn.textContent = 'Please wait...';
  try {
    let resp, data;
    if (authMode === 'login') {
      resp = await fetch('/api/login', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ username, password }) });
    } else {
      const name = document.getElementById('auth-name').value.trim();
      const email = document.getElementById('auth-email').value.trim();
      if (!name) { throw new Error('Please enter your full name.'); }
      resp = await fetch('/api/signup', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ username, password, name, email }) });
    }
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || 'Auth failed'); }
    data = await resp.json();
    currentUser = data;
    closeAuthModal();
    updateHeaderUser();
  } catch(e) {
    errEl.textContent = '⚠ ' + e.message;
    errEl.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = authMode === 'login' ? 'Login' : 'Create Account';
  }
}

function updateHeaderUser() {
  if (!currentUser) return;
  const initials = currentUser.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
  document.getElementById('hdr-avatar').textContent = initials;
  document.getElementById('hdr-name').textContent = currentUser.name;
  document.getElementById('hdr-user').style.display = 'flex';
  document.getElementById('hdr-login-btn').style.display = 'none';
  // Profile dropdown
  document.getElementById('pd-avatar').textContent = initials;
  document.getElementById('pd-name').textContent = currentUser.name;
  document.getElementById('pd-email').textContent = currentUser.email || currentUser.username;
  document.getElementById('pd-screens').textContent = `✦ ${results.length} resume${results.length!==1?'s':''} screened this session`;
}

function toggleProfile() {
  const dd = document.getElementById('profile-dropdown');
  // Update screen count live
  document.getElementById('pd-screens').textContent = `✦ ${results.length} resume${results.length!==1?'s':''} screened this session`;
  dd.style.display = dd.style.display === 'block' ? 'none' : 'block';
}

function logout() {
  currentUser = null;
  document.getElementById('hdr-user').style.display = 'none';
  document.getElementById('hdr-login-btn').style.display = 'block';
  document.getElementById('profile-dropdown').style.display = 'none';
}

// Close profile dropdown when clicking outside
document.addEventListener('click', e => {
  const dd = document.getElementById('profile-dropdown');
  const av = document.getElementById('hdr-avatar');
  const nm = document.getElementById('hdr-name');
  if (dd && !dd.contains(e.target) && e.target !== av && e.target !== nm) {
    dd.style.display = 'none';
  }
  const modal = document.getElementById('auth-modal');
  if (modal && e.target === modal) closeAuthModal();
});

setAuto(true);
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML

# ── PDF / File Upload endpoint ────────────────────────────────
from fastapi import UploadFile, File as FastFile, Form

@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = FastFile(...)):
    """Accept PDF or TXT resume, return extracted text."""
    content = await file.read()
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

if ext == "pdf":
        try:
            import fitz
            import io
            doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            return {"text": text.strip(), "filename": filename}
        except ImportError:
            try:
                from pdfminer.high_level import extract_text as pdf_extract
                import io
                text = pdf_extract(io.BytesIO(content))
                return {"text": text.strip(), "filename": filename}
            except Exception as e:
                raise HTTPException(400, f"Could not read PDF: {e}")
    elif ext in ("txt", "md"):
        try:
            text = content.decode("utf-8", errors="ignore")
            return {"text": text.strip(), "filename": filename}
        except Exception as e:
            raise HTTPException(400, f"Could not read file: {e}")
    else:
        raise HTTPException(400, "Unsupported file type. Please upload a PDF or TXT file.")


# ── User Auth (simple in-memory store) ───────────────────────
from fastapi.responses import JSONResponse

_users = {}   # { username: { password, name, email, history: [] } }

class SignupReq(BaseModel):
    name: str
    email: str
    username: str
    password: str

class LoginReq(BaseModel):
    username: str
    password: str

@app.post("/api/signup")
async def signup(req: SignupReq):
    if req.username in _users:
        raise HTTPException(400, "Username already taken")
    _users[req.username] = {"name": req.name, "email": req.email, "password": req.password, "history": []}
    logger.info("user.signup", {"username": req.username})
    return {"ok": True, "name": req.name, "username": req.username}

@app.post("/api/login")
async def login(req: LoginReq):
    user = _users.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(401, "Invalid username or password")
    logger.info("user.login", {"username": req.username})
    return {"ok": True, "name": user["name"], "username": req.username, "email": user["email"]}

@app.get("/api/profile/{username}")
async def get_profile(username: str):
    user = _users.get(username)
    if not user:
        raise HTTPException(404, "User not found")
    return {"name": user["name"], "email": user["email"], "username": username, "screens": len(user.get("history", []))}


if __name__ == "__main__":
    print()
    print("  ✦  RecruitAI Pro — Powered by Groq (FREE)")
    print("  ──────────────────────────────────────────")
    print("  ✓  Ultra fast — Llama 3.3 70B via Groq")
    print("  ✓  100% free — no credit card needed")
    print("  ✓  Opening browser automatically...")
    print("  →  http://localhost:8000")
    print("  →  Press Ctrl+C to stop")
    print()

    # Auto-open browser after 1.5s (gives server time to start)
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
