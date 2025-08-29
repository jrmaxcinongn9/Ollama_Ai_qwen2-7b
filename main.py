# main.py
import os, json, re
import requests
from typing import List, Literal, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL   = os.getenv("MODEL_ID", "qwen2:7b")

app = FastAPI(title="Qwen2:7b (Ollama) Service", version="1.0.0")

# CORS (เปิดกว้างตอน dev; โปรดจำกัด origin ในโปรดักชัน)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- System prompts ----------
SYSTEM_THAI = (
    "คุณคือระบบวิเคราะห์ข้อความลูกค้า ตอบกลับเป็น JSON เดียวเท่านั้น "
    "รูปแบบ: {\"summary\":\"...\",\"category\":\"general|complaint|request|other\","
    "\"urgency\":\"low|medium|high\",\"language\":\"th|en\"} "
    "กติกา: ให้เขียน summary เป็น 'ภาษาไทย' และกำหนด language เป็น 'th' เท่านั้น "
    "ห้ามใส่ข้อความอื่นนอกเหนือจาก JSON เดียว"
)

SYSTEM_EN = (
    "You are a text analysis system. Reply with ONE JSON object only: "
    "{\"summary\":\"...\",\"category\":\"general|complaint|request|other\","
    "\"urgency\":\"low|medium|high\",\"language\":\"th|en\"} "
    "Rules: write the summary in 'English' and set language to 'en' only. "
    "No extra text outside the single JSON."
)

SYSTEM_AUTO = (
     "You are a multilingual text analysis system. Detect the language automatically. "
    "Reply with ONE JSON object only: "
    "{\"summary\":\"...\",\"category\":\"general|complaint|request|other\","
    "\"urgency\":\"low|medium|high\",\"language\":\"xx\"} "
    "Rules: (1) Set language as ISO 639-1 code (e.g. th, en, ja, zh, fr, de, es); "
    "(2) Write the summary in the same language as the input; "
    "(3) Output only the single JSON, no extra text."
)

# ---------- Schemas ----------
Role = Literal["system", "user", "assistant"]

class ChatMessage(BaseModel):
    role: Role
    content: str

class ChatReq(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = 0.2
    options: Optional[Dict[str, Any]] = None  # ผ่าน options ให้ Ollama ได้

class ChatResp(BaseModel):
    model: str
    reply: str
    stats: Optional[Dict[str, Any]] = None

class AnalyzeReq(BaseModel):
    text: str
    language: Literal["th", "en", "auto", "multi"] = "auto"
    model: Optional[str] = None

class AnalyzeResult(BaseModel):
    summary: str
    category: Literal["general","complaint","request","other"]
    urgency: Literal["low","medium","high"]
    language: str   # เปลี่ยนจาก Literal เป็น str เพื่อให้รองรับภาษาที่มากกว่า th|en

class AnalyzeResp(BaseModel):
    model: str
    result: AnalyzeResult

# ---------- Helpers ----------
def _post_ollama(path: str, payload: dict):
    url = f"{OLLAMA_BASE_URL}{path}"
    try:
        r = requests.post(url, json=payload, timeout=300)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {e}")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama error: {r.text}")
    return r.json()

def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text

def _pick_system_msg(lang: str) -> str:
    if lang == "th":
        return SYSTEM_THAI
    if lang == "en":
        return SYSTEM_EN
    if lang == "multi":
        return SYSTEM_AUTO
    return SYSTEM_AUTO

def _analyze_once(text: str, language: str, model: Optional[str]) -> AnalyzeResult:
    # validate language
    language = (language or "auto").lower()
    if language not in {"th", "en", "auto"}:
        raise HTTPException(status_code=422, detail="language must be one of: th | en | auto")

    mdl = model or DEFAULT_MODEL
    system_msg = _pick_system_msg(language)
    payload = {
        "model": mdl,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    data = _post_ollama("/api/chat", payload)
    raw = (data.get("message") or {}).get("content", "").strip()
    json_text = _extract_json(raw)
    try:
        obj = json.loads(json_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {e} | raw={raw[:200]}")
    try:
        return AnalyzeResult(**obj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON shape mismatch: {e} | raw={raw[:200]}")

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "ollama": OLLAMA_BASE_URL, "default_model": DEFAULT_MODEL}

@app.post("/v1/chat", response_model=ChatResp)
def chat(req: ChatReq):
    model = req.model or DEFAULT_MODEL
    payload = {
        "model": model,
        "messages": [m.model_dump() for m in req.messages],
        "stream": False,
        "options": req.options or {"temperature": req.temperature},
    }
    data = _post_ollama("/api/chat", payload)
    message = (data.get("message") or {}).get("content", "")
    stats = {
        "total_duration_ns": data.get("total_duration"),
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
    }
    return ChatResp(model=model, reply=message, stats=stats)

@app.post("/v1/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    result = _analyze_once(req.text, req.language, req.model)
    return AnalyzeResp(model=(req.model or DEFAULT_MODEL), result=result)

@app.get("/v1/analyze", response_model=AnalyzeResp)
def analyze_get(
    text: str = Query(..., min_length=1, description="ข้อความที่จะวิเคราะห์"),
    language: str = Query("auto", description="th | en | auto"),
    model: Optional[str] = Query(None, description="ระบุโมเดล ถ้าไม่ใส่จะใช้ค่าเริ่มต้น"),
):
    result = _analyze_once(text, language, model)
    return AnalyzeResp(model=(model or DEFAULT_MODEL), result=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4001, reload=True)
