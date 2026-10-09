"""
خادم وسيط (Gateway) بين صفحة VR ونموذج Claude.
الغرض: المفتاح يبقى على الخادم، تحديد معدل الطلبات، سقف تكلفة يومي، سجل لكل استدعاء.

حالة هذا الملف: كُتب ودُقّق بناءً (py_compile) فقط. لم يُشغَّل مع fastapi ولا مع Anthropic API فعلياً.
قبل الإنتاج: استبدل الذاكرة المحلية بـ Redis، وSQLite بـ Postgres (جدول ai_calls في interior_vr_schema.sql).
"""
import hashlib, json, os, sqlite3, time
from collections import defaultdict, deque

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hub import Hub

API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]          # لا يُكتب في الكود
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "24000"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2000"))
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "10"))
DAILY_CALL_CAP = int(os.environ.get("DAILY_CALL_CAP", "300"))
# الأسعار لا أخمّنها: إن لم تُضبط تُحسب الحدود بعدد الطلبات فقط.
PRICE_IN = os.environ.get("USD_PER_MTOK_IN")
PRICE_OUT = os.environ.get("USD_PER_MTOK_OUT")
DAILY_USD_CAP = os.environ.get("DAILY_USD_CAP")
# org_id:sha256(api_key) مفصولة بفواصل، مثال: emad:ab12...
ORG_KEYS = dict(x.split(":", 1) for x in os.environ.get("ORG_KEY_HASHES", "").split(",") if ":" in x)

db = sqlite3.connect(os.environ.get("DB_PATH", "gateway.db"), check_same_thread=False)
db.execute("""CREATE TABLE IF NOT EXISTS ai_calls(
  id INTEGER PRIMARY KEY, org TEXT, ts REAL, ok INTEGER, ms INTEGER,
  in_tok INTEGER, out_tok INTEGER, usd REAL, err TEXT)""")
db.commit()
hits: dict[str, deque] = defaultdict(deque)

app = FastAPI(title="VR Interior Gateway")
# الصفحة تُستضاف على نطاق آخر، فلا بد من السماح لنطاقها فقط (لا تضع * في الإنتاج)
app.add_middleware(CORSMiddleware,
    allow_origins=[o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o],
    allow_methods=["POST", "GET"], allow_headers=["Content-Type", "X-Org-Key"])


def org_from_key(x_org_key: str = Header(...)) -> str:
    h = hashlib.sha256(x_org_key.encode()).hexdigest()
    for org, want in ORG_KEYS.items():
        if want == h:
            return org
    raise HTTPException(401, "مفتاح غير صالح")


def check_limits(org: str) -> None:
    now = time.time()
    q = hits[org]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        raise HTTPException(429, "طلبات كثيرة، انتظر قليلاً")
    day = now - 86400
    n, usd = db.execute("SELECT COUNT(*), COALESCE(SUM(usd),0) FROM ai_calls WHERE org=? AND ts>?", (org, day)).fetchone()
    if n >= DAILY_CALL_CAP:
        raise HTTPException(429, "تجاوزت الحد اليومي")
    if DAILY_USD_CAP and usd >= float(DAILY_USD_CAP):
        raise HTTPException(429, "تجاوزت سقف التكلفة اليومي")
    q.append(now)


class Ask(BaseModel):
    prompt: str = Field(min_length=1)
    system: str | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ai/json")
async def ai_json(body: Ask, org: str = Depends(org_from_key)):
    if len(body.prompt) + len(body.system or "") > MAX_PROMPT_CHARS:
        raise HTTPException(413, "الطلب كبير جداً")
    check_limits(org)
    t0 = time.time()
    payload = {"model": MODEL, "max_tokens": MAX_TOKENS,
               "messages": [{"role": "user", "content": body.prompt}]}
    if body.system:
        payload["system"] = body.system
    err, data = None, None
    try:
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.post(API_URL, json=payload, headers={
                "x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        if r.status_code != 200:
            try:
                et = r.json().get("error", {}).get("type", "")
            except Exception:
                et = ""
            err = f"upstream {r.status_code} {et}".strip()
        else:
            data = r.json()
    except httpx.HTTPError as e:
        err = type(e).__name__
    ms = int((time.time() - t0) * 1000)
    use = (data or {}).get("usage", {})
    tin, tout = use.get("input_tokens", 0), use.get("output_tokens", 0)
    usd = (tin * float(PRICE_IN) + tout * float(PRICE_OUT)) / 1e6 if PRICE_IN and PRICE_OUT else 0.0
    db.execute("INSERT INTO ai_calls(org,ts,ok,ms,in_tok,out_tok,usd,err) VALUES(?,?,?,?,?,?,?,?)",
               (org, time.time(), 0 if err else 1, ms, tin, tout, usd, err))
    db.commit()
    if err:
        raise HTTPException(502, f"تعذر الحصول على رد من النموذج ({err})")
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return {"json": json.loads(text)}
    except json.JSONDecodeError:
        raise HTTPException(422, "الرد ليس JSON صالحاً")


# ================= الجلسة المشتركة (مصمم على التابلت + عميل بالنظارة) =================
hub = Hub()
ALLOWED = [o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o]


@app.post("/session/new")
def session_new(org: str = Depends(org_from_key)):
    """المصمم يبدأ جلسة: يحصل على رمزين. رمز العميل فقط هو الذي يُشارَك (لا مفتاح المنظمة)."""
    return hub.create(org)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, room: str, token: str):
    origin = ws.headers.get("origin")
    if ALLOWED and origin not in ALLOWED:
        await ws.close(code=4403)
        return
    await ws.accept()
    try:
        await hub.connect(room, token, ws)
    except PermissionError as e:
        await ws.send_text(json.dumps({"type": "error", "msg": str(e)}, ensure_ascii=False))
        await ws.close(code=4401)
        return
    try:
        while True:
            err = await hub.handle(room, ws, await ws.receive_text())
            if err:
                await ws.send_text(json.dumps({"type": "error", "msg": err}, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(room, ws)
