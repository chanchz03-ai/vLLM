
"""
q1_foundations/server.py
─────────────────────────
Q1 — FastAPI server with SSE streaming endpoint + static chat UI.

Run:
    cd llm_streaming
    uvicorn q1_foundations.server:app --reload --port 8000
"""
import sys, json, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared.config import cfg
from shared.metrics import start_metrics_server
from q1_foundations.streaming import stream_chat

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
cfg.validate()

app = FastAPI(
    title="LLM Streaming — Q1 Foundations",
    description="Streaming chat API powered by Groq",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    system_prompt: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": cfg.MODEL_FAST}


@app.post("/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE streaming endpoint.
    Each data: line is a JSON object with an 'event' field.
    """
    messages = [m.model_dump() for m in req.messages]
    return StreamingResponse(
        stream_chat(
            messages=messages,
            model=req.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            system_prompt=req.system_prompt,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering
        },
    )


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    """Serve a minimal streaming chat interface."""
    return HTMLResponse(content=CHAT_HTML)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    start_metrics_server()
    print(f"\n🚀  Q1 Streaming Server running → http://{cfg.HOST}:{cfg.PORT}")
    print(f"📖  API docs                    → http://{cfg.HOST}:{cfg.PORT}/docs\n")


# ── Embedded Chat UI ──────────────────────────────────────────────────────────
CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>LLM Streaming — Q1</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }
    header { background: #1e293b; padding: 16px 24px; border-bottom: 1px solid #334155; display: flex; align-items: center; gap: 12px; }
    header h1 { font-size: 1.1rem; font-weight: 600; color: #60a5fa; }
    .badge { background: #0ea5e9; color: white; font-size: 0.7rem; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
    #metrics-bar { background: #1e293b; padding: 8px 24px; border-bottom: 1px solid #1e293b; display: flex; gap: 24px; font-size: 0.78rem; color: #94a3b8; }
    .metric { display: flex; align-items: center; gap: 6px; }
    .metric .val { color: #34d399; font-weight: 700; font-family: monospace; font-size: 0.85rem; }
    #messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
    .msg { max-width: 80%; border-radius: 12px; padding: 12px 16px; line-height: 1.6; font-size: 0.92rem; }
    .msg.user { background: #1d4ed8; color: white; align-self: flex-end; border-bottom-right-radius: 4px; }
    .msg.assistant { background: #1e293b; color: #e2e8f0; align-self: flex-start; border-bottom-left-radius: 4px; border: 1px solid #334155; }
    .msg.assistant.streaming::after { content: '▋'; animation: blink 0.7s infinite; color: #60a5fa; }
    @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }
    #input-area { background: #1e293b; border-top: 1px solid #334155; padding: 16px 24px; display: flex; gap: 12px; }
    #prompt { flex: 1; background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; color: #e2e8f0; font-size: 0.92rem; resize: none; outline: none; }
    #prompt:focus { border-color: #3b82f6; }
    #send-btn { background: #2563eb; color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 0.92rem; font-weight: 600; cursor: pointer; transition: background .2s; }
    #send-btn:hover { background: #1d4ed8; }
    #send-btn:disabled { background: #334155; cursor: not-allowed; }
    #model-select { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; color: #e2e8f0; font-size: 0.85rem; }
  </style>
</head>
<body>
  <header>
    <h1>⚡ LLM Streaming — Q1 Foundations</h1>
    <span class="badge">Groq</span>
    <span class="badge" style="background:#7c3aed">SSE</span>
  </header>

  <div id="metrics-bar">
    <div class="metric">TTFT <span class="val" id="m-ttft">—</span></div>
    <div class="metric">Tokens/s <span class="val" id="m-tps">—</span></div>
    <div class="metric">Total time <span class="val" id="m-total">—</span></div>
    <div class="metric">Tokens out <span class="val" id="m-tokens">—</span></div>
  </div>

  <div id="messages">
    <div class="msg assistant">👋 Hello! I'm powered by Groq streaming. Ask me anything — watch your first token appear in milliseconds.</div>
  </div>

  <div id="input-area">
    <select id="model-select">
      <option value="llama3-8b-8192">Llama 3 8B (fast)</option>
      <option value="llama-3.3-70b-versatile">Llama 3.3 70B (capable)</option>
      <option value="mixtral-8x7b-32768">Mixtral 8x7B</option>
      <option value="gemma2-9b-it">Gemma 2 9B</option>
    </select>
    <textarea id="prompt" rows="1" placeholder="Type a message… (Enter to send, Shift+Enter for newline)"></textarea>
    <button id="send-btn">Send ➤</button>
  </div>

  <script>
    const messagesEl = document.getElementById('messages');
    const promptEl   = document.getElementById('prompt');
    const sendBtn    = document.getElementById('send-btn');
    const modelEl    = document.getElementById('model-select');
    const history    = [];

    function addMessage(role, text = '', streaming = false) {
      const div = document.createElement('div');
      div.className = `msg ${role}${streaming ? ' streaming' : ''}`;
      div.textContent = text;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return div;
    }

    function setMetric(id, val) {
      document.getElementById(id).textContent = val;
    }

    async function sendMessage() {
      const text = promptEl.value.trim();
      if (!text) return;
      promptEl.value = '';
      sendBtn.disabled = true;

      history.push({ role: 'user', content: text });
      addMessage('user', text);

      const assistantDiv = addMessage('assistant', '', true);
      let fullText = '';

      // Reset metrics
      ['m-ttft','m-tps','m-total','m-tokens'].forEach(id => setMetric(id, '…'));

      try {
        const resp = await fetch('/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: history, model: modelEl.value }),
        });

        const reader  = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop(); // keep incomplete line

          for (const line of lines) {
            if (!line.startsWith('data:')) continue;
            const payload = JSON.parse(line.slice(5).trim());

            if (payload.event === 'ttft')  setMetric('m-ttft',   payload.ms + ' ms');
            if (payload.event === 'token') {
              fullText += payload.text;
              assistantDiv.textContent = fullText;
              messagesEl.scrollTop = messagesEl.scrollHeight;
            }
            if (payload.event === 'done') {
              setMetric('m-tps',    payload.tps + ' tok/s');
              setMetric('m-total',  payload.total_ms + ' ms');
              setMetric('m-tokens', payload.tokens + ' tok');
              assistantDiv.classList.remove('streaming');
            }
            if (payload.event === 'error') {
              assistantDiv.textContent = '❌ ' + payload.msg;
              assistantDiv.classList.remove('streaming');
            }
          }
        }

        history.push({ role: 'assistant', content: fullText });

      } catch (e) {
        assistantDiv.textContent = '❌ Connection error: ' + e.message;
        assistantDiv.classList.remove('streaming');
      } finally {
        sendBtn.disabled = false;
        promptEl.focus();
      }
    }

    sendBtn.addEventListener('click', sendMessage);
    promptEl.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("q1_foundations.server:app", host=cfg.HOST, port=cfg.PORT, reload=True)