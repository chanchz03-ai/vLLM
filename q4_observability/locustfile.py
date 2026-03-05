"""
q4_observability/locustfile.py
───────────────────────────────
Q4 — Locust load test for production server.

Run:
    cd llm_streaming
    locust -f q4_observability/locustfile.py --host http://localhost:8000 \
           -u 20 -r 2 --run-time 2m --headless
    
    # Or open UI:
    locust -f q4_observability/locustfile.py --host http://localhost:8000
"""
import json, random, time
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner


SHORT_PROMPTS = [
    "What is the capital of Germany?",
    "How many bytes in a kilobyte?",
    "Who invented the telephone?",
    "What does HTTP stand for?",
]

MEDIUM_PROMPTS = [
    "Explain the difference between a list and a tuple in Python.",
    "What are the main principles of REST API design?",
    "How does a hash table work?",
    "What is the CAP theorem?",
]

LONG_PROMPTS = [
    "Explain how neural network backpropagation works, including the chain rule of calculus.",
    "Write a Python implementation of a thread-safe queue with a timeout parameter.",
    "Compare microservices vs monolithic architecture with pros and cons of each.",
]

SYSTEM_PROMPTS = [
    "You are a concise technical assistant. Keep answers under 100 words.",
    "You are an expert software engineer. Provide precise, idiomatic answers.",
    None,
]


class LLMStreamUser(HttpUser):
    """Simulates a real user making streaming chat requests."""
    wait_time = between(2, 8)   # Think time between requests

    @task(5)
    def short_query(self):
        """Most common: short, factual queries."""
        self._stream_request(
            messages=[{"role": "user", "content": random.choice(SHORT_PROMPTS)}],
            system_prompt=random.choice(SYSTEM_PROMPTS),
            max_tokens=150,
        )

    @task(3)
    def medium_query(self):
        """Medium complexity queries."""
        self._stream_request(
            messages=[{"role": "user", "content": random.choice(MEDIUM_PROMPTS)}],
            system_prompt=SYSTEM_PROMPTS[0],    # Repeated → tests prefix cache
            max_tokens=300,
        )

    @task(2)
    def long_query(self):
        """Complex, long-output queries."""
        self._stream_request(
            messages=[{"role": "user", "content": random.choice(LONG_PROMPTS)}],
            max_tokens=600,
        )

    @task(1)
    def structured_query(self):
        """Structured JSON output request."""
        self._stream_request(
            messages=[{"role": "user", "content": "List the top 5 programming languages in 2024 with their primary use cases."}],
            requires_json=True,
            max_tokens=400,
        )

    def _stream_request(
        self,
        messages: list,
        system_prompt: str | None = None,
        max_tokens: int = 256,
        requires_json: bool = False,
    ):
        payload = {
            "messages":      messages,
            "system_prompt": system_prompt,
            "max_tokens":    max_tokens,
            "temperature":   0.7,
            "requires_json": requires_json,
        }

        start = time.perf_counter()
        first_token = None
        token_count = 0

        with self.client.post(
            "/v1/chat/stream",
            json=payload,
            stream=True,
            catch_response=True,
            timeout=60,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return

            try:
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:].strip())

                    if data.get("event") == "ttft" and first_token is None:
                        first_token = time.perf_counter()

                    elif data.get("event") == "token":
                        token_count += 1

                    elif data.get("event") == "done":
                        resp.success()
                        return

                    elif data.get("event") == "error":
                        resp.failure(data.get("msg", "unknown error"))
                        return

            except Exception as e:
                resp.failure(str(e))


class LLMBurstUser(HttpUser):
    """Simulates burst traffic — no think time."""
    wait_time = between(0.1, 0.5)
    weight = 1   # 1 burst user for every 5 normal users

    @task
    def burst_short(self):
        payload = {
            "messages": [{"role": "user", "content": random.choice(SHORT_PROMPTS)}],
            "max_tokens": 100,
        }
        with self.client.post("/v1/chat/stream", json=payload, stream=True, catch_response=True, timeout=30) as resp:
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line and b"done" in line:
                        resp.success()
                        return
            else:
                resp.failure(f"HTTP {resp.status_code}")