import os
import asyncio
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="MULTI-BRAIN AI")
app.mount("/static", StaticFiles(directory="static"), name="static")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY", "").strip()
CUSTOM_API_BASE_URL = os.getenv("CUSTOM_API_BASE_URL", "").strip().rstrip("/")
CUSTOM_API_MODEL = os.getenv("CUSTOM_API_MODEL", "").strip()

TIMEOUT = httpx.Timeout(75.0, connect=15.0)


class DebateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=6000)


def require_config() -> None:
    missing = []
    for name, value in [
        ("GEMINI_API_KEY", GEMINI_API_KEY),
        ("GEMINI_MODEL", GEMINI_MODEL),
        ("CUSTOM_API_KEY", CUSTOM_API_KEY),
        ("CUSTOM_API_BASE_URL", CUSTOM_API_BASE_URL),
        ("CUSTOM_API_MODEL", CUSTOM_API_MODEL),
    ]:
        if not value:
            missing.append(name)
    if missing:
        raise HTTPException(
            status_code=503,
            detail="Server setup is incomplete. Missing environment variables: "
                   + ", ".join(missing),
        )


async def gemini_generate(client: httpx.AsyncClient, prompt: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    response = await client.post(
        url,
        params={"key": GEMINI_API_KEY},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.6, "maxOutputTokens": 1200},
        },
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("Gemini returned an empty or unexpected response.")


async def custom_generate(client: httpx.AsyncClient, prompt: str) -> str:
    url = f"{CUSTOM_API_BASE_URL}/chat/completions"
    response = await client.post(
        url,
        headers={
            "Authorization": f"Bearer {CUSTOM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": CUSTOM_API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "max_tokens": 1200,
        },
    )
    response.raise_for_status()
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "") for part in content
                if isinstance(part, dict)
            )
        return str(content).strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("Custom API returned an empty or unexpected response.")


async def safe_call(label: str, call) -> dict[str, Any]:
    try:
        return {"provider": label, "text": await call, "error": None}
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        body = exc.response.text[:500]
        return {
            "provider": label,
            "text": "",
            "error": f"Provider HTTP {status}: {body}",
        }
    except Exception as exc:
        return {"provider": label, "text": "", "error": str(exc)[:500]}


@app.get("/")
async def home():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {
        "app": "MULTI-BRAIN AI",
        "gemini_configured": bool(GEMINI_API_KEY and GEMINI_MODEL),
        "custom_configured": bool(
            CUSTOM_API_KEY and CUSTOM_API_BASE_URL and CUSTOM_API_MODEL
        ),
    }


@app.post("/api/debate")
async def debate(body: DebateRequest):
    require_config()
    user_prompt = body.prompt.strip()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Round 1: both models answer independently and concurrently.
        first_prompt = (
            "You are an independent analyst. Answer the user's question clearly, "
            "state important assumptions, and mention uncertainty. Do not invent facts.\n\n"
            f"USER QUESTION:\n{user_prompt}"
        )
        first_gemini, first_custom = await asyncio.gather(
            safe_call("Gemini", gemini_generate(client, first_prompt)),
            safe_call("Custom API", custom_generate(client, first_prompt)),
        )

        if first_gemini["error"] and first_custom["error"]:
            raise HTTPException(
                status_code=502,
                detail="Both providers failed. Check the server logs and API settings.",
            )

        # Round 2: each model reviews the other model's answer.
        critique_tasks = []
        if not first_gemini["error"] and not first_custom["error"]:
            critique_tasks = [
                safe_call(
                    "Gemini critique",
                    gemini_generate(
                        client,
                        "Review the other AI's answer. Identify valid points, "
                        "possible errors, missing context, and unsupported claims. "
                        "Be fair and concise.\n\nQUESTION:\n"
                        + user_prompt
                        + "\n\nOTHER AI ANSWER:\n"
                        + first_custom["text"],
                    ),
                ),
                safe_call(
                    "Custom API critique",
                    custom_generate(
                        client,
                        "Review the other AI's answer. Identify valid points, "
                        "possible errors, missing context, and unsupported claims. "
                        "Be fair and concise.\n\nQUESTION:\n"
                        + user_prompt
                        + "\n\nOTHER AI ANSWER:\n"
                        + first_gemini["text"],
                    ),
                ),
            ]
        critiques = await asyncio.gather(*critique_tasks) if critique_tasks else []

        # Round 3: synthesize only from successful, visible debate material.
        pieces = [
            f"QUESTION:\n{user_prompt}",
            f"GEMINI INITIAL ANSWER:\n{first_gemini['text'] or '[Unavailable: ' + str(first_gemini['error']) + ']'}",
            f"CUSTOM API INITIAL ANSWER:\n{first_custom['text'] or '[Unavailable: ' + str(first_custom['error']) + ']'}",
        ]
        for item in critiques:
            pieces.append(
                f"{item['provider'].upper()}:\n"
                + (item["text"] if not item["error"] else "[Unavailable: " + item["error"] + "]")
            )

        synthesis_prompt = (
            "You are the neutral moderator of an AI debate. Produce a useful final answer "
            "based only on the question and debate material below. Separate areas of agreement "
            "from disagreements, correct unsupported claims where possible, and clearly say "
            "when something remains uncertain. Do not pretend that either model is always right.\n\n"
            + "\n\n".join(pieces)
        )

        if not first_gemini["error"]:
            final_result = await safe_call(
                "Final synthesis (Gemini)",
                gemini_generate(client, synthesis_prompt),
            )
        else:
            final_result = await safe_call(
                "Final synthesis (Custom API)",
                custom_generate(client, synthesis_prompt),
            )

    return {
        "initial_answers": [first_gemini, first_custom],
        "critiques": critiques,
        "final": final_result,
    }
