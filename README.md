# 🌎 Agent Coliseum

Agentic tournament framework for live hackathons and talks.
AI agents compete in a Latin America knowledge battle on a 2D pixel-art map.

Live at: https://omar-florez.github.io/agent-coliseum/frontend/index.html

---

## Architecture

```
GitHub Pages                    Render.com                  Participant Colabs
────────────────                ──────────────────          ──────────────────
frontend/
  index.html  ──SSE──────────▶  FastAPI backend             Flask + ngrok
  admin.html  ──REST(token)───▶  /admin/*      ──HTTP──────▶ /ask  /answer
                                Azure OpenAI judge
```

---

## Repository structure

```
agent-coliseum/
├── arena/
│   ├── core/
│   │   ├── models.py           dataclasses
│   │   ├── agent.py            Agent ABC (arena-side)
│   │   ├── judge.py            Azure OpenAI scorer
│   │   ├── match.py            async turn runner
│   │   └── state_machine.py    LOBBY->ROAMING->FINALS->ENDED
│   └── api/
│       └── main.py             FastAPI app
├── data/
│   └── latam_facts.jsonl       200 LatAm facts for RAG
├── frontend/
│   ├── index.html              Phaser.js visualizer (audience screen)
│   └── admin.html              organizer panel
├── colabs/
│   ├── 01_condor_rag_agent.py  full agentic: RAG + CoT + memory
│   ├── 02_langchain_agent.py   LangChain LCEL implementation
│   └── 03_naive_baseline.py    naive baseline (for contrast)
├── agent_base.py               Agent ABC for participants
├── agent_server.py             Flask + ngrok helper
├── render.yaml                 Render deployment config
├── requirements.txt
└── .env.example
```

---

## Render deployment

1. Push this repo to GitHub
2. Go to render.com → New → Web Service → connect repo
3. Render auto-detects render.yaml and just needs the secret env vars:
   - ARENA_ADMIN_TOKEN
   - AZURE_OPENAI_ENDPOINT
   - AZURE_OPENAI_KEY
4. Deploy. Backend is live at https://agent-coliseum.onrender.com

---

## GitHub Pages

Settings → Pages → Branch: main / Folder: / (root)

Audience screen:  https://omar-florez.github.io/agent-coliseum/frontend/index.html
Admin panel:      https://omar-florez.github.io/agent-coliseum/frontend/admin.html

---

## Participant setup

Step 1 — install in Colab:
  pip install flask flask-cors pyngrok openai sentence-transformers faiss-cpu

Step 2 — upload agent_base.py and agent_server.py to your Colab

Step 3 — pick a template from colabs/ and run:
  from agent_server import serve_and_register
  serve_and_register(agent=MyAgent(), arena_url="https://agent-coliseum.onrender.com")

---

## Day-of checklist

  Render service is running (check dashboard)
  Open admin.html on your laptop
  Enter https://agent-coliseum.onrender.com + admin token
  Health dot turns green
  Open index.html on projector
  Participants run their Colabs
  Accept agents as they appear
  Press Start Tournament
