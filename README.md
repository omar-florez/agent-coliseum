# 🌎 LatAm Arena

Agentic tournament framework for live hackathons and talks.
AI agents compete in a Latin America knowledge battle on a 2D pixel-art map.

---

## Architecture

```
GitHub Pages                Azure VM                    Participant Colabs
─────────────               ─────────────────           ──────────────────
frontend/                   FastAPI backend             Flask + ngrok
  index.html  ──SSE──────▶  /stream                     /ask
  admin.html  ──REST─────▶  /admin/*      ──HTTP──────▶ /answer
                            Azure OpenAI judge           /eliminated
```

Participants bring their own API keys. CENIA only pays for the judge.

---

## Azure VM Setup

```bash
# 1. Provision Standard_B2s (2 vCPU, 4 GB RAM) in Azure portal
#    Estimated cost: ~$0.04/hr during the event

# 2. SSH in and clone
git clone https://github.com/latam-gpt/latam-arena
cd latam-arena

# 3. Install
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
nano .env   # fill in tokens and Azure OpenAI credentials

# 5. Allow sudo shutdown without password (for admin panel Stop VM button)
echo "$USER ALL=(ALL) NOPASSWD: /sbin/shutdown" | sudo tee /etc/sudoers.d/arena

# 6. Run (port 8000, bind 0.0.0.0)
uvicorn arena.api.main:app --host 0.0.0.0 --port 8000
```

### Network security group rules (Azure portal)
| Port | Protocol | Source | Purpose        |
|------|----------|--------|----------------|
| 8000 | TCP      | Any    | Arena API + SSE |

---

## Organizer Panel (admin.html)

Open `frontend/admin.html` in your browser **on your laptop only**.

1. Enter the VM's public IP: `http://VM_IP:8000`
2. Enter the `ARENA_ADMIN_TOKEN` from `.env`
3. Click **Connect** — you'll see the health indicator turn green

**Before the talk:**
- Ping to confirm health
- Ask participants to register (they run their Colabs)
- Accept agents one by one as they appear in pending list

**During the talk:**
- Press **🚀 Start Tournament** when ready
- Use **💀 Eliminate** for manual overrides
- Use **⏹ End Tournament** after the winner is clear

**After the talk:**
- Press **🔴 Stop VM** — sends `sudo shutdown now` to the server

---

## Audience Screen (index.html)

Open `frontend/index.html` on the projector browser.

1. Enter the VM's public IP
2. Click **Connect** — the map will appear
3. Accepted agents spawn on the map
4. Matches auto-start when agents are adjacent
5. Color-coded dialogue bubbles show reasoning in real time

**Color coding:**
- 🟣 Purple — agent thinking (CoT scratchpad)
- 🟡 Amber — question asked
- 🔵 Teal — answer given
- 🟢 Green — judge score and reason

---

## Participant Setup (3 steps)

### Step 1: Install

```python
!pip install flask flask-cors pyngrok openai sentence-transformers faiss-cpu
```

### Step 2: Upload files to Colab

Upload these two files from this repo:
- `agent_base.py`   ← the Agent ABC
- `agent_server.py` ← the serve_and_register helper

### Step 3: Pick a Colab template

| File | Strategy | Difficulty |
|------|----------|-----------|
| `colabs/01_condor_rag_agent.py` | Full: RAG + CoT + memory + targeting | Advanced |
| `colabs/02_langchain_agent.py`  | LangChain LCEL + match memory | Intermediate |
| `colabs/03_naive_baseline.py`   | Naive: no memory, no RAG | Beginner |

Edit the config at the top (API key, arena URL, ngrok token) and run.
You'll see your agent appear in the admin panel pending list.

---

## Repository structure

```
latam-arena/
├── arena/
│   ├── core/
│   │   ├── models.py          ← dataclasses
│   │   ├── agent.py           ← Agent ABC (arena-side)
│   │   ├── judge.py           ← Azure OpenAI scoring
│   │   ├── match.py           ← turn runner
│   │   └── state_machine.py   ← LOBBY→ROAMING→FINALS→ENDED
│   └── api/
│       └── main.py            ← FastAPI app
├── data/
│   └── latam_facts.jsonl      ← 200 LatAm facts for RAG
├── frontend/
│   ├── index.html             ← Phaser.js visualizer (audience screen)
│   └── admin.html             ← organizer panel
├── colabs/
│   ├── 01_condor_rag_agent.py ← full agentic implementation
│   ├── 02_langchain_agent.py  ← LangChain implementation
│   └── 03_naive_baseline.py   ← naive baseline (for contrast)
├── agent_base.py              ← Agent ABC + dataclasses for participants
├── agent_server.py            ← Flask server + ngrok helper
├── requirements.txt
└── .env.example
```

---

## Talk narrative

| Phase | Audience sees | You say |
|-------|--------------|---------|
| LOBBY | Agents appearing on map as each Colab registers | "Each Colab is a different LLM strategy" |
| Start | Press 🚀 — agents begin walking | "Same base model, different architecture" |
| Round 1 | First match, thinking bubbles visible | "Watch what each agent thinks before answering" |
| Round 2 | RAG agent wins consistently | "That's why retrieval-augmented generation matters" |
| Finals | Two agents left | "Now it's personal" |
| Winner | Victory screen | "This is what agency looks like" |
