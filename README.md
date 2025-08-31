# Brand Compliance Checker

Automatically extracts brand-specific marketing guidelines, groups them and checks marketing image compliance. Offers compliance score, with transparent reasoning.

- **Inputs:** a Brand Guidelines **PDF** + a **candidate image** (PNG/JPG).
- **Pipeline:** extract text → classify guidelines (font/safe zone/colors/palette) → export “About the logo” reference pages → vision check of the candidate → scoring.
- **Interfaces:**
  - **FastAPI** backend (`/analyze`)
  - Optional **Streamlit** drag‑and‑drop UI

> ⚠️ Set `OPENAI_API_KEY` as an environment variable. 

---

## Architecture

```
logic.py        # Core pipeline (extraction, classification, vision eval, scoring)
main.py         # FastAPI endpoints
ui_streamlit.py # Streamlit UI (optional)
data/           # Uploaded files + generated outputs (mounted in Docker)
```

High‑level flow:

1. Extract PDF text (PyMuPDF).
2. LLM (text) groups concrete rules into categories:
   - Font style
   - Logo safe zone
   - Logo colours
   - Colour palette (overall image)
3. Export “About the logo” pages from the PDF for visual reference.
4. LLM evaluates candidate image based on extracted guidelines.
5. Compute per‑category and total scores; write JSON outputs.

---

## Requirements

- **Python 3.11+** (or Docker)
- `OPENAI_API_KEY` environment variable
- Packages: `fastapi`, `uvicorn[standard]`, `PyMuPDF`, `openai`, `python-multipart` (+ `streamlit` if using the UI)

---

## Quickstart (local dev)

```powershell
# create & activate env
conda create -n maria_env python=3.11 -y
conda activate maria_env

# install deps
pip install fastapi "uvicorn[standard]" PyMuPDF openai python-multipart streamlit

# set your key for this session (PowerShell)
$env:OPENAI_API_KEY = "sk-...YOUR_KEY..."

# run API
uvicorn main:app --reload
# → http://localhost:8000/docs

# (optional) run UI in another terminal
streamlit run ui_streamlit.py
# → http://localhost:8501
```

---

## Docker

### Build API image
```powershell
docker build -t brand-api .
```

### Run
```powershell
# .env should contain: OPENAI_API_KEY=sk-...
mkdir data  # once
docker run --rm -p 8000:8000 `
  --env-file .env `
  -v "$PWD\data:/app/data" `
  --name brand_api brand-api
# → http://localhost:8000/docs
```

### Compose with UI (optional)

`docker-compose.yml`:
```yaml
services:
  api:
    build: .
    container_name: brand_api
    env_file: .env
    ports: ["8000:8000"]
    volumes: ["./data:/app/data"]

  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    container_name: brand_ui
    env_file: .env   # only if the UI imports OpenAI directly
    ports: ["8501:8501"]
    volumes: ["./data:/app/data"]
    depends_on: ["api"]
```

Run:
```bash
docker compose up --build
# UI → http://localhost:8501
# API → http://localhost:8000/docs
```

---

## Environment variables

| Name             | Required | Description                              |
|------------------|----------|------------------------------------------|
| `OPENAI_API_KEY` | yes      | OpenAI key for text/vision calls         |
| `API_BASE`       | no       | UI only; API URL (default localhost:8000; in compose use `http://api:8000`) |


---

## API Reference

### `POST /analyze`
Upload a brand PDF and a candidate image. Returns grouped guidelines, per‑guideline decisions, and scores.  
**Content-Type:** `multipart/form-data`

**Form fields**
- `pdf` – Brand guidelines PDF
- `image` – Candidate image (png/jpg/jpeg)

**200 Response (shape)**

```json
{
  "grouped_path": "data/outputs/guidelines_grouped.json",
  "compliance_path": "data/outputs/candidate_compliance.json",
  "score_path": "data/outputs/candidate_score.json",
  "grouped": { "...": "..." },
  "compliance": { "...": "..." },
  "score": {
    "per_category": {
      "Font style": {"yes": 1, "total": 1, "score": 1.0},
      "Logo safe zone": {"yes": 0, "total": 1, "score": 0.0},
      "Logo colours": {"yes": 0, "total": 0, "score": 0.0},
      "Colour palette (overall image)": {"yes": 0, "total": 0, "score": 0.0}
    },
    "total_score": 1.0,
    "max_total": 4,
    "normalized_0_100": 25.0
  }
}
```


**PowerShell**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/analyze" -Method Post -Form @{
  pdf   = Get-Item ".\Neurons_brand_kit.pdf"
  image = Get-Item ".\neurons_2.png"
}
```

---

## Streamlit UI

Run locally:
```powershell
streamlit run ui_streamlit.py
# http://localhost:8501
```

Theming with `.streamlit/config.toml`:
```toml
[theme]
base = "dark"
primaryColor = "#7c3aed"
backgroundColor = "#0b1221"
secondaryBackgroundColor = "#111827"
textColor = "#e5e7eb"
```

If the UI calls the API, set in `ui_streamlit.py`:
```python
import os
API_BASE = os.getenv("API_BASE", "http://localhost:8000")  # in docker-compose use http://api:8000
```

---

## Outputs

Generated under `data/`:
- `guidelines_grouped.json` – extracted + categorized rules
- `<image>_compliance.json` – per‑guideline yes/no decisions
- `<image>_score.json` – category + total scores
- `about_logo_page_*.png` – reference pages exported from the PDF

Mount `./data:/app/data` in Docker so outputs persist on your host.

---
