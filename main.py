from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid

from logic import run_brand_compliance

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Brand Compliance API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded and output files
app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")

@app.post("/analyze")
async def analyze(pdf: UploadFile = File(...), image: UploadFile = File(...)):
    """
    Upload a brand PDF and a candidate image.
    Returns grouped guidelines, compliance, and score.
    """

    # save uploads
    pdf_path = DATA_DIR / f"{uuid.uuid4()}_{pdf.filename}"
    img_path = DATA_DIR / f"{uuid.uuid4()}_{image.filename}"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)
    with open(img_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        result = run_brand_compliance(pdf_path, img_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "grouped": result["grouped"],
        "compliance": result["compliance"],
        "score": result["score"],
        "files": {
            "grouped": result["grouped_path"],
            "compliance": result["compliance_path"],
            "score": result["score_path"],
        },
    }
