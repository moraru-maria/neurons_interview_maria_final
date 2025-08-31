from pathlib import Path
import fitz  # PyMuPDF
from typing import Dict, Any, List
from openai import OpenAI
import json
import re
import base64
import os

# ========== OpenAI client ==========

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ------------------  Text extractor  --------------------
def extract_guidelines_text(pdf_path: str | Path, out_text_path: str | Path | None = None) -> dict:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            txt = page.get_text("text").strip()
            if txt:
                pages.append(txt)

    full_text = "\n\n".join(pages)

    if out_text_path:
        out_text_path = Path(out_text_path)
        out_text_path.parent.mkdir(parents=True, exist_ok=True)
        out_text_path.write_text(full_text, encoding="utf-8")

    return {"pages": pages, "text": full_text}

# ----------   Save “About the Logo” pages as images, for reference ----------
def save_about_logo_pages(pdf_path, out_dir=None, zoom=2.5):
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir or pdf_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if "about the logo" in text.lower():
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                out_path = out_dir / f"about_logo_page_{i:02d}.png"
                pix.save(out_path)
                saved.append(str(out_path))
    return saved

CATEGORIES = [
    "Font style",
    "Logo safe zone",
    "Logo colours",
    "Colour palette (overall image)",
    "other",
]

SYSTEM_PROMPT = """You are a precise information extractor.
From the provided brand-guidelines text, extract concrete guideline statements and group them into EXACTLY these categories:
- Font style
- Logo safe zone
- Logo colours
- Colour palette (overall image)
- other

Rules:
- Return ONLY JSON. No markdown, no prose.
- JSON shape:
{
  "Font style": [ { "guideline": str, "evidence": str, "pages": [int, ...] } ],
  "Logo safe zone": [ { "guideline": str, "evidence": str, "pages": [int, ...] } ],
  "Logo colours": [ ... ],
  "Colour palette (overall image)": [ ... ],
  "other": [ ... ]
}
- "guideline": a short, self-contained instruction (imperative or declarative).
- "evidence": a short quote or paraphrase from the text that justifies the guideline.
- "pages": list of page numbers (1-based) where the evidence appears. If unknown, provide [].
- Classify:
  • "Logo safe zone": clear space, safe zone, spacing, X-measure references.
  • "Logo colours": color rules for the logo/icon.
  • "Colour palette (overall image)": brand palette, HEX lists, primary/secondary colors.
  • "Font style": typography families, weights, sizes, usage.
  • "other": everything else.
- Be concise. Do not invent facts. If a category has no items, return an empty list for it.
"""

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s).strip()
        s = re.sub(r"\s*```$", "", s).strip()
    return s
def classify_guidelines_full_text(
    full_text: str, 
    model: str = "o4-mini", 
    out_path: str | Path | None = None
) -> Dict[str, Any]:
    """
    Sends all extracted text at once and asks the model to extract + group guidelines.
    Saves JSON to disk if out_path is given.
    """
    user_prompt = (
        "Extract and group guideline statements from the following brand-guidelines text.\n\n"
        f"{full_text}"
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    out = _strip_fences(resp.output_text)

    try:
        data = json.loads(out)
    except Exception:
        data = {k: [] for k in CATEGORIES}

    for k in CATEGORIES:
        data.setdefault(k, [])

    # --- auto save if requested ---
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved classifications to {out_path}")

    return data

# ---- Keep only the 4 target categories and flatten to a list of guideline strings ----
FOUR_CATEGORIES = [
    "Font style",
    "Logo safe zone",
    "Logo colours",
    "Colour palette (overall image)",
]

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s).strip()
        s = re.sub(r"\s*```$", "", s).strip()
    return s
 
def keep_target_guidelines_by_category(grouped: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    From the grouped extraction (labels), keep only the 4 target categories,
    preserving the category names. Returns {category: [guideline, ...]}.
    Handles both shapes: list[dict{"guideline": ...}] or list[str].
    """
    out: Dict[str, List[str]] = {c: [] for c in FOUR_CATEGORIES}
    for cat in FOUR_CATEGORIES:
        items = grouped.get(cat, [])
        for it in items:
            g = (it.get("guideline") if isinstance(it, dict) else str(it)).strip()
            if g:
                out[cat].append(g)
        # de-dup while preserving order
        seen, uniq = set(), []
        for g in out[cat]:
            if g not in seen:
                uniq.append(g); seen.add(g)
        out[cat] = uniq
    return out



SYSTEM_VISION = (
    "You are a strict brand-compliance checker. Given CATEGORY-GROUPED brand guidelines "
    "and ONE image, decide for each guideline whether the image RESPECTS it. "
    "Return ONLY JSON with exactly these 4 top-level keys:\n"
    '{\n'
    '  "Font style": [ {"guideline":"...", "respected":"yes"|"no"} ],\n'
    '  "Logo safe zone": [ {"guideline":"...", "respected":"yes"|"no"} ],\n'
    '  "Logo colours": [ {"guideline":"...", "respected":"yes"|"no"} ],\n'
    '  "Colour palette (overall image)": [ {"guideline":"...", "respected":"yes"|"no"} ]\n'
    '}\n'
    "No markdown, no commentary."
)

def evaluate_guidelines_on_image_chat_grouped(
    grouped_guidelines: Dict[str, Any],
    image_path: str | Path,
    model: str = "gpt-4o-mini",
    out_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Visual check that preserves category names in both prompt and output.
    Auto-saves JSON named after the image if out_path is None.
    """
    grouped = keep_target_guidelines_by_category(grouped_guidelines)

    # If everything is empty, short-circuit
    if not any(grouped.values()):
        results = {c: [] for c in FOUR_CATEGORIES}
        # save if requested / derive filename
        if out_path is None:
            imgp = Path(image_path)
            out_path = imgp.with_name(imgp.stem + "_compliance.json")
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved compliance results to {outp}")
        return results

    # Build compact, category-grouped text for the model
    lines = []
    for cat in FOUR_CATEGORIES:
        if grouped[cat]:
            lines.append(f"{cat}:")
            for g in grouped[cat][:50]:  # keep it sane if very long
                lines.append(f"- {g}")
            lines.append("")  # blank line
    guidelines_text = "\n".join(lines)

    # Prep image as data URL
    imgp = Path(image_path)
    if not imgp.exists():
        raise FileNotFoundError(f"Image not found: {imgp.resolve()}")
    ext = imgp.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    import base64
    data_url = f"data:{mime};base64,{base64.b64encode(imgp.read_bytes()).decode('utf-8')}"


    # Call Chat Completions (vision)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_VISION},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Evaluate ONLY these guidelines against the attached image. Return pure JSON as specified."},
                    {"type": "text", "text": guidelines_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0,
    )

    raw = (resp.choices[0].message.content or "").strip()
    # Strip accidental code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()

    # Parse and normalize
    try:
        data = json.loads(raw)
    except Exception:
        # Conservative fallback: keep categories, mark all as "no"
        data = {c: [{"guideline": g, "respected": "no"}] for c, gls in grouped.items() for g in gls}
        # The above flattened; fix to proper shape:
        data = {c: [{"guideline": g, "respected": "no"} for g in grouped[c]] for c in FOUR_CATEGORIES}

    # Ensure all 4 keys exist and values are lists of dicts with yes/no
    norm: Dict[str, List[Dict[str, str]]] = {}
    for cat in FOUR_CATEGORIES:
        items = data.get(cat, [])
        fixed = []
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and "guideline" in it and "respected" in it:
                v = str(it["respected"]).strip().lower()
                v = "yes" if v == "yes" else "no"
                fixed.append({"guideline": str(it["guideline"]).strip(), "respected": v})
        norm[cat] = fixed

    # Auto-derive filename from image if not provided
    if out_path is None:
        out_path = imgp.with_name(imgp.stem + "_compliance.json")

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(norm, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved compliance results to {outp}")

    return norm



GROUPED_PATH = Path("guidelines_grouped.json")

def load_or_create_grouped(pdf_path: str, model: str = "o4-mini") -> dict:
    """
    If guidelines_grouped.json exists, load it.
    Otherwise extract from PDF, classify, and save to guidelines_grouped.json.
    """
    if GROUPED_PATH.exists():
        print("Found existing guidelines_grouped.json – loading it.")
        with open(GROUPED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print("guidelines_grouped.json not found – extracting and classifying.")
        result = extract_guidelines_text(pdf_path)
        full_text = result["text"]

        labels = classify_guidelines_full_text(
            full_text,
            model=model,
            out_path=GROUPED_PATH
        )
        return labels

def _to_bool_yes_no(v: str) -> bool:
    """Robustly interpret the model's 'yes'/'no' (case-insensitive)."""
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"yes", "true", "1"}

def score_compliance(compliance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute scores from a compliance dict shaped like:
    {
      "Font style": [ {"guideline": "...", "respected": "yes"|"no"}, ... ],
      ...
    }

    Each category is worth up to 1.0. Within a category, each guideline has equal weight.
    Category score = (# respected) / (total guidelines). If a category has 0 guidelines, score = 0.
    Returns a summary with per-category breakdown and total.
    """
    per_category = {}
    total_score = 0.0
    max_total = float(len(FOUR_CATEGORIES))

    for cat in FOUR_CATEGORIES:
        items = compliance.get(cat, [])
        total = len(items) if isinstance(items, list) else 0
        yes_count = 0

        if total > 0:
            for it in items:
                if isinstance(it, dict):
                    if _to_bool_yes_no(it.get("respected")):
                        yes_count += 1
            score = yes_count / total  # 0..1
        else:
            score = 0.0

        per_category[cat] = {
            "yes": yes_count,
            "total": total,
            "score": round(score, 3),
        }
        total_score += score

    result = {
        "per_category": per_category,
        "total_score": round(total_score, 3),  # 0..4
        "max_total": int(max_total),
        "normalized_0_100": round((total_score / max_total) * 100, 1),  # handy %
    }
    return result

def score_from_compliance_file(
    compliance_json_path: str | Path,
    out_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Load a *_compliance.json file, compute scores, and save them to *_score.json
    (unless out_path is provided). Returns the score dict.
    """
    compliance_json_path = Path(compliance_json_path)
    if not compliance_json_path.exists():
        raise FileNotFoundError(f"Compliance file not found: {compliance_json_path}")

    compliance = json.loads(compliance_json_path.read_text(encoding="utf-8"))
    scores = score_compliance(compliance)

    if out_path is None:
        out_path = compliance_json_path.with_name(
            compliance_json_path.stem.replace("_compliance", "") + "_score.json"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved scores to {out_path}")
    return scores

def score_from_image_path(
    image_path: str | Path,
    out_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Convenience: derive the compliance JSON path from an image path.
    Expects the compliance file to be named <image_stem>_compliance.json
    (e.g., neurons_3.png -> neurons_3_compliance.json).
    """
    image_path = Path(image_path)
    comp_path = image_path.with_name(image_path.stem + "_compliance.json")
    return score_from_compliance_file(comp_path, out_path)



def run_brand_compliance(
    pdf_path: str | Path,
    image_path: str | Path,
    text_model: str = "o4-mini",
    vision_model: str = "gpt-4o-mini",
    force_reclassify: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline:
      1) Load or create grouped guidelines from the PDF (text_model).
      2) Evaluate the image against grouped guidelines (vision_model).
      3) Score the result (per-category max 1 point, equal weight per guideline).
      4) Return dict with objects + file paths.

    Returns:
      {
        "grouped_path": "guidelines_grouped.json",
        "compliance_path": "<image_stem>_compliance.json",
        "score_path": "<image_stem>_score.json",
        "grouped": {...},
        "compliance": {...},
        "score": {...}
      }
    """
    pdf_path = Path(pdf_path)
    image_path = Path(image_path)
    grouped_path = Path("guidelines_grouped.json")

    # 1) grouped guidelines (load or build)
    if grouped_path.exists() and not force_reclassify:
        print("Found existing guidelines_grouped.json – loading it.")
        grouped = json.loads(grouped_path.read_text(encoding="utf-8"))
    else:
        print("Building guidelines_grouped.json from PDF…")
        grouped = load_or_create_grouped(str(pdf_path), model=text_model)

    # 2) vision compliance (auto-saves as <image>_compliance.json if out_path=None)
    print(f"Evaluating compliance for {image_path.name}…")
    compliance = evaluate_guidelines_on_image_chat_grouped(
        grouped_guidelines=grouped,
        image_path=str(image_path),
        model=vision_model,
        out_path=None,  # auto -> <image_stem>_compliance.json
    )
    compliance_path = image_path.with_name(image_path.stem + "_compliance.json")

    # 3) scoring (auto-saves as <image>_score.json)
    print(f"Scoring compliance for {image_path.name}…")
    score = score_from_image_path(image_path)
    score_path = image_path.with_name(image_path.stem + "_score.json")

    return {
        "grouped_path": str(grouped_path),
        "compliance_path": str(compliance_path),
        "score_path": str(score_path),
        "grouped": grouped,
        "compliance": compliance,
        "score": score,
    }

# --- Optional CLI-style usage ---
if __name__ == "__main__":
    out = run_brand_compliance(
        pdf_path="Neurons_brand_kit.pdf",
        image_path="neurons_2.png",
        text_model="o4-mini",
        vision_model="gpt-4o-mini",
        force_reclassify=False,  # set True to rebuild guidelines_grouped.json
    )
    print("Done.\n"
          f"- Grouped:   {out['grouped_path']}\n"
          f"- Compliance:{out['compliance_path']}\n"
          f"- Score:     {out['score_path']}")
    


