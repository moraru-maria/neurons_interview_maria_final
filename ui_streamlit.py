import os
import json
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
from PIL import Image

# --- Your logic module ---
from logic import (
    extract_guidelines_text,
    classify_guidelines_full_text,
    save_about_logo_pages,
    evaluate_guidelines_on_image_chat_grouped,
    score_compliance,
)

# ---------- Settings ----------
st.set_page_config(page_title="Brand Compliance Checker", layout="wide")
DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
for p in [DATA_DIR, UPLOADS_DIR, OUTPUTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- Header ----------
st.title("Brand Compliance Checker")
st.caption(
    "Upload a brand PDF and a candidate image. We’ll extract guidelines, "
    "pull reference logo pages, and evaluate compliance."
)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Settings")
    text_model = st.text_input("Text model", "o4-mini")
    vision_model = st.text_input("Vision model", "gpt-4o-mini")
    st.markdown("---")
    st.info("Make sure `OPENAI_API_KEY` is set, otherwise the evaluation cannot be completed.")

# ---------- Uploaders (drag & drop) ----------
col_pdf, col_img = st.columns(2)
with col_pdf:
    pdf_file = st.file_uploader("Brand guidelines PDF", type=["pdf"], accept_multiple_files=False)
with col_img:
    img_file = st.file_uploader("Candidate image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

# ---------- Run Button ----------
run = st.button("Run evaluation", type="primary", use_container_width=True)

# ---------- Helper ----------
def save_upload(upload, target_dir: Path) -> Path:
    out = target_dir / upload.name
    with open(out, "wb") as f:
        f.write(upload.read())
    return out

def show_image(path: Path, caption: str = ""):
    try:
        st.image(str(path), caption=caption, use_container_width=True)
    except Exception:
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=True)

# ---------- Main flow ----------
if run:
    if not pdf_file or not img_file:
        st.error("Please upload both a PDF and an image.")
        st.stop()

    with st.status("Processing…", expanded=True) as status:
        # Save uploads
        st.write("Saving files…")
        pdf_path = save_upload(pdf_file, UPLOADS_DIR)
        candidate_img_path = save_upload(img_file, UPLOADS_DIR)

        # 1) Extract + classify guidelines
        st.write("Extracting text from PDF…")
        extracted = extract_guidelines_text(pdf_path, out_text_path=OUTPUTS_DIR / "guidelines_text.txt")

        st.write("Classifying guidelines with the text model…")
        grouped = classify_guidelines_full_text(
            full_text=extracted["text"],
            model=text_model,
            out_path=OUTPUTS_DIR / "guidelines_grouped.json"
        )

        # 2) Extract reference logo pages (no zoom arg)
        st.write("Extracting reference logo pages (About the logo)…")
        try:
            ref_images = save_about_logo_pages(pdf_path, out_dir=OUTPUTS_DIR)
        except Exception as e:
            st.warning(f"Reference extraction failed: {e}")
            ref_images = []

        # Preview reference and candidate
        with st.expander("Preview images", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Candidate image**")
                show_image(candidate_img_path, caption=candidate_img_path.name)
            with c2:
                st.markdown("**Reference logo pages**")
                if ref_images:
                    for p in ref_images:
                        show_image(Path(p), caption=Path(p).name)
                else:
                    st.write("_No reference pages found in the PDF._")

        # 3) Evaluate compliance (attach references if function supports them)
        st.write("Evaluating compliance with the vision model…")
        try:
            compliance = evaluate_guidelines_on_image_chat_grouped(
                grouped_guidelines=grouped,
                image_path=str(candidate_img_path),
                model=vision_model,
                out_path=OUTPUTS_DIR / f"{candidate_img_path.stem}_compliance.json",
                reference_logo_images=ref_images if ref_images else None,
            )
        except TypeError:
            compliance = evaluate_guidelines_on_image_chat_grouped(
                grouped_guidelines=grouped,
                image_path=str(candidate_img_path),
                model=vision_model,
                out_path=OUTPUTS_DIR / f"{candidate_img_path.stem}_compliance.json",
            )

        # 4) Score
        st.write("Scoring results…")
        scores = score_compliance(compliance)
        score_path = OUTPUTS_DIR / f"{candidate_img_path.stem}_score.json"
        score_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

        status.update(label="Done ✅", state="complete")

    # ---------- Results display ----------
    st.subheader("Results")

    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.markdown("### Per-category breakdown")
        for cat, meta in scores["per_category"].items():
            st.write(f"**{cat}** — {meta['yes']} / {meta['total']} respected  |  score: {meta['score']:.3f}")
            items = compliance.get(cat, [])
            if items:
                for it in items:
                    st.write(f"- {it.get('guideline','').strip()} → **{it.get('respected','no')}**")

    with c_right:
        st.markdown("### Summary")
        st.metric("Total (0..4)", f"{scores['total_score']:.3f}")
        st.metric("Normalized %", f"{scores['normalized_0_100']:.1f}%")

    st.markdown("---")
    st.markdown("### Downloads")
    grouped_path = OUTPUTS_DIR / "guidelines_grouped.json"
    compliance_path = OUTPUTS_DIR / f"{candidate_img_path.stem}_compliance.json"

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button(
            "Download grouped guidelines JSON",
            data=grouped_path.read_bytes(),
            file_name=grouped_path.name,
            mime="application/json",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "Download compliance JSON",
            data=compliance_path.read_bytes(),
            file_name=compliance_path.name,
            mime="application/json",
            use_container_width=True,
        )
    with col_c:
        st.download_button(
            "Download scores JSON",
            data=score_path.read_bytes(),
            file_name=score_path.name,
            mime="application/json",
            use_container_width=True,
        )

else:
    st.info("Drag-and-drop your PDF and image above, then click **Run evaluation**.")
