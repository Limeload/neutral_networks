---
title: Brain Tumor MRI Classification
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Brain Tumor Classification

Streamlit app that classifies brain MRI scans into four categories — **glioma, meningioma, no tumor, pituitary** — using a custom CNN with OpenAI-powered clinical reports and multi-turn image chat.

## Stack

- **Model** — Custom residual CNN trained on the Brain Tumor MRI Dataset (Kaggle)
- **App** — Streamlit with saliency map overlays and probability charts
- **AI** — OpenAI GPT-4o for clinical report generation and MRI chat

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — set OPENAI_API_KEY and HF_MODEL_REPO
streamlit run app/app.py
```

## Project structure

```
├── app/              Streamlit app
├── notebooks/        Training notebooks
├── models/           Saved .keras model files (git-ignored, downloaded from HF Hub)
├── utils/            Inference helpers
└── Dockerfile        HF Spaces deployment
```
