---
title: Brain Tumor MRI Classification
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Brain Tumor MRI Classification

A Streamlit application that classifies brain MRI scans into four categories using two independent deep-learning models, then generates AI-assisted clinical summaries and supports free-form image chat powered by OpenAI.

**Classes:** glioma · meningioma · no tumor · pituitary

---

## Features

| Tab | Description |
|---|---|
| Class Probabilities | Per-model confidence bars for all four classes |
| Model Comparison | Side-by-side probability charts and saliency map overlays |
| Clinical Report | GPT-4o structured report (8-section clinical summary, downloadable as Markdown) |
| Image Analysis | Multi-turn chat — ask follow-up questions about the MRI |

---

## Stack

| Layer | Technology |
|---|---|
| Classification | TensorFlow / Keras — Xception (transfer learning) + Custom SE-ResNet CNN |
| App | Streamlit |
| AI assistant | OpenAI GPT-4o / GPT-4o mini |
| Model storage | Hugging Face Hub (downloaded on first run if not present locally) |
| Deployment | Docker on Hugging Face Spaces |

---

## Local setup

**Requirements:** Python 3.11+

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Open .env and fill in the two variables (see Environment variables below)

# 4. Launch the app
streamlit run app/app.py
```

The app opens at `http://localhost:8501`.

---

## Environment variables

Create `.env` at the repo root (copy from `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | For report & chat tabs | Your OpenAI API key (`sk-...`). Never commit this value. |
| `HF_MODEL_REPO` | If model files are absent locally | Hugging Face repo ID that hosts the `.keras` weights, e.g. `youruser/brain-tumor-models`. |

**Alternatively**, enter your OpenAI API key directly in the sidebar or in the inline input that appears on the Report and Image Analysis tabs. The key is stored only in the browser session (`st.session_state`) and is never logged or transmitted beyond the OpenAI API call.

For Hugging Face Spaces deployments, set both variables in the Space's **Settings → Repository secrets** panel instead of in `.env`.

---

## Model weights

Weights are not committed to the repository. On first run the app checks for local `.keras` files in `models/`; if absent and `HF_MODEL_REPO` is set, they are downloaded from Hugging Face Hub automatically.

To place weights manually, drop the files into `models/` with the filenames defined in `config.toml`.

---

## Training

```bash
# Train both models (full run, ~2–3 h on GPU)
python train.py

# Train one model only
python train.py --model xception
python train.py --model cnn

# Quick smoke-test (few epochs, verifies the pipeline end-to-end)
python train.py --quick

# Strict mode — exits with code 1 if any model misses its accuracy target
# Use this in CI to catch data or environment regressions
python train.py --strict
```

Accuracy targets: Xception ≥ 99% · Custom CNN ≥ 98%

Training notebooks with detailed commentary are in `notebooks/`:

- `01_xception_model.ipynb` — transfer learning with two-phase fine-tuning
- `02_custom_cnn.ipynb` — custom SE-ResNet built from scratch, with architecture rationale

---

## Configuration

Model registry, LLM options, and class metadata live in [`config.toml`](config.toml) at the repo root — no Python editing needed to add or swap a model:

```toml
[[models]]
name       = "My New Model"
filename   = "my_model.keras"
input_size = [224, 224]
```

---

## Project structure

```
├── app/
│   └── app.py                  Streamlit application
├── notebooks/
│   ├── 01_xception_model.ipynb Xception transfer-learning notebook
│   └── 02_custom_cnn.ipynb     Custom SE-ResNet notebook
├── utils/
│   └── predict.py              Inference helpers (predict, compute_saliency)
├── models/                     .keras weight files (git-ignored)
├── data/                       Dataset (git-ignored)
├── config.toml                 Model registry and class metadata
├── train.py                    Training script
├── Dockerfile                  HF Spaces deployment
└── .streamlit/config.toml      Streamlit server settings
```

---

## Deployment security

`.streamlit/config.toml` keeps `enableCORS` and `enableXsrfProtection` set to `true` (the Streamlit defaults). **Do not set either to `false` in a shared or public deployment** — doing so disables CORS and CSRF protections and exposes users to browser-based attacks. If you need to work around a reverse-proxy issue locally, disable them only in a private environment and re-enable before deploying.

---

## Limitations and responsible AI disclaimer

> **This tool is for research and educational purposes only. It does not constitute medical advice and must not be used as a substitute for professional clinical diagnosis.**

- Model predictions are probabilistic and can be wrong, particularly for ambiguous or low-quality scans.
- The models were trained on the [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and may not generalise to MRI scans acquired with different equipment, protocols, or patient populations.
- GPT-4o-generated clinical reports are AI-produced text. They have not been reviewed by a medical professional and may contain errors or hallucinations.
- Saliency maps highlight regions the model weighted most — they are an interpretability aid, not a clinically validated localisation tool.
- No patient data is stored by this application. Uploaded images are processed in memory and written to a temporary path (`/tmp`) for the duration of the session only.
