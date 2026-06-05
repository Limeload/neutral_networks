---
title: Brain Tumor MRI Classification
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Brain Tumor Classification

Streamlit app that classifies brain MRI scans into four categories — **glioma, meningioma, no tumor, pituitary** — using a custom CNN and Xception, with OpenAI-powered clinical reports and multi-turn image chat.

## Stack

- **Models** — Xception (transfer learning, target ≥99%) + Custom SE-ResNet CNN (target ≥98%)
- **App** — Streamlit with saliency map overlays and probability charts
- **AI** — OpenAI GPT-4o for clinical report generation and MRI chat
- **Model storage** — Hugging Face Hub (downloaded on first run if not present locally)

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — set OPENAI_API_KEY and HF_MODEL_REPO
streamlit run app/app.py
```

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI key for the report and chat tabs. Can also be entered in the app sidebar — never commit this value. |
| `HF_MODEL_REPO` | HF repo ID hosting the `.keras` weights (e.g. `youruser/brain-tumor-models`). Not needed if weights are present in `models/`. |

For HF Spaces deployments set both in **Settings → Repository secrets**.

## Training

```bash
python train.py                  # train both models
python train.py --model xception # one model only
python train.py --quick          # smoke-test (few epochs)
python train.py --strict         # exit 1 if accuracy targets are missed (CI)
```

Training notebooks with architecture notes are in `notebooks/`.

## Configuration

Model registry and class metadata are in [`config.toml`](config.toml). Add a `[[models]]` entry there to register a new model — no Python changes needed.

## Project structure

```
├── app/            Streamlit app
├── notebooks/      Training notebooks
├── utils/          Inference helpers
├── models/         .keras weight files (git-ignored)
├── config.toml     Model registry and class metadata
├── train.py        Training script
└── Dockerfile      HF Spaces deployment
```

## Deployment security

`.streamlit/config.toml` keeps `enableCORS` and `enableXsrfProtection` set to `true`. **Do not disable either in a shared deployment** — see [Streamlit security docs](https://docs.streamlit.io/develop/api-reference/configuration/config.toml#server) for details.

## Disclaimer

**For research and educational purposes only. Not medical advice.**

Model predictions can be wrong. GPT-generated reports are AI-produced and unreviewed by clinicians. Do not use this tool as a substitute for professional diagnosis.
