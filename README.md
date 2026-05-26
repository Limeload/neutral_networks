# Brain Tumor Classification

Deep learning app that classifies brain MRI scans into four categories — **glioma, meningioma, no tumor, pituitary** — using two CNN models with AI-generated explanations via Gemini.

## Stack

- **Models** — Xception (transfer learning) + custom residual CNN, trained in Jupyter
- **Backend** — FastAPI serving predictions and saliency maps
- **Frontend** — Next.js 14 + Tailwind CSS
- **AI explanations** — Gemini 1.5 Flash / Pro (multimodal)

## Setup

```bash
# 1. Download dataset (requires Kaggle API key at ~/.kaggle/kaggle.json)
python data/download_dataset.py

# 2. Train models — open and run notebooks in order
#    notebooks/01_xception_model.ipynb
#    notebooks/02_custom_cnn.ipynb

# 3. Install backend dependencies
pip install -r api/requirements.txt

# 4. Install frontend dependencies
cd frontend && npm install && cd ..

# 5. Add your Gemini API key
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...

# 6. Start both servers (run from project root)
./start.sh

# Or run separately in two terminals:
# Terminal 1: cd api && uvicorn main:app --reload --port 8000
# Terminal 2: cd frontend && npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Project structure

```
├── api/              FastAPI backend (predict, explain, chat endpoints)
├── frontend/         Next.js app
├── notebooks/        Training notebooks (Xception + custom CNN)
├── models/           Saved .keras model files (git-ignored)
├── data/             Dataset download script
└── utils/            Shared inference helpers
```

## Features

- Drag-and-drop MRI upload
- Side-by-side predictions from both models with confidence bars
- Saliency map overlays showing what each model focuses on
- AI explanation with selectable LLM (Gemini 1.5 Flash or Pro)
- Chat with the MRI scan for follow-up questions
# neutral_networks
