import base64
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.predict import CLASSES, compute_saliency, predict

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

st.set_page_config(page_title='Brain Tumor Classifier', page_icon='🧠', layout='wide')

MODEL_PATHS = {
    'Custom CNN': (
        os.path.join(os.path.dirname(__file__), '..', 'models', 'custom_cnn_brain_tumor.keras'),
        (224, 224),
    ),
    'Xception': (
        os.path.join(os.path.dirname(__file__), '..', 'models', 'xception_brain_tumor.keras'),
        (299, 299),
    ),
}

OPENAI_MODELS = {
    'GPT-4o':      'gpt-4o',
    'GPT-4o mini': 'gpt-4o-mini',
}

CLASS_INFO = {
    'glioma': {
        'name': 'Glioma',
        'description': 'A tumor arising from glial cells in the brain or spinal cord.',
        'subtypes': 'Astrocytoma, Oligodendroglioma, Glioblastoma (GBM)',
        'prevalence': '~33% of all brain tumors',
        'severity': 'High',
        'color': '#FF4B4B',
    },
    'meningioma': {
        'name': 'Meningioma',
        'description': 'A tumor that forms on the meninges — membranes covering the brain and spinal cord.',
        'subtypes': 'Grade I (benign ~80%), Grade II (atypical), Grade III (anaplastic)',
        'prevalence': '~37% of all brain tumors',
        'severity': 'Moderate',
        'color': '#FFA500',
    },
    'no_tumor': {
        'name': 'No Tumor',
        'description': 'No tumor detected in this MRI scan.',
        'subtypes': 'N/A',
        'prevalence': 'N/A',
        'severity': 'None',
        'color': '#00CC44',
    },
    'pituitary': {
        'name': 'Pituitary Tumor',
        'description': 'A tumor that forms in the pituitary gland at the base of the brain.',
        'subtypes': 'Adenoma (most common), Craniopharyngioma (rare)',
        'prevalence': '~17% of all brain tumors',
        'severity': 'Low–Moderate',
        'color': '#4B8BFF',
    },
}


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None


def _b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()


def _image_message(img_bytes: bytes, text: str) -> dict:
    return {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': text},
            {'type': 'image_url', 'image_url': {
                'url': f'data:image/jpeg;base64,{_b64(img_bytes)}', 'detail': 'high',
            }},
        ],
    }


def _prediction_summary(results: dict) -> str:
    return '\n'.join(
        f'- {name}: {CLASS_INFO[r["class"]]["name"]} ({r["confidence"]:.1%} confidence)'
        for name, r in results.items()
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title('⚙️ Settings')

    api_key = st.text_input(
        'OpenAI API Key',
        value=os.getenv('OPENAI_API_KEY', ''),
        type='password',
        help='Loaded from .env if present.',
    )

    st.markdown('---')
    st.subheader('Models')
    model_enabled = {name: st.checkbox(name, value=True) for name in MODEL_PATHS}

    enabled_models = {n: MODEL_PATHS[n] for n, on in model_enabled.items() if on}

    st.markdown('---')
    st.subheader('LLM (Challenge 3)')
    selected_llm_name = st.selectbox('Multimodal model', list(OPENAI_MODELS.keys()))
    selected_llm_id   = OPENAI_MODELS[selected_llm_name]


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title('🧠 Brain Tumor MRI Classifier')
st.caption('Upload an MRI scan to classify it with multiple CNN models and get AI-powered insights.')

uploaded = st.file_uploader('Upload MRI image', type=['jpg', 'jpeg', 'png'])

results: dict = {}
img_bytes: bytes | None = None

if uploaded:
    img_bytes = uploaded.read()
    tmp_path  = f'/tmp/{uploaded.name}'
    with open(tmp_path, 'wb') as f:
        f.write(img_bytes)

    if not enabled_models:
        st.warning('Enable at least one model in the sidebar.')
    else:
        with st.spinner('Running predictions…'):
            for name, (path, size) in enabled_models.items():
                model = load_model(path)
                if model is None:
                    st.error(f'{name} model not found at `{path}`.')
                    continue
                results[name] = predict(model, tmp_path, size)
                results[name]['model_name'] = name
                results[name]['model_size'] = size
                results[name]['model']      = model
