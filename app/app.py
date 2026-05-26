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
