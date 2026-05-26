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
