import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# Configuração da Página
st.set_page_config(page_title="Predição de Obesidade", layout="wide")

# --- DEFINIÇÃO DE CAMINHOS ---
# Certifique-se de que estes arquivos existam na mesma pasta do app.py
MODEL_PATH = 'modelo_obesidade.pkl'
ENCODER_PATH = 'label_encoder.pkl' # Ajuste o nome se for diferente

# --- FUNÇÃO DE CARGA ROBUSTA ---
@st.cache_resource
def carregar_recursos():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Arquivo de modelo '{MODEL_PATH}' não encontrado.")
        return None, None
    
    try:
        conteudo = joblib.load(MODEL_PATH)
        
        # Caso 1: O .pkl já contém [modelo, encoder] juntos
        if isinstance(conteudo, (tuple, list)) and len(conteudo) == 2:
            return conteudo[0], conteudo[1]
        
        # Caso 2: O .pkl só tem o modelo, busca o encoder em arquivo separado
        pipeline = conteudo
        if os.path.exists(ENCODER_PATH):
            le = joblib.load(ENCODER_PATH)
            return pipeline, le
        else:
            st.error("O arquivo de modelo foi carregado, mas o LabelEncoder não foi encontrado.")
            return pipeline, None
            
    except Exception as e:
        st.error(f"Erro crítico no carregamento: {e}")
        return None, None

# Inicialização dos recursos
pipeline, le = carregar
