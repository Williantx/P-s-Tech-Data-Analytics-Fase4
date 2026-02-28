import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# 1. Configuração da Página
st.set_page_config(page_title="Predição de Obesidade", layout="wide")

# Caminhos dos arquivos
MODEL_PATH = 'modelo_obesidade.pkl'
ENCODER_PATH = 'label_encoder.pkl'

# 2. Função de Carregamento
@st.cache_resource
def carregar_recursos():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Arquivo '{MODEL_PATH}' não encontrado.")
        return None, None
    try:
        dados = joblib.load(MODEL_PATH)
        if isinstance(dados, (list, tuple)) and len(dados) == 2:
            return dados[0], dados[1]
        
        pipeline = dados
        le = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None
        return pipeline, le
    except Exception as e:
        st.error(f"Erro ao carregar recursos: {e}")
        return None, None

pipeline, le = carregar_recursos()

# 3. Interface Principal
st.title("🏥 Sistema de Apoio ao Diagnóstico de Obesidade")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios"])

with tab1:
    st.header("Formulário do Paciente")
    
    if pipeline is None:
        st.warning("Aguardando carregamento do modelo...")
    else:
        col1, col2, col3 = st.columns(3)

        # Dicionários de Tradução (Visual -> Modelo)
        mapa_genero = {'Masculino': 'Male', 'Feminino': 'Female'}
        mapa_sim_nao = {'Sim': 'yes', 'Não': 'no'}
        mapa_frequencia = {
            'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 
            'Sempre': 'Always', 'Não': 'no'
        }
        mapa_transporte = {
            'Transporte Público': 'Public_Transportation', 'Caminhada': 'Walking', 
            'Carro': 'Automobile', 'Moto': 'Motorbike', 'Bicicleta': 'Bike'
        }

        with col1:
            genero = st.selectbox("Gênero", list(mapa_genero.keys()))
            idade = st.number_input("Idade", 1, 120, 25)
            altura = st.number_input("Altura (m)", 0.5, 2.5, 1.70)
            peso = st.number_input("Peso (kg)", 10.0, 300.0, 70.0)
            hist_fam = st.
