import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# 1. Configuração da Página
st.set_page_config(page_title="Predição de Obesidade", layout="wide")

MODEL_PATH = 'modelo_obesidade.pkl'
ENCODER_PATH = 'label_encoder.pkl'

@st.cache_resource
def carregar_recursos():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Arquivo '{MODEL_PATH}' não encontrado.")
        return None, None
    try:
        dados = joblib.load(MODEL_PATH)
        if isinstance(dados, (list, tuple)) and len(dados) == 2:
            return dados[0], dados[1]
        return dados, None
    except Exception as e:
        st.error(f"Erro ao carregar recursos: {e}")
        return None, None

pipeline, le = carregar_recursos()

# --- INTERFACE ---
st.title("🏥 Sistema de Apoio ao Diagnóstico de Obesidade")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios"])

with tab1:
    st.header("Formulário do Paciente")
    
    if pipeline is None:
        st.error("Modelo não carregado corretamente.")
    else:
        col1, col2, col3 = st.columns(3)

        # Mapeamentos ajustados para os valores que o seu modelo espera (verifique se é 'yes'/'no' ou 'Sim'/'Não')
        mapa_sim_nao = {'Sim': 'yes', 'Não': 'no'}
        mapa_genero = {'Masculino': 'Male', 'Feminino': 'Female'}

        with col1:
            genero = st.selectbox("Gênero", ["Masculino", "Feminino"])
            idade = st.number_input("Idade", 1, 120, 25)
            altura = st.number_input("Altura (m)", 0.5, 2.5, 1.70)
            peso = st.number_input("Peso (kg)", 10.0, 300.0, 70.0)

        with col2:
            hist_fam = st.selectbox("Histórico Familiar de Sobrepeso?", ["Sim", "Não"])
            favc = st.selectbox("Consome comida calórica frequentemente?", ["Sim", "Não"])
            fcvc = st.slider("Frequência de vegetais (1-3)", 1, 3, 2)
            ncp = st.slider("Refeições principais", 1, 4, 3)
            caec = st.selectbox("Come entre refeições?", ['Sometimes', 'Frequently', 'Always', 'no'])

        with col3:
            smoke = st.selectbox("Fumante?", ["Sim", "Não"])
            ch2o = st.slider("Consumo de água (1-3)", 1, 3, 2)
            scc = st.selectbox("Monitora calorias?", ["Sim", "Não"])
            faf = st.slider("Atividade física (0-3)", 0, 3, 1)
            tue = st.slider("Uso de eletrônicos (0-2)", 0, 2, 1)
            calc = st.selectbox("Consumo de álcool", ['Sometimes', 'Frequently', 'Always', 'no'])
            mtrans = st.selectbox("Meio de transporte", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

        if st.button("Realizar Diagnóstico"):
            try:
                # DICIONÁRIO COM OS NOMES EXATOS QUE O SEU ERRO PEDIU
                dados_paciente = {
                    'genero': [mapa_genero[genero]],
                    'idade': [idade],
                    'altura_m': [altura],
                    'peso_kg': [peso],
                    'historia_familiar_sobrepeso': [mapa_sim_nao[hist_fam]],
                    'come_comida_calorica_freq': [mapa_sim_nao[favc]],
                    'freq_consumo_vegetais': [fcvc],
                    'num_refeicoes_principais': [ncp],
                    'come_entre_refeicoes': [caec],
                    'fumante': [mapa_sim_nao[smoke]],
                    'consumo_agua_litros': [ch2o],
                    'monitora_calorias': [mapa_sim_nao[scc]],
                    'freq_atividade_fisica': [faf],
                    'tempo_uso_dispositivos': [tue],
                    'freq_consumo_alcool': [calc],
                    'meio_transporte': [mtrans]
                }

                df_input = pd.DataFrame(dados_paciente)

                pred = pipeline.predict(df_input)
                
                if le:
                    resultado = le.inverse_transform(pred)[0]
                else:
                    resultado = pred[0]

                st.success(f"### Resultado: {resultado}")
                st.info(f"**IMC:** {peso/(altura**2):.2f}")
                
            except Exception as e:
                st.error(f"Erro na predição: {e}")

# (Abas 2 e 3 permanecem as mesmas do código anterior)
