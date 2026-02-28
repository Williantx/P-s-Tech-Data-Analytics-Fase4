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

# 2. Função de Carregamento de Recursos
@st.cache_resource
def carregar_recursos():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Arquivo '{MODEL_PATH}' não encontrado.")
        return None, None
    try:
        dados = joblib.load(MODEL_PATH)
        # Se o .pkl contiver [modelo, encoder]
        if isinstance(dados, (list, tuple)) and len(dados) == 2:
            return dados[0], dados[1]
        
        # Se contiver apenas o modelo, tenta carregar encoder separado
        pipeline = dados
        le = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None
        return pipeline, le
    except Exception as e:
        st.error(f"Erro ao carregar recursos: {e}")
        return None, None

# Chamada da função
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

        col1, col2, col3 = st.columns(3)

        with col1:
            genero_visual = st.selectbox("Gênero", list(mapa_genero.keys()))
            idade = st.number_input("Idade", 1, 120, 25)
            altura = st.number_input("Altura (m)", 0.5, 2.5, 1.70)
            peso = st.number_input("Peso (kg)", 10.0, 300.0, 70.0)
            hist_fam_visual = st.selectbox("Histórico Familiar de Sobrepeso?", list(mapa_sim_nao.keys()))

        with col2:
            favc_visual = st.selectbox("Consome comida calórica frequentemente?", list(mapa_sim_nao.keys()))
            fcvc = st.slider("Frequência de consumo de vegetais (1-3)", 1, 3, 2)
            ncp = st.slider("Número de refeições principais", 1, 4, 3)
            caec_visual = st.selectbox("Come entre refeições?", list(mapa_frequencia.keys()))
            smoke_visual = st.selectbox("Fumante?", list(mapa_sim_nao.keys()))

        with col3:
            ch2o = st.slider("Consumo de água diário (1-3L)", 1, 3, 2)
            scc_visual = st.selectbox("Monitora calorias ingeridas?", list(mapa_sim_nao.keys()))
            faf = st.slider("Frequência de atividade física (0-3)", 0, 3, 1)
            tue = st.slider("Tempo usando dispositivos (0-2)", 0, 2, 1)
            calc_visual = st.selectbox("Consumo de álcool", list(mapa_frequencia.keys()))
            mtrans_visual = st.selectbox("Meio de transporte principal", list(mapa_transporte.keys()))

        if st.button("Realizar Diagnóstico"):
            try:
                # DataFrame com os nomes exatos exigidos pelo seu modelo (Português)
                df_input = pd.DataFrame({
                    'genero': [mapa_genero[genero_visual]],
                    'idade': [idade],
                    'altura_m': [altura],
                    'peso_kg': [peso],
                    'historia_familiar_sobrepeso': [mapa_sim_nao[hist_fam_visual]],
                    'come_comida_calorica_freq': [mapa_sim_nao[favc_visual]],
                    'freq_consumo_vegetais': [fcvc],
                    'num_refeicoes_principais': [ncp],
                    'come_entre_refeicoes': [mapa_frequencia[caec_visual]],
                    'fumante': [mapa_sim_nao[smoke_visual]],
                    'consumo_agua_litros': [ch2o],
                    'monitora_calorias': [mapa_sim_nao[scc_visual]],
                    'freq_atividade_fisica': [faf],
                    'tempo_uso_dispositivos': [tue],
                    'freq_consumo_alcool': [mapa_frequencia[calc_visual]],
                    'meio_transporte': [mapa_transporte[mtrans_visual]]
                })

                # Predição
                pred = pipeline.predict(df_input)
                
                # Descodificação
                if le:
                    resultado = le.inverse_transform(pred)[0]
                else:
                    resultado = pred[0]

                st.success(f"### Diagnóstico Sugerido: {resultado}")
                st.metric("IMC Calculado", f"{peso/(altura**2):.2f}")
                
            except Exception as e:
                st.error(f"Erro na predição: {e}")

# --- Outras Abas ---
with tab2:
    st.header("📊 Dashboard Analítico")
    st.info("Estatísticas da base de dados original.")

with tab3:
    st.header("📝 Relatórios e Insights")
    st.link_button("🚀 Abrir no Looker Studio", "https://lookerstudio.google.com")
