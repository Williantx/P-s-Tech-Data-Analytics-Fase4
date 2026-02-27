import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit.components.v1 as components

# Configuração da Página
st.set_page_config(page_title="Predição de Obesidade", layout="wide")

# Caminhos locais
MODEL_PATH = 'modelo_obesidade.pkl'
LE_PATH = 'label_encoder.pkl'
DATA_PATH = 'Obesity.csv'

# Carregar o modelo e o encoder
@st.cache_resource
def carregar_recursos():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        st.error("Erro: Arquivos 'modelo_obesidade.pkl' ou 'label_encoder.pkl' não encontrados.")
        return None, None
    try:
        modelo = joblib.load(MODEL_PATH)
        encoder = joblib.load(LE_PATH)
        return modelo, encoder
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None

pipeline, le = carregar_recursos()

# Título
st.title("🏥 Sistema de Apoio ao Diagnóstico de Obesidade")
st.markdown("---")

# Abas
tab1, tab2, tab3 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios e Insights"])

with tab1:
    st.header("Formulário do Paciente")
    col1, col2, col3 = st.columns(3)

    # Dicionários de Tradução (Visual -> Modelo)
    mapa_genero = {'Masculino': 'Female', 'Feminino': 'Male'} # Ajustado conforme seu notebook
    mapa_sim_nao = {'Sim': 'yes', 'Não': 'no'}
    mapa_frequencia = {'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always', 'Não': 'no'}
    mapa_transporte = {
        'Transporte Público': 'Public_Transportation', 'Caminhada': 'Walking', 
        'Carro': 'Automobile', 'Moto': 'Motorbike', 'Bicicleta': 'Bike'
    }

    with col1:
        genero_visual = st.selectbox("Gênero", list(mapa_genero.keys()))
        idade = st.number_input("Idade", 1, 120, 25)
        altura = st.number_input("Altura (m)", 0.5, 2.5, 1.70)
        peso = st.number_input("Peso (kg)", 10.0, 300.0, 70.0)
        historia_fam_visual = st.selectbox("Histórico Familiar de Sobrepeso?", list(mapa_sim_nao.keys()))

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
        if pipeline and le:
            # DataFrame com os nomes EXATOS das colunas do treinamento no notebook
            df_input = pd.DataFrame({
                'Genero': [mapa_genero[genero_visual]],
                'Idade': [idade],
                'Altura': [altura],
                'Peso': [peso],
                'Historico_Familiar_Obesidade': [mapa_sim_nao[historia_fam_visual]],
                'Frequencia_Consumo_Alimento_Calorico': [mapa_sim_nao[favc_visual]],
                'Frequencia_Consumo_Vegetais': [fcvc],
                'Numero_Refeicoes_Principais': [ncp],
                'Consumo_Alimento_Entre_Refeicoes': [mapa_frequencia[caec_visual]],
                'Fumante': [mapa_sim_nao[smoke_visual]],
                'Consumo_Agua': [ch2o],
                'Monitoramento_Calorico': [mapa_sim_nao[scc_visual]],
                'Frequencia_Atividade_Fisica': [faf],
                'Tempo_Uso_Tecnologia': [tue],
                'Consumo_Alcool': [mapa_frequencia[calc_visual]],
                'Meio_Transporte': [mapa_transporte[mtrans_visual]]
            })

            try:
                # 1. Predição numérica (XGBoost gera 0, 1, 2...)
                pred_codificada = pipeline.predict(df_input)
                # 2. Tradução para texto usando o LabelEncoder
                resultado_final = le.inverse_transform(pred_codificada)[0]
                
                imc = peso / (altura ** 2)
                st.success(f"### Resultado: {resultado_final.replace('_', ' ')}")
                st.info(f"**IMC Calculado:** {imc:.2f}")
            except Exception as e:
                st.error(f"Erro na predição: {e}")

import streamlit as st
import streamlit.components.v1 as components

# ... (restante do seu código)

with tab2:
    st.header("📊 Dashboard Analítico")
    
    # URL de incorporação oficial
    url_looker = "https://lookerstudio.google.com/embed/reporting/29f80ed0-090c-437e-a0e8-a3fd3b00e5be/page/2V5oF"
    
    # Renderização via iframe
    components.iframe(url_looker, height=700, scrolling=True)
    

















