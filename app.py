import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# 1. Configuração da Página
st.set_page_config(page_title="Predição de Obesidade", layout="wide")

# 2. Caminhos dos arquivos
MODEL_PATH = 'modelo_obesidade.pkl'
ENCODER_PATH = 'label_encoder.pkl' # Caso o encoder esteja separado

# 3. Função de Carregamento (Corrigida)
@st.cache_resource
def carregar_recursos():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Arquivo '{MODEL_PATH}' não encontrado.")
        return None, None
    
    try:
        dados = joblib.load(MODEL_PATH)
        
        # Se o .pkl contiver uma lista [modelo, encoder]
        if isinstance(dados, (list, tuple)) and len(dados) == 2:
            return dados[0], dados[1]
        
        # Se o .pkl contiver apenas o modelo, tenta carregar encoder separado
        pipeline = dados
        le = None
        if os.path.exists(ENCODER_PATH):
            le = joblib.load(ENCODER_PATH)
        
        return pipeline, le
    except Exception as e:
        st.error(f"Erro ao carregar recursos: {e}")
        return None, None

# 4. Chamada da Função (Linha onde ocorria o NameError corrigida)
pipeline, le = carregar_recursos()

# --- INTERFACE ---
st.title("🏥 Sistema de Apoio ao Diagnóstico de Obesidade")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios"])

with tab1:
    st.header("Formulário do Paciente")
    
    if pipeline is None:
        st.error("Modelo não carregado. Verifique o arquivo .pkl.")
    else:
        col1, col2, col3 = st.columns(3)

        mapa_genero = {'Masculino': 'Male', 'Feminino': 'Female'} 
        mapa_sim_nao = {'Sim': 'yes', 'Não': 'no'}
        mapa_frequencia = {'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always', 'Não': 'no'}
        mapa_transporte = {'Transporte Público': 'Public_Transportation', 'Caminhada': 'Walking', 'Carro': 'Automobile', 'Moto': 'Motorbike', 'Bicicleta': 'Bike'}

        with col1:
            genero = st.selectbox("Gênero", list(mapa_genero.keys()))
            idade = st.number_input("Idade", 1, 120, 25)
            altura = st.number_input("Altura (m)", 0.5, 2.5, 1.70)
            peso = st.number_input("Peso (kg)", 10.0, 300.0, 70.0)

        with col2:
            hist_fam = st.selectbox("Histórico Familiar de Sobrepeso?", list(mapa_sim_nao.keys()))
            favc = st.selectbox("Consome comida calórica frequentemente?", list(mapa_sim_nao.keys()))
            fcvc = st.slider("Frequência de vegetais (1-3)", 1, 3, 2)
            ncp = st.slider("Refeições principais", 1, 4, 3)

        with col3:
            caec = st.selectbox("Come entre refeições?", list(mapa_frequencia.keys()))
            faf = st.slider("Atividade física (0-3)", 0, 3, 1)
            calc = st.selectbox("Consumo de álcool", list(mapa_frequencia.keys()))
            mtrans = st.selectbox("Meio de transporte", list(mapa_transporte.keys()))

        if st.button("Realizar Diagnóstico"):
            try:
                # Criar DataFrame para predição
                df_input = pd.DataFrame({
                    'Gender': [mapa_genero[genero]], 'Age': [idade], 'Height': [altura], 'Weight': [peso],
                    'family_history_with_overweight': [mapa_sim_nao[hist_fam]], 'FAVC': [mapa_sim_nao[favc]],
                    'FCVC': [fcvc], 'NCP': [ncp], 'CAEC': [mapa_frequencia[caec]],
                    'FAF': [faf], 'CALC': [mapa_frequencia[calc]], 'MTRANS': [mapa_transporte[mtrans]],
                    # Adicione aqui outras colunas se o seu modelo exigir (ex: SMOKE, CH2O, SCC, TUE)
                })

                pred = pipeline.predict(df_input)
                
                # Traduzir resultado
                if le:
                    resultado = le.inverse_transform(pred)[0]
                else:
                    resultado = pred[0]

                st.success(f"### Resultado: {resultado}")
                st.info(f"**IMC:** {peso/(altura**2):.2f}")
            except Exception as e:
                st.error(f"Erro na predição: {e}")

with tab2:
    st.header("📊 Dashboard")
    fig = px.pie(names=['Normal', 'Sobrepeso', 'Obesidade'], values=[30, 40, 30], hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.link_button("🚀 Relatório Completo", "https://lookerstudio.google.com/...")
