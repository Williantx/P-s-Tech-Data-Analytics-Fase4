import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import streamlit.components.v1 as components

# 1. Configuração da Página
st.set_page_config(page_title="Sistema de Diagnóstico - Obesidade", layout="wide")

# Caminhos dos arquivos (Certifique-se de que estão no seu GitHub)
MODEL_PATH = 'modelo_obesidade.pkl'
LE_PATH = 'label_encoder.pkl'
DATA_PATH = 'Obesity.csv'

# 2. Função para carregar o modelo e o encoder
@st.cache_resource
def carregar_recursos():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        st.error("Erro: Arquivos 'modelo_obesidade.pkl' ou 'label_encoder.pkl' não encontrados no repositório.")
        return None, None
    try:
        modelo = joblib.load(MODEL_PATH)
        encoder = joblib.load(LE_PATH)
        return modelo, encoder
    except Exception as e:
        st.error(f"Erro técnico ao carregar recursos: {e}")
        return None, None

pipeline, le = carregar_recursos()

# 3. Cabeçalho Principal
st.title("🏥 Sistema de Apoio ao Diagnóstico de Obesidade")
st.subheader("Hospital Vita Nova - Clínica de Diagnóstico") [cite: 3]
st.markdown("---")

# Definição das Abas
tab1, tab2, tab3 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios e Insights"])

# --- TAB 1: FORMULÁRIO E PREDIÇÃO ---
with tab1:
    st.header("Formulário do Paciente")
    col1, col2, col3 = st.columns(3)

    # Dicionários de Tradução (Visual PT -> Modelo EN)
    mapa_genero = {'Masculino': 'Female', 'Feminino': 'Male'} 
    mapa_sim_nao = {'Sim': 'yes', 'Não': 'no'}
    mapa_frequencia = {'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always', 'Não': 'no'}
    mapa_transporte = {
        'Transporte Público': 'Public_Transportation', 'Caminhada': 'Walking', 
        'Carro': 'Automobile', 'Moto': 'Motorbike', 'Bicicleta': 'Bike'
    }

    with col1:
        genero_v = st.selectbox("Gênero", list(mapa_genero.keys()))
        idade = st.number_input("Idade", 1, 120, 24) [cite: 28]
        altura = st.number_input("Altura (m)", 0.5, 2.5, 1.70)
        peso = st.number_input("Peso (kg)", 10.0, 300.0, 86.59) [cite: 27]
        hist_fam = st.selectbox("Histórico Familiar de Sobrepeso?", list(mapa_sim_nao.keys()))

    with col2:
        favc = st.selectbox("Consome comida calórica frequentemente?", list(mapa_sim_nao.keys()))
        fcvc = st.slider("Frequência de consumo de vegetais (1-3)", 1, 3, 2)
        ncp = st.slider("Número de refeições principais", 1, 4, 3)
        caec = st.selectbox("Come entre refeições?", list(mapa_frequencia.keys()))
        smoke = st.selectbox("Fumante?", list(mapa_sim_nao.keys()))

    with col3:
        ch2o = st.slider("Consumo de água diário (1-3L)", 1, 3, 2)
        scc = st.selectbox("Monitora calorias ingeridas?", list(mapa_sim_nao.keys()))
        faf = st.slider("Frequência de atividade física (0-3)", 0, 3, 1) # Linha corrigida
        tue = st.slider("Tempo usando dispositivos (0-2)", 0, 2, 1)
        calc = st.selectbox("Consumo de álcool", list(mapa_frequencia.keys()))
        mtrans = st.selectbox("Meio de transporte principal", list(mapa_transporte.keys()))

    if st.button("Realizar Diagnóstico"):
        if pipeline and le:
            df_input = pd.DataFrame({
                'Genero': [mapa_genero[genero_v]],
                'Idade': [idade],
                'Altura': [altura],
                'Peso': [peso],
                'Historico_Familiar_Obesidade': [mapa_sim_nao[hist_fam]],
                'Frequencia_Consumo_Alimento_Calorico': [mapa_sim_nao[favc]],
                'Frequencia_Consumo_Vegetais': [fcvc],
                'Numero_Refeicoes_Principais': [ncp],
                'Consumo_Alimento_Entre_Refeicoes': [mapa_frequencia[caec]],
                'Fumante': [mapa_sim_nao[smoke]],
                'Consumo_Agua': [ch2o],
                'Monitoramento_Calorico': [mapa_sim_nao[scc]],
                'Frequencia_Atividade_Fisica': [faf],
                'Tempo_Uso_Tecnologia': [tue],
                'Consumo_Alcool': [mapa_frequencia[calc]],
                'Meio_Transporte': [mapa_transporte[mtrans]]
            })

            try:
                pred = pipeline.predict(df_input)
                resultado = le.inverse_transform(pred)[0]
                imc = peso / (altura ** 2)
                st.success(f"### Resultado: {resultado.replace('_', ' ')}")
                st.info(f"**IMC Calculado:** {imc:.2f}")
            except Exception as e:
                st.error(f"Erro na predição: {e}")

# --- TAB 2: DASHBOARD NATIVO (PLOTLY) ---
with tab2:
    st.header("📊 Indicadores da Clínica")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Pacientes Analisados", "2.111") [cite: 25]
    m2.metric("Peso Médio", "86,59 kg") [cite: 27]
    m3.metric("Idade Média", "24 anos") [cite: 28]
    
    st.markdown("---")
    
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Distribuição de Obesidade")
        fig_p = px.pie(
            names=['Obesidade I', 'Obesidade III', 'Obesidade II', 'Sobrepeso II', 'Sobrepeso I', 'Peso Normal', 'Abaixo do Peso'],
            values=[16.6, 15.3, 14.1, 13.7, 13.7, 13.6, 12.9], [cite: 5, 6, 7, 8, 9, 10, 11]
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_p, use_container_width=True)
        
    with g2:
        st.subheader("Transporte e Sedentarismo")
        d_transp = {'Meio': ['Público', 'Automóvel', 'Caminhada'], 'Qtd': [1558, 463, 88]} [cite: 48, 53, 62]
        fig_t = px.bar(d_transp, x='Meio', y='Qtd', color='Meio', text_auto=True)
        st.plotly_chart(fig_t, use_container_width=True)

# --- TAB 3: RELATÓRIO LOOKER STUDIO ---
with tab3:
    st.header("📝 Relatório Detalhado (Looker Studio)")
    
    looker_html = """
    <iframe width="100%" height="600" 
        src="https://lookerstudio.google.com/embed/reporting/29f80ed0-090c-437e-a0e8-a3fd3b00e5be/page/2V5oF" 
        frameborder="0" style="border:0" allowfullscreen 
        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
    </iframe>
    """
    components.html(looker_html, height=620)
    
    st.info("💡 **Insight:** O histórico familiar é o fator com maior correlação positiva nos casos de obesidade nível III.") [cite: 30]
