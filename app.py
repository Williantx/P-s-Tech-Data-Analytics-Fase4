import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# Configuração da Página
st.set_page_config(page_title="Predição de Obesidade", layout="wide")

# Caminhos locais
MODEL_PATH = 'modelo_obesidade.pkl'

# --- FUNÇÃO DE CARREGAMENTO CORRIGIDA ---
@st.cache_resource
def carregar_recursos():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Erro: Arquivo '{MODEL_PATH}' não encontrado.")
        return None, None
    
    try:
        recursos = joblib.load(MODEL_PATH)
        # Verifica se o arquivo contém o par (pipeline, encoder)
        if isinstance(recursos, (tuple, list)) and len(recursos) == 2:
            return recursos[0], recursos[1]
        else:
            st.warning("O arquivo .pkl não contém (modelo, encoder). Retornando apenas o modelo.")
            return recursos, None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None, None

# Chamada única para carregar os objetos
pipeline, le = carregar_recursos()

# Título
st.title("🏥 Sistema de Apoio ao Diagnóstico de Obesidade")
st.markdown("---")

# Abas
tab1, tab2, tab3 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios e Insights"])

with tab1:
    st.header("Formulário do Paciente")
    col1, col2, col3 = st.columns(3)

    mapa_genero = {'Masculino': 'Male', 'Feminino': 'Female'} 
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
        if pipeline is not None and le is not None:
            df_input = pd.DataFrame({
                'Gender': [mapa_genero[genero_visual]],
                'Age': [idade],
                'Height': [altura],
                'Weight': [peso],
                'family_history_with_overweight': [mapa_sim_nao[historia_fam_visual]],
                'FAVC': [mapa_sim_nao[favc_visual]],
                'FCVC': [fcvc],
                'NCP': [ncp],
                'CAEC': [mapa_frequencia[caec_visual]],
                'SMOKE': [mapa_sim_nao[smoke_visual]],
                'CH2O': [ch2o],
                'SCC': [mapa_sim_nao[scc_visual]],
                'FAF': [faf],
                'TUE': [tue],
                'CALC': [mapa_frequencia[calc_visual]],
                'MTRANS': [mapa_transporte[mtrans_visual]]
            })

            try:
                pred_codificada = pipeline.predict(df_input)
                resultado_raw = le.inverse_transform(pred_codificada)[0]

                def normalize(level):
                    traducoes = {
                        'Insufficient_Weight': "Abaixo do peso",
                        'Normal_Weight': "Peso normal",
                        'Overweight_Level_I': "Sobrepeso",
                        'Overweight_Level_II': "Sobrepeso",
                        'Obesity_Type_I': "Obeso Grau I",
                        'Obesity_Type_II': "Obeso Grau II",
                        'Obesity_Type_III': "Obeso Grau III"
                    }
                    return traducoes.get(level, "Obeso")

                resultado_final = normalize(resultado_raw)
                imc = peso / (altura ** 2)

                st.success(f"### Resultado: {resultado_final}")
                st.info(f"**Classificação Detalhada:** {resultado_raw.replace('_', ' ')}")
                st.info(f"**IMC Calculado:** {imc:.2f}")

            except Exception as e:
                st.error(f"Erro na predição: {e}")
        else:
            st.error("O modelo ou o encoder não foram carregados corretamente.")

with tab2:
    st.header("📊 Dashboard Analítico")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pacientes Analisados", "2.111")
    c2.metric("Peso Médio", "86,59 kg")
    c3.metric("Idade Média", "24 anos")
    
    st.markdown("---")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Distribuição de Obesidade")
        df_dist = pd.DataFrame({
            "Categoria": ['Obesidade I', 'Obesidade III', 'Obesidade II', 'Sobrepeso II', 'Sobrepeso I', 'Peso Normal', 'Abaixo do Peso'],
            "Valores": [16.6, 15.3, 14.1, 13.7, 13.7, 13.6, 12.9]
        })
        fig_pizza = px.pie(df_dist, names='Categoria', values='Valores', hole=0.4)
        st.plotly_chart(fig_pizza, use_container_width=True)
        
    with col_g2:
        st.subheader("Histórico Familiar vs Obesidade")
        df_hist = pd.DataFrame({"Histórico": ["Sim", "Não"], "Quantidade": [1750, 400]})
        fig_hist = px.bar(df_hist, x="Histórico", y="Quantidade", color="Histórico")
        st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.header("📝 Relatórios e Insights")
    st.link_button("🚀 Abrir Relatório Completo no Looker Studio", 
                   "https://lookerstudio.google.com/u/0/reporting/29f80ed0-090c-437e-a0e8-a3fd3b00e5be/page/2V5oF")
    st.markdown("---")
    st.subheader("Visualização Rápida")
    st.components.v1.iframe(
        "https://lookerstudio.google.com/embed/reporting/29f80ed0-090c-437e-a0e8-a3fd3b00e5be/page/2V5oF",
        height=700, scrolling=True
    )
