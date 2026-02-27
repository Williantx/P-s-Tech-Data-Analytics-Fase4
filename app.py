import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit.components.v1 as components
import plotly.express as px

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

    # Dicionários de Tradução (Ajustado: Masculino -> Male, Feminino -> Female)
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
        if pipeline and le:
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
                # Predição
                pred_codificada = pipeline.predict(df_input)
                resultado_raw = le.inverse_transform(pred_codificada)[0]

                # Lógica de Normalização Integrada
                def normalize(level):
                    if level == 'Insufficient_Weight':
                        return "Abaixo do peso"
                    elif level == 'Normal_Weight':
                        return "Peso normal"
                    elif level in ['Overweight_Level_I', 'Overweight_Level_II']:
                        return "Sobrepeso"
                    else:
                        return "Obeso"

                resultado_final = normalize(resultado_raw)
                imc = peso / (altura ** 2)

                # Exibição
                st.success(f"### Resultado: {resultado_final}")
                st.info(f"**Classificação Detalhada:** {resultado_raw.replace('_', ' ')}")
                st.info(f"**IMC Calculado:** {imc:.2f}")

            except Exception as e:
                st.error(f"Erro na predição: {e}")



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
        fig_pizza = px.pie(df_dist, names='Categoria', values='Valores', hole=0.4,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pizza, use_container_width=True)
        
    with col_g2:
        st.subheader("Histórico Familiar vs Obesidade")
        df_hist = pd.DataFrame({"Histórico": ["Sim", "Não"], "Quantidade": [1750, 400]})
        fig_hist = px.bar(df_hist, x="Histórico", y="Quantidade", color="Histórico",
                         color_discrete_map={"Sim": "#ef553b", "Não": "#636efa"})
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Meios de Transporte e Sedentarismo")
    df_transp = pd.DataFrame({
        'Meio': ['Transporte Público', 'Automóvel', 'Caminhada', 'Bicicleta', 'Motocicleta'],
        'Qtd': [1558, 463, 88, 14, 9]
    })
    fig_transp = px.bar(df_transp, x='Meio', y='Qtd', color='Meio', text_auto=True)
    st.plotly_chart(fig_transp, use_container_width=True)

with tab3:
    st.header("📝 Relatórios e Insights")
    
    # Botão de link direto
    st.link_button("🚀 Abrir Relatório Completo no Looker Studio", 
                   "https://lookerstudio.google.com/u/0/reporting/29f80ed0-090c-437e-a0e8-a3fd3b00e5be/page/2V5oF")

    st.markdown("---")
    
    # Mantendo o iframe caso queira que o usuário visualize sem sair da página
    st.subheader("Visualização Rápida")
    st.components.v1.iframe(
        "https://lookerstudio.google.com/embed/reporting/29f80ed0-090c-437e-a0e8-a3fd3b00e5be/page/2V5oF",
        height=700,
        scrolling=True
    )




