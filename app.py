import streamlit as st
import pandas as pd
import joblib
import os

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
        st.error("Erro: Arquivos do modelo ou label encoder não encontrados.")
        return None, None
    modelo = joblib.load(MODEL_PATH)
    encoder = joblib.load(LE_PATH)
    return modelo, encoder

pipeline, le = carregar_recursos()

st.title("🏥 Sistema de Apoio ao Diagnóstico de Obesidade")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios"])

with tab1:
    st.header("Formulário do Paciente")
    col1, col2, col3 = st.columns(3)

    # Dicionários de Tradução (O modelo espera os valores em EN conforme o notebook)
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
        scc_visual = st.selectbox("Monitora calorias?", list(mapa_sim_nao.keys()))
        faf = st.slider("Frequência de atividade física (0-3)", 0, 3, 1)
        tue = st.slider("Tempo usando dispositivos (0-2)", 0, 2, 1)
        calc_visual = st.selectbox("Consumo de álcool", list(mapa_frequencia.keys()))
        mtrans_visual = st.selectbox("Meio de transporte", list(mapa_transporte.keys()))

    if st.button("Realizar Diagnóstico"):
        if pipeline and le:
            # Criação do DataFrame com os nomes EXATOS das colunas do seu notebook
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
                # Predição numérica (XGBoost)
                pred_numerica = pipeline.predict(df_input)
                # Tradução para texto usando o LabelEncoder salvo no notebook
                resultado_texto = le.inverse_transform(pred_numerica)[0]
                
                st.success(f"### Nível de Obesidade Previsto: {resultado_texto}")
                st.info(f"IMC Calculado: {peso / (altura ** 2):.2f}")
            except Exception as e:
                st.error(f"Erro na predição: {e}")