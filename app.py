import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# ==============================
# CONFIGURAÇÃO DA PÁGINA
# ==============================
st.set_page_config(page_title="Predição de Obesidade", layout="wide")

MODEL_PATH = 'modelo_obesidade.pkl'
LE_PATH = 'label_encoder.pkl'
DATA_PATH = 'Obesity.csv'

# ==============================
# CARREGAR MODELO E ENCODER
# ==============================
@st.cache_resource
def carregar_recursos():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        return None, None
    modelo = joblib.load(MODEL_PATH)
    encoder = joblib.load(LE_PATH)
    return modelo, encoder

pipeline, le = carregar_recursos()

# ==============================
# TÍTULO
# ==============================
st.title("🏥 Sistema de Apoio ao Diagnóstico de Obesidade")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios e Insights"]
)

# =========================================================
# 🔮 TAB 1 - PREDIÇÃO
# =========================================================
with tab1:
    st.header("Formulário do Paciente")
    col1, col2, col3 = st.columns(3)

    # Mapas
    mapa_genero = {'Masculino': 'Male', 'Feminino': 'Female'}
    mapa_sim_nao = {'Sim': 'yes', 'Não': 'no'}
    mapa_frequencia = {
        'Às vezes': 'Sometimes',
        'Frequentemente': 'Frequently',
        'Sempre': 'Always',
        'Não': 'no'
    }
    mapa_transporte = {
        'Transporte Público': 'Public_Transportation',
        'Caminhada': 'Walking',
        'Carro': 'Automobile',
        'Moto': 'Motorbike',
        'Bicicleta': 'Bike'
    }

    with col1:
        genero_visual = st.selectbox("Gênero", list(mapa_genero.keys()))
        idade = st.number_input("Idade", 1, 120, 25)
        altura = st.number_input("Altura (m)", 0.5, 2.5, 1.70)
        peso = st.number_input("Peso (kg)", 10.0, 300.0, 70.0)
        historia_fam_visual = st.selectbox(
            "Histórico Familiar de Sobrepeso?",
            list(mapa_sim_nao.keys())
        )

    with col2:
        favc_visual = st.selectbox(
            "Consome comida calórica frequentemente?",
            list(mapa_sim_nao.keys())
        )
        fcvc = st.slider("Frequência de consumo de vegetais (1-3)", 1, 3, 2)
        ncp = st.slider("Número de refeições principais", 1, 4, 3)
        caec_visual = st.selectbox(
            "Come entre refeições?",
            list(mapa_frequencia.keys())
        )
        smoke_visual = st.selectbox("Fumante?", list(mapa_sim_nao.keys()))

    with col3:
        ch2o = st.slider("Consumo de água diário (1-3L)", 1, 3, 2)
        scc_visual = st.selectbox(
            "Monitora calorias ingeridas?",
            list(mapa_sim_nao.keys())
        )
        faf = st.slider("Frequência de atividade física (0-3)", 0, 3, 1)
        tue = st.slider("Tempo usando dispositivos (0-2)", 0, 2, 1)
        calc_visual = st.selectbox(
            "Consumo de álcool",
            list(mapa_frequencia.keys())
        )
        mtrans_visual = st.selectbox(
            "Meio de transporte principal",
            list(mapa_transporte.keys())
        )

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
                pred_codificada = pipeline.predict(df_input)
                resultado_final = le.inverse_transform(pred_codificada)[0]

                # IMC
                if altura > 0:
                    imc = peso / (altura ** 2)
                else:
                    imc = 0

                # Classificação IMC
                if imc < 18.5:
                    class_imc = "Abaixo do Peso"
                elif imc < 25:
                    class_imc = "Peso Normal"
                elif imc < 30:
                    class_imc = "Sobrepeso"
                else:
                    class_imc = "Obesidade"

                st.success(f"### Resultado: {resultado_final.replace('_', ' ')}")
                st.info(f"IMC Calculado: {imc:.2f} ({class_imc})")

                # Probabilidades
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(df_input)[0]
                    classes = le.classes_

                    df_proba = pd.DataFrame({
                        "Classificação": classes,
                        "Probabilidade": proba
                    })

                    fig_proba = px.bar(
                        df_proba,
                        x="Classificação",
                        y="Probabilidade",
                        text_auto=True
                    )

                    st.plotly_chart(fig_proba, use_container_width=True)

            except Exception as e:
                st.error(f"Erro na predição: {e}")

        else:
            st.error("Modelo ou Encoder não carregado.")


# =========================================================
# 📊 TAB 2 - DASHBOARD
# =========================================================
with tab2:
    st.header("📊 Dashboard Analítico")

    try:
        df = pd.read_csv(DATA_PATH)

        c1, c2, c3 = st.columns(3)
        c1.metric("Pacientes Analisados", len(df))
        c2.metric("Peso Médio", f"{df['Weight'].mean():.2f} kg")
        c3.metric("Idade Média", f"{df['Age'].mean():.0f} anos")

        st.markdown("---")

        # Distribuição da variável alvo
        fig_dist = px.pie(
            df,
            names='NObeyesdad',
            hole=0.4
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    except:
        st.warning("Arquivo CSV não encontrado para dashboard dinâmico.")


# =========================================================
# 📝 TAB 3 - RELATÓRIO
# =========================================================
with tab3:
    st.header("📝 Relatórios e Insights")

    st.components.v1.iframe(
        "https://lookerstudio.google.com/embed/reporting/29f80ed0-090c-437e-a0e8-a3fd3b00e5be/page/2V5oF",
        height=700
    )
