# ... (mantenha as importações e configurações iniciais iguais)

with tab1:
    st.header("Formulário do Paciente")
    col1, col2, col3 = st.columns(3)

    # Dicionários de Tradução (Visual -> Modelo)
    # Corrigido: Masculino -> Male, Feminino -> Female
    mapa_genero = {'Masculino': 'Male', 'Feminino': 'Female'} 
    mapa_sim_nao = {'Sim': 'yes', 'Não': 'no'}
    mapa_frequencia = {'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always', 'Não': 'no'}
    mapa_transporte = {
        'Transporte Público': 'Public_Transportation', 'Caminhada': 'Walking', 
        'Carro': 'Automobile', 'Moto': 'Motorbike', 'Bicicleta': 'Bike'
    }

    # ... (mantenha os inputs das colunas col1, col2, col3 iguais)

    if st.button("Realizar Diagnóstico"):
        if pipeline and le:
            # Certifique-se de que os nomes das colunas abaixo são EXATAMENTE 
            # os mesmos usados no df.fit() do seu modelo
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
                
                imc = peso / (altura ** 2)
                st.success(f"### Resultado: {resultado_final.replace('_', ' ')}")
                st.info(f"**IMC Calculado:** {imc:.2f}")
            except Exception as e:
                st.error(f"Erro na predição: {e}")

with tab2:
    # ... (mantenha seu código do Dashboard igual, ele está correto)
    st.header("📊 Dashboard Analítico")
    # (seu código de métricas e gráficos aqui)

with tab3:
    st.header("📝 Relatórios e Insights")
    # Centralizado dentro da tab3
    st.components.v1.iframe(
        "https://lookerstudio.google.com/embed/reporting/29f80ed0-090c-437e-a0e8-a3fd3b00e5be/page/2V5oF",
        height=700,
        scrolling=True
    )
