import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

from utils import validate_image_file, anonymize_metadata
from preprocess import load_image, preprocess_image
from segment import segment_image, extract_features, draw_contours_on_image
from classify import AnomalyClassifier
from cnn_model import CNNClassifier
from metrics import generate_evaluation_metrics, export_to_pdf

# Configuração da página Streamlit (deve ser a primeira chamada)
st.set_page_config(page_title="Análise Médica com IA", layout="wide", page_icon="🩺")

# Estilização CSS Premium (Glassmorphism e Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@500;700&display=swap');
    
    :root {
        --primary-accent: #0ea5e9;
        --secondary-accent: #8b5cf6;
        --danger-color: #ef4444;
        --danger-glow: rgba(239, 68, 68, 0.4);
        --success-color: #10b981;
        --success-glow: rgba(16, 185, 129, 0.4);
        --surface-color: rgba(30, 41, 59, 0.7);
        --text-color: #f8fafc;
        --bg-color: #020617;
    }
    
    /* Global e Typography */
    .stApp {
        background-color: var(--bg-color);
        background-image: radial-gradient(circle at 15% 50%, rgba(14, 165, 233, 0.05), transparent 25%),
                          radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.05), transparent 25%);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }
    
    /* Ocultar elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Header */
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.2rem;
        margin-bottom: 0.2rem;
        text-align: center;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 3rem;
    }
    
    /* Disclaimer */
    .disclaimer-box {
        background: rgba(239, 68, 68, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-left: 4px solid var(--danger-color);
        padding: 1.2rem;
        margin-bottom: 2rem;
        border-radius: 8px;
        color: #fca5a5;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Metric Cards (Glassmorphism) */
    .metric-card {
        background: var(--surface-color);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.7);
    }
    
    .metric-card h3 {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        color: #cbd5e1;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value-normal { 
        color: var(--success-color); 
        font-size: 2.2rem; 
        font-weight: 800;
        text-shadow: 0 0 20px var(--success-glow);
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-value-anomaly { 
        color: var(--danger-color); 
        font-size: 2.2rem; 
        font-weight: 800;
        text-shadow: 0 0 20px var(--danger-glow);
        font-family: 'Outfit', sans-serif;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { text-shadow: 0 0 10px var(--danger-glow); }
        50% { text-shadow: 0 0 25px rgba(239, 68, 68, 0.7); }
        100% { text-shadow: 0 0 10px var(--danger-glow); }
    }
    
    /* Fixar a caixa de upload e botões para acompanhar o visual dark */
    div[data-testid="stFileUploader"] {
        border-radius: 12px;
        border: 1px dashed rgba(255, 255, 255, 0.2);
        background: rgba(15, 23, 42, 0.5);
    }
    
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }

    /* Responsividade */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem !important;
        }
        .sub-header {
            font-size: 0.9rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Cache do modelo Clássico
@st.cache_resource
def load_rf_classifier():
    classifier = AnomalyClassifier()
    classifier.train_synthetic()
    return classifier

# Cache do modelo CNN
@st.cache_resource
def load_cnn_classifier():
    classifier = CNNClassifier()
    return classifier

rf_model = load_rf_classifier()
cnn_model = load_cnn_classifier()

if 'results' not in st.session_state:
    st.session_state.results = []
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Eu sou o assistente do sistema. Me faça perguntas sobre o funcionamento, arquitetura ou algoritmos deste MVP baseadas no nosso README!"}
    ]

def get_readme_context(query):
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
            
        # Uma busca rudimentar por palavras-chave na query
        keywords = re.findall(r'\w+', query.lower())
        relevant_paragraphs = []
        
        for paragraph in content.split('\n\n'):
            # Conta quantas palavras-chave aparecem no parágrafo
            match_count = sum(1 for kw in keywords if kw in paragraph.lower() and len(kw) > 3)
            if match_count > 0:
                relevant_paragraphs.append((match_count, paragraph))
                
        if relevant_paragraphs:
            # Ordena pelos que tem mais matches
            relevant_paragraphs.sort(key=lambda x: x[0], reverse=True)
            top_matches = [p[1] for p in relevant_paragraphs[:2]] # Pega os 2 melhores parágrafos
            return "Com base na documentação do sistema, encontrei o seguinte:\n\n" + "\n\n".join(top_matches)
        else:
            return "Desculpe, não encontrei informações específicas sobre isso no nosso README. Tente perguntar sobre bibliotecas, métricas, dicom ou arquitetura!"
            
    except Exception as e:
        return "Erro ao ler a base de conhecimento (README.md)."

def main():
    st.markdown('<h1 class="main-header">✨ Sistema de Análise Médica (MVP)</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Plataforma Inteligente para Triagem Radiológica em Lote usando Deep Learning e Visão Computacional</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ AVISO ÉTICO E CLÍNICO IMPORTANTÍSSIMO:</strong><br>
        Este sistema é <strong>apenas um suporte de processamento de imagem e avaliação computacional preliminar</strong>. 
        Sob hipótese alguma substitui o diagnóstico ou aconselhamento de um médico qualificado.
        Os resultados apresentados são aproximações baseadas em algoritmos clássicos de limiarização e modelos sintéticos não validados clinicamente.
    </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_metrics, tab_about, tab_assistant = st.tabs(["Análise de Imagem", "Métricas do Modelo Clássico", "Sobre o Sistema", "🤖 Assistente Virtual"])

    with tab_upload:
        st.subheader("1. Configuração e Exame")
        
        model_choice = st.radio(
            "Escolha o Motor Analítico:",
            options=["Machine Learning Clássico (Otsu + Canny + RF)", "Deep Learning (ResNet50 + Grad-CAM)"]
        )
        
        uploaded_files = st.file_uploader(
            "Selecione as imagens em Lote (JPG, PNG ou DCM)", 
            type=['jpg', 'jpeg', 'png', 'dcm'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Executar Análise em Lote", type="primary"):
                st.session_state.results = []
                # Barra de progresso para UX
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processando imagem {i+1} de {len(uploaded_files)}: {uploaded_file.name}...")
                    
                    is_valid, msg = validate_image_file(uploaded_file)
                    if not is_valid:
                        st.error(f"Erro de validação em {uploaded_file.name}: {msg}")
                        continue
                    
                    anon_id = anonymize_metadata(uploaded_file.name)
                    
                    try:
                        original_image = load_image(uploaded_file)
                        rgb_resized, gray_processed = preprocess_image(original_image)
                        mask, edges, contours = segment_image(gray_processed)
                        features, valid_contours = extract_features(contours)
                        
                        if "Clássico" in model_choice:
                            processed_display = draw_contours_on_image(rgb_resized, valid_contours)
                            is_anomaly, confidence = rf_model.predict(features)
                            model_used = "RF"
                        else:
                            is_anomaly, confidence = cnn_model.predict(rgb_resized)
                            heatmap_display = cnn_model.get_gradcam(rgb_resized)
                            processed_display = heatmap_display
                            model_used = "CNN"
                            
                        st.session_state.results.append({
                            "anon_id": anon_id,
                            "original_image": original_image,
                            "rgb_resized": rgb_resized,
                            "processed_display": processed_display,
                            "features": features,
                            "is_anomaly": is_anomaly,
                            "confidence": confidence,
                            "model_used": model_used
                        })
                    except Exception as e:
                        st.error(f"Ocorreu um erro ao tentar processar {uploaded_file.name}.")
                        st.exception(e)
                    
                    # Atualiza a barra
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Processamento em lote concluído!")

        if st.session_state.results:
            st.divider()
            st.subheader(f"2. Resultados da Análise ({len(st.session_state.results)} imagens processadas)")
            
            for i, result in enumerate(st.session_state.results):
                status_label = "ANOMALIA DETECTADA" if result['is_anomaly'] else "NORMAL"
                
                with st.expander(f"📄 Resultado: Paciente ID {result['anon_id']} - {status_label}", expanded=True):
                    col_res1, col_res2, col_res3 = st.columns([1, 1, 1])
                    
                    with col_res1:
                        st.image(result['original_image'], use_container_width=True, caption="Imagem Original")
                        
                    with col_res2:
                        caption_text = "Contornos em Vermelho" if result['model_used'] == "RF" else "Grad-CAM (Mapa de Calor)"
                        st.image(result['processed_display'], use_container_width=True, caption=caption_text)
                        
                    with col_res3:
                        anomaly_flag = result['is_anomaly']
                        conf_val = result['confidence']
                        
                        if anomaly_flag:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>Diagnóstico Sugerido</h3>
                                <div class="metric-value-anomaly">ANOMALIA</div>
                                <p>Confiança ({result['model_used']}): <strong>{conf_val:.2f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>Diagnóstico Sugerido</h3>
                                <div class="metric-value-normal">NORMAL</div>
                                <p>Confiança ({result['model_used']}): <strong>{conf_val:.2f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Bônus: Exportação de PDF
                        pdf_bytes = export_to_pdf(
                            result['rgb_resized'], 
                            result['processed_display'], 
                            result['features'], 
                            result['is_anomaly'], 
                            result['confidence']
                        )
                        
                        st.download_button(
                            label=f"📄 Baixar Laudo ({result['anon_id']})",
                            data=pdf_bytes,
                            file_name=f"laudo_{result['anon_id']}.pdf",
                            mime="application/pdf",
                            type="secondary",
                            key=f"download_pdf_{i}"
                        )


    with tab_metrics:
        st.subheader("Validação de Desempenho do Modelo (RF - Dados Sintéticos)")
        st.write("Estas métricas referem-se ao modelo Clássico Random Forest (treinado simuladamente).")
        
        st.info("""
        **Por que os valores são 1.000 (100%)?**
        
        Em um cenário clínico do mundo real, métricas perfeitas são extremamente raras e geralmente indicam *overfitting*. 
        No entanto, como este sistema é um **MVP** para fins educacionais e de demonstração da arquitetura (sem um banco de dados real de 100 mil imagens), nós geramos **dados sintéticos** no código.
        
        A regra matemática para criar esses dados é fixa e perfeitamente separável (ex: mais contornos + área maior = classe 1). O algoritmo de Machine Learning aprende essa regra matemática artificial com total facilidade, o que resulta em um acerto de 100%. 
        A presença deste painel atende aos requisitos de validação do projeto, provando que a engenharia de matrizes de confusão e curva ROC está devidamente implementada.
        """)
        
        with st.spinner("Computando matrizes de confusão e AUC..."):
            metrics = generate_evaluation_metrics(rf_model.model)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Precision (Precisão)", f"{metrics['precision']:.3f}")
            m_col2.metric("Recall (Sensibilidade)", f"{metrics['recall']:.3f}")
            m_col3.metric("F1-Score", f"{metrics['f1_score']:.3f}")
            
            st.subheader("Curva ROC (Receiver Operating Characteristic)")
            fig, ax = plt.subplots(figsize=(8, 4))
            fpr, tpr, roc_auc = metrics['roc_curve']
            
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#1e293b')
            
            ax.plot(fpr, tpr, color='#3b82f6', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='#ef4444', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', color='white')
            ax.set_ylabel('True Positive Rate', color='white')
            ax.set_title('Receiver Operating Characteristic', color='white')
            ax.legend(loc="lower right")
            
            st.pyplot(fig)
            
    with tab_about:
        st.subheader("Arquitetura e Funcionalidades do Sistema")
        st.write("""
        Este sistema foi projetado como um MVP (Minimum Viable Product) para auxiliar na análise de imagens médicas usando técnicas de Machine Learning Clássico e Deep Learning.
        
        ### Funcionalidades Principais:
        *   **Upload em Lote (Batch Processing):** Permite ao usuário enviar múltiplas imagens simultaneamente. O sistema processa a fila sequencialmente com feedback visual de progresso e renderiza resultados individuais em abas isoladas.
        *   **Suporte a DICOM:** Integração nativa com a biblioteca `pydicom` para leitura de imagens médicas no formato padrão, com conversão automatizada para espectro RGB analisável.
        *   **Proteção de Dados (Anonimização):** Em conformidade com a LGPD e boas práticas médicas, o sistema remove todos os metadados identificáveis do paciente (como nome e IDs) antes de rodar os modelos.
        *   **Exportação de Laudo (PDF):** Geração de relatório automatizado em PDF contendo as imagens originais, os mapas processados, atributos detectados e a sugestão de diagnóstico.
        
        ### Motores Analíticos:
        *   **Motor Clássico (Morphological RF):** Utiliza filtros clássicos de visão computacional (Limiarização de Otsu e Filtro de Canny) para destacar contornos e treinar um classificador Random Forest com base em atributos físicos (Área, Perímetro).
        *   **Motor Deep Learning (ResNet50):** Utiliza Transfer Learning com uma arquitetura de Rede Neural Convolucional avançada (ResNet50) para detectar anomalias sutis através de extração de features em múltiplas camadas.
        *   **Visualização Grad-CAM:** Mapa de Calor de Ativação acoplado à CNN. Ele rastreia matematicamente quais pixels e áreas na imagem mais influenciaram a rede neural a dar o diagnóstico final de anomalia.
        """)
        
    with tab_assistant:
        st.subheader("🤖 Assistente de Dúvidas do MVP")
        st.write("Tire dúvidas sobre arquitetura, requisitos e funcionamento do sistema. As respostas são baseadas estritamente na documentação (README.md).")
        
        # Container para exibir o histórico de mensagens
        chat_container = st.container(height=400)
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Caixa de entrada do usuário
        if prompt := st.chat_input("Pergunte algo (ex: 'Quais bibliotecas foram usadas?'):"):
            # Adiciona mensagem do usuário ao estado
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # Gera a resposta via RAG simulado
            response = get_readme_context(prompt)
            
            # Adiciona resposta ao estado e exibe
            st.session_state.messages.append({"role": "assistant", "content": response})
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)

if __name__ == "__main__":
    main()
