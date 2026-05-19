# Sistema de Análise de Imagens Médicas (MVP)

Este é um sistema web funcional projetado para analisar imagens médicas, classificar potenciais anomalias e destacar as regiões de interesse detectadas através de algoritmos de processamento de imagem clássicos.

## Aviso Ético e Conformidade
**O sistema é APENAS UM SUPORTE DIAGNÓSTICO E UMA DEMONSTRAÇÃO DE ENGENHARIA. Não substitui, sob nenhuma circunstância, a avaliação de um profissional de saúde qualificado (médico radiologista ou especialista).** 
*   Este software não deve ser usado para diagnóstico clínico final.
*   Os modelos aqui presentes são simulados e treinados em dados sintéticos para fins de demonstração da arquitetura de software e viabilidade técnica.
*   Nenhum dado do paciente é armazenado de forma persistente. Metadados do arquivo (como o nome original) são removidos em tempo de execução para anonimização.

## Funcionalidades
- **Upload e Validação:** Suporta JPG, PNG e DCM. Upload de arquivos em lotes.
- **Pré-processamento:** Conversão para escala de cinza, normalização de intensidade, redimensionamento mantendo proporção (até 512x512) e remoção de ruídos (Filtro Gaussiano).
- **Análise & Segmentação:** Utiliza limiarização de Otsu e o algoritmo de Canny para segmentação e detecção de contornos.
- **Classificador e Métricas:** Extrai features da imagem para classificar usando Random Forest. Uma área exclusiva para métricas com Matriz de Confusão, ROC e F1-Score simulados é fornecida.
- **Interface e Exportação:** Construído com Streamlit, permite a visualização lado-a-lado (original vs. segmentado) e exportação do laudo em PDF.

## Como Instalar e Executar

1. **Requisitos:** Certifique-se de ter o Python 3.9+ instalado.
2. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Gerar Dados de Teste:**
   Antes de rodar o aplicativo, execute o script de geração para povoar o diretório com as amostras de teste:
   ```bash
   python generate_samples.py
   ```
4. **Rodar a Aplicação:**
   ```bash
   streamlit run app.py
   ```

## Limitações Conhecidas
- O Random Forest empregado é treinado com amostras sintéticas (mock). Em um cenário do mundo real, exigiria um dataset real e extenso curado por especialistas.
- O limiar de Otsu funciona bem com histogramas bimodais; tecidos complexos com variação extensa de contraste podem gerar falsos positivos nos contornos, afetando as features.
- DICOM é suportado apenas parcialmente; imagens médicas reais demandam um pipeline de metadados extenso para ser interpretado em toda sua totalidade (embora o código forneça ganchos via `pydicom`).

