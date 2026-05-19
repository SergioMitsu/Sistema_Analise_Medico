# 🏥 Sistema de Análise de Imagens Médicas (MVP)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-blue.svg)]()

> **Sistema web funcional para análise de imagens médicas, classificação de anomalias e segmentação de regiões de interesse utilizando processamento clássico de imagem e aprendizado de máquina.**

---

## 📑 Índice

- [Aviso Ético e Conformidade](#-aviso-ético-e-conformidade)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Arquitetura do Sistema](#-arquitetura-do-sistema)
- [Como Instalar e Executar](#-como-instalar-e-executar)
- [Métricas de Validação](#-métricas-de-validação)
- [Limitações Conhecidas](#-limitações-conhecidas)
- [Referências](#-referências)

---

## ⚠️ Aviso Ético e Conformidade

> **🚨 O sistema é APENAS UM SUPORTE DIAGNÓSTICO E UMA DEMONSTRAÇÃO DE ENGENHARIA. Não substitui, sob nenhuma circunstância, a avaliação de um profissional de saúde qualificado (médico radiologista ou especialista).**

### Pontos Críticos:
- ❌ **Não** deve ser usado para diagnóstico clínico final
- 📊 Os modelos são **simulados e treinados em dados sintéticos** para demonstração da arquitetura
- 🔒 Nenhum dado do paciente é armazenado de forma persistente
- 🕵️ Metadados do arquivo (nome original, etc.) são removidos em tempo de execução para **anonimização completa**
- 🎓 Projeto com **fins estritamente educacionais e de pesquisa**

### Conformidade com LGPD (Lei Geral de Proteção de Dados):
- ✅ **Finalidade:** Uso exclusivo para demonstração técnica
- ✅ **Minimização:** Coleta apenas de pixels da imagem
- ✅ **Anonimização:** Remoção de metadados antes do processamento
- ✅ **Segurança:** Dados não persistem após o fim da sessão
- ✅ **Transparência:** Documentação completa de limitações

---

## 🎯 Funcionalidades

### Core Features
| Funcionalidade | Status | Descrição |
|----------------|--------|------------|
| Upload e Validação | ✅ | Suporta JPG, PNG e DCM (DICOM) |
| Upload em Lotes | ✅ | Processamento de múltiplas imagens |
| Pré-processamento | ✅ | Conversão cinza, normalização, redimensionamento |
| Remoção de Ruído | ✅ | Filtro Gaussiano (kernel 5x5) |
| Segmentação | ✅ | Limiarização de Otsu |
| Detecção de Bordas | ✅ | Algoritmo de Canny |
| Extração de Features | ✅ | Área, perímetro, número de regiões |
| Classificação | ✅ | Random Forest (dados sintéticos) |
| Interface Lado-a-Lado | ✅ | Original vs Segmentado |
| Exportação PDF | ✅ | Laudo completo com ReportLab |
| Métricas de Validação | ✅ | Matriz de Confusão, ROC, F1-Score |

### Funcionalidades Técnicas

- 🔄 Processamento em lote com barra de progresso
- 📊 Visualização interativa de métricas com Plotly
- 💾 Cache de modelos para performance otimizada
- 📱 Design responsivo (Streamlit)

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Finalidade | Por que foi escolhida? |
|------------|--------|------------|------------------------|
| 🐍 Python | 3.10+ | Linguagem principal | Ecossistema rico para IA/visão computacional |
| 🎨 Streamlit | 1.20+ | Interface web | Prototipagem rápida, integração nativa com ML |
| 🧠 TensorFlow | 2.10+ | Deep learning | (Opcional) Para futuras implementações de CNN |
| 🖼️ OpenCV | 4.7+ | Processamento de imagem | Biblioteca padrão da indústria para visão computacional |
| 📊 Scikit-learn | 1.2+ | Machine Learning | Algoritmos clássicos e métricas de validação |
| 🔢 NumPy | 1.23+ | Computação numérica | Manipulação eficiente de arrays de pixels |
| 📈 SciPy | 1.10+ | Processamento científico | Filtros morfológicos e análise de sinais |
| 🖌️ Pillow | 9.0+ | Manipulação básica | Fallback e operações simples de imagem |
| 🔬 Scikit-image | 0.20+ | Algoritmos avançados | Complemento ao OpenCV para segmentação |
| 🏥 PyDICOM | 2.3+ | Padrão médico | Leitura de arquivos DICOM com metadados |
| 📄 ReportLab | 3.6+ | Geração PDF | Exportação profissional de laudos |
| 📊 Matplotlib | 3.7+ | Visualização | Gráficos estáticos e matriz de confusão |
| 🎨 Plotly | 5.17+ | Visualização interativa | Gráficos dinâmicos e interativos |

---

## 🏗️ Arquitetura do Sistema

## 🏗️ Arquitetura do Sistema

### Fluxo de Processamento

| Etapa | Componente | Tecnologia | Status |
|-------|------------|------------|--------|
| 1 | Upload de Imagem | Streamlit | ✅ |
| 2 | Validação | Python | ✅ |
| 3 | Pré-processamento | OpenCV | ✅ |
| 4 | Segmentação | Otsu + Canny | ✅ |
| 5 | Extração de Features | NumPy/SciPy | ✅ |
| 6 | Classificação | Random Forest | ✅ |
| 7 | Métricas | Scikit-learn | ✅ |
| 8 | Exportação PDF | ReportLab | ✅ |

---

## 🚀 Como Instalar e Executar

### Pré-requisitos

# Versões mínimas necessárias
Python >= 3.9
pip >= 21.0
Git (opcional)

Passo a Passo

1. Clone o repositório
```
git clone https://github.com/SergioMitsu/Sistema_Analise_Medico.git
cd Sistema_Analise_Medico
```

2. Crie um ambiente virtual (recomendado)
```
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências
```
pip install -r requirements.txt
```

4. Gere os dados de teste
```
python generate_samples.py
```

5. Execute a aplicação
```
streamlit run app.py
```

6. Acesse no navegador
```
http://localhost:8501
```
---

## 📊 Métricas de Validação
Métricas Utilizadas
Métrica	Fórmula	Interpretação Clínica
Recall (Sensibilidade)	TP / (TP + FN)	Capacidade de detectar anomalias reais
Precisão	TP / (TP + FP)	Proporção de alertas corretos
F1-Score	2 × (P × R) / (P + R)	Balanceamento entre Recall e Precisão
AUC-ROC	Área sob a curva	Capacidade de discriminação do modelo

---

## ⚠️ Limitações Conhecidas

Técnicas

    Dataset Sintético

        ❌ O Random Forest é treinado com amostras sintéticas (mock)

        🔄 Em cenário real, exigiria dataset extenso e curado por especialistas

        📊 Distribuição de classes artificialmente balanceada

    Segmentação

        ⚠️ O limiar de Otsu funciona bem com histogramas bimodais

        🔍 Tecidos complexos com variação extensa de contraste podem gerar falsos positivos

        🧠 Necessário ajuste fino para cada tipo de imagem médica

    Suporte DICOM

        📄 Suporte parcial (apenas extração básica de pixels)

        🏥 Imagens médicas reais demandam pipeline extenso de metadados

        🔐 Necessário sistema completo de gerenciamento de dados de pacientes

Operacionais

    Performance

        ⏱️ Processamento em CPU (sem aceleração GPU)

        💾 Imagens muito grandes (4096×4096+) podem exceder limites de memória

    Validação Clínica

        🚫 Nenhuma validação com dados reais de pacientes

        📋 Não aprovado por órgãos reguladores (ANVISA/FDA)

        👨‍⚕️ Não substitui avaliação de radiologista

---

## 📚 Referências

Bibliotecas Utilizadas

    Streamlit - Framework web
    OpenCV - Processamento de imagem
    Scikit-learn - Machine learning
    TensorFlow - Deep learning
    PyDICOM - Imagens médicas

Datasets Públicos (para futuro)

    NIH Chest X-Ray
    CheXpert
    MIMIC-CXR

Artigos Científicos

    Otsu, N. (1979). "A threshold selection method from gray-level histograms"
    Canny, J. (1986). "A computational approach to edge detection"
    Esteva, A. et al. (2017). "Dermatologist-level classification of skin cancer"

Regulamentação

    ANVISA - Software Médico
    FDA - Software as Medical Device (SaMD)
    LGPD - Lei Geral de Proteção de Dados

📧 Contato
	
Autor Sergio Mitsushigue
Curso	Tecnólogo em Inteligência Artificial
Disciplina	Processamento de Imagem e Sinais
Instituição	FMU

📄 Licença

Este projeto é open-source para fins educacionais e de pesquisa.
text

Copyright (c) 2026 SergioMitsu

Permissão é concedida gratuitamente a qualquer pessoa que obtenha uma cópia
deste software para fins educacionais e de pesquisa, sem restrições.

---
