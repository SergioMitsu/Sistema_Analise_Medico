import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from typing import Dict, Any

def generate_evaluation_metrics(model, num_test_samples: int = 100) -> Dict[str, Any]:
    """
    Gera métricas de avaliação usando o classificador em dados sintéticos de teste.
    Isso simula o processo de validação requerido.
    """
    X_test = []
    y_test = []
    
    # Gera um batch de teste análogo ao treinamento
    for _ in range(num_test_samples):
        if np.random.rand() > 0.5:
            num_regions = np.random.randint(0, 5)
            area = np.random.uniform(0, 5000)
            perim = np.random.uniform(0, 1000)
            maj_ax = np.random.uniform(0, 50)
            min_ax = np.random.uniform(0, 30)
            label = 0
        else:
            num_regions = np.random.randint(3, 20)
            area = np.random.uniform(3000, 50000)
            perim = np.random.uniform(500, 5000)
            maj_ax = np.random.uniform(40, 200)
            min_ax = np.random.uniform(20, 100)
            label = 1
            
        X_test.append([num_regions, area, perim, maj_ax, min_ax])
        y_test.append(label)
        
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] # Probabilidades da classe positiva
    
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_curve": (fpr, tpr, roc_auc)
    }

def export_to_pdf(image_original, image_processed, features, classification, confidence) -> bytes:
    """
    Gera um relatório PDF simples em memória contendo as imagens e o diagnóstico e retorna em bytes.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Laudo Analítico (Suporte Diagnóstico via IA)")
    
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(1, 0, 0)
    c.drawString(50, height - 70, "AVISO: Este sistema NÃO substitui avaliação de um médico qualificado.")
    c.setFillColorRGB(0, 0, 0)
    
    # Textos do Resultado
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 100, f"Classificação: {'ANOMALIA DETECTADA' if classification else 'NORMAL'}")
    c.drawString(50, height - 120, f"Confiança: {confidence:.2f}%")
    
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 150, "Atributos Extraídos das Regiões Suspeitas:")
    c.drawString(60, height - 170, f"Total de Regiões: {int(features.get('num_regions', 0))}")
    c.drawString(60, height - 190, f"Área Total: {features.get('total_area', 0):.2f} px")
    c.drawString(60, height - 210, f"Perímetro Total: {features.get('total_perimeter', 0):.2f} px")
    
    # Processando as imagens para colocar no PDF (ReportLab requer objeto stream ou path)
    # Como as imagens podem ser arrays numpy de tamanhos diferentes, é mais seguro converter via PIL.
    from PIL import Image
    
    img_orig_pil = Image.fromarray(image_original)
    buffer_orig = io.BytesIO()
    img_orig_pil.save(buffer_orig, format="PNG")
    buffer_orig.seek(0)
    img_reader_orig = ImageReader(buffer_orig)
    
    img_proc_pil = Image.fromarray(image_processed)
    buffer_proc = io.BytesIO()
    img_proc_pil.save(buffer_proc, format="PNG")
    buffer_proc.seek(0)
    img_reader_proc = ImageReader(buffer_proc)
    
    # Renderiza as imagens (lado a lado se possível, ou uma embaixo da outra)
    c.drawString(50, height - 250, "Imagem Original:")
    c.drawImage(img_reader_orig, 50, height - 460, width=200, height=200, preserveAspectRatio=True)
    
    c.drawString(300, height - 250, "Regiões Destacadas:")
    c.drawImage(img_reader_proc, 300, height - 460, width=200, height=200, preserveAspectRatio=True)
    
    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer.getvalue()
