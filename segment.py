import cv2
import numpy as np
from typing import Tuple, List, Dict
from utils import logger

def segment_image(gray_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Executa a segmentação utilizando Limiarização de Otsu e Canny Edge Detection.
    Retorna a máscara binária, as bordas (Canny) e os contornos detectados.
    """
    logger.info("Iniciando segmentação...")
    
    # 1. Limiarização de Otsu
    # Aplicamos Otsu; dependendo da imagem médica, pode ser necessário inverter
    # assumiremos que queremos as áreas mais "brilhantes" ou "escuras" anormais.
    # Usaremos uma limiarização binária comum para o foco de Otsu.
    ret, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Canny Edge Detection
    edges = cv2.Canny(gray_img, threshold1=ret*0.5, threshold2=ret)
    
    # 3. Extrair Contornos baseados na máscara de Otsu (podemos usar as bordas também)
    # Aqui usaremos a máscara de Otsu para encontrar áreas sólidas suspeitas
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    logger.info(f"Segmentação concluída. Contornos encontrados: {len(contours)}")
    return mask, edges, list(contours)

def extract_features(contours: List[np.ndarray], min_area: float = 50.0) -> Dict[str, float]:
    """
    Extrai atributos (features) dos contornos detectados.
    Filtra contornos muito pequenos (ruído residual).
    """
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    num_regions = len(valid_contours)
    
    total_area = sum(cv2.contourArea(c) for c in valid_contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in valid_contours)
    
    # Eixo Maior e Menor médio (aproximação usando elipse se o contorno tiver pelo menos 5 pontos)
    major_axes = []
    minor_axes = []
    
    for c in valid_contours:
        if len(c) >= 5:
            try:
                (x, y), (MA, ma), angle = cv2.fitEllipse(c)
                major_axes.append(max(MA, ma))
                minor_axes.append(min(MA, ma))
            except Exception:
                pass
                
    avg_major_axis = np.mean(major_axes) if major_axes else 0.0
    avg_minor_axis = np.mean(minor_axes) if minor_axes else 0.0
    
    features = {
        "num_regions": float(num_regions),
        "total_area": float(total_area),
        "total_perimeter": float(total_perimeter),
        "avg_major_axis": float(avg_major_axis),
        "avg_minor_axis": float(avg_minor_axis)
    }
    
    logger.info(f"Features extraídas: {features}")
    return features, valid_contours

def draw_contours_on_image(image: np.ndarray, contours: List[np.ndarray], color=(255, 0, 0)) -> np.ndarray:
    """
    Desenha os contornos em cima de uma cópia da imagem original.
    Cor padrão: Vermelho para anomalias.
    """
    display_img = image.copy()
    cv2.drawContours(display_img, contours, -1, color, 2)
    return display_img
