import cv2
import numpy as np
from PIL import Image
from typing import Tuple
from utils import logger
import io
import pydicom

def load_image(uploaded_file) -> np.ndarray:
    """
    Carrega a imagem via PIL ou pydicom (se for DCM) e converte para array NumPy.
    """
    try:
        if uploaded_file.name.lower().endswith('.dcm') or uploaded_file.type == 'application/dicom':
            # DICOM loading
            dicom_data = pydicom.dcmread(uploaded_file)
            pixel_array = dicom_data.pixel_array
            
            # Normaliza para 0-255 uint8
            norm_img = cv2.normalize(pixel_array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            norm_img = np.uint8(norm_img)
            
            # Converte para RGB se for Grayscale
            if len(norm_img.shape) == 2:
                img_array = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
            else:
                img_array = norm_img
                
            logger.info(f"DICOM carregado com shape original: {img_array.shape}")
            return img_array
            
        else:
            image = Image.open(uploaded_file)
            # Converte para RGB se estiver em outro formato (ex: RGBA ou L)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            logger.info(f"Imagem carregada com shape original: {img_array.shape}")
            return img_array
    except Exception as e:
        logger.error(f"Erro ao carregar imagem: {str(e)}")
        raise RuntimeError("Falha ao ler o conteúdo da imagem.")
def preprocess_image(image: np.ndarray, max_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executa o pipeline de pré-processamento obrigatório.
    - Redimensionamento (máx max_size x max_size, mantendo proporção)
    - Conversão Grayscale
    - Normalização
    - Blur Gaussiano (Kernel 5x5)
    
    Retorna: a imagem rgb redimensionada e a imagem grayscale processada.
    """
    logger.info("Iniciando pré-processamento...")
    # 1. Redimensionamento mantendo proporção
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Imagem redimensionada para: {image.shape}")
    
    # 2. Escala de Cinza
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 3. Normalização (0-255). cv2.normalize garante que espalhe bem os valores
    gray_norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # 4. Filtro Gaussiano para remoção de ruído
    processed_gray = cv2.GaussianBlur(gray_norm, (5, 5), 0)
    
    logger.info("Pré-processamento concluído.")
    return image, processed_gray
