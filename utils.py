import logging
import uuid
from typing import Tuple, Optional

# Configuração de Logging para debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("medical_analysis_system")

def validate_image_file(uploaded_file) -> Tuple[bool, str]:
    """
    Valida o arquivo enviado verificando se ele existe e tem uma extensão permitida.
    """
    if uploaded_file is None:
        return False, "Nenhum arquivo enviado."
    
    # Anonimização simples: ignorar o nome real em logs
    allowed_types = ['image/jpeg', 'image/png', 'application/dicom']
    if uploaded_file.type not in allowed_types:
        if not uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png', '.dcm')):
             return False, f"Formato inválido. Tipos permitidos: JPG, PNG, DCM. Recebido: {uploaded_file.type}"
             
    logger.info("Arquivo de imagem validado com sucesso.")
    return True, "Arquivo válido"

def anonymize_metadata(filename: str) -> str:
    """
    Simula anonimização gerando um ID único para processamento e logs em vez do nome do arquivo
    """
    anon_id = str(uuid.uuid4())[:8]
    logger.info(f"Arquivo anonimizado. ID de processamento interno: {anon_id}")
    return anon_id
