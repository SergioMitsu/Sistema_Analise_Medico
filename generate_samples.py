import numpy as np
import cv2
import os

def create_synthetic_image(filename: str, image_type: str):
    """
    Cria uma imagem de 'raio-x' ruidosa sintética usando NumPy e a salva.
    Tipos: 'normal', 'leve', 'grave'.
    """
    # 1. Base (fundo escuro similar ao ar nos pulmões/fundo) com ruído sal e pimenta
    img = np.random.normal(50, 20, (512, 512)).astype(np.uint8)
    
    # 2. Desenhando estruturas base (ex: Pulmões aproximados / costelas)
    cv2.ellipse(img, (150, 250), (80, 150), 0, 0, 360, 120, -1) # Pulmão Esq
    cv2.ellipse(img, (360, 250), (80, 150), 0, 0, 360, 120, -1) # Pulmão Dir
    
    # Suaviza as "estruturas orgânicas"
    img = cv2.GaussianBlur(img, (25, 25), 0)
    
    if image_type == 'normal':
        # Não adiciona anomalias marcantes
        pass
        
    elif image_type == 'leve':
        # Anomalia leve: algumas pequenas manchas brancas (nódulos)
        for _ in range(5):
            x = np.random.randint(100, 400)
            y = np.random.randint(150, 350)
            radius = np.random.randint(5, 15)
            cv2.circle(img, (x, y), radius, 200, -1)
            
    elif image_type == 'grave':
        # Anomalia grave: grandes massas densas (brancas) / consolidação
        for _ in range(3):
            x = np.random.randint(150, 350)
            y = np.random.randint(200, 350)
            axes = (np.random.randint(30, 60), np.random.randint(40, 80))
            cv2.ellipse(img, (x, y), axes, np.random.randint(0, 180), 0, 360, 220, -1)
        
        # Adiciona várias manchas menores ao redor
        for _ in range(15):
            x = np.random.randint(100, 400)
            y = np.random.randint(150, 400)
            radius = np.random.randint(5, 20)
            cv2.circle(img, (x, y), radius, 180, -1)

    # Aplica mais um leve blur para parecer menos artificial
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Salva a imagem
    output_path = os.path.join("data", "sample_images", filename)
    cv2.imwrite(output_path, img)
    print(f"Imagem sintética criada: {output_path}")

if __name__ == "__main__":
    # Garante que as pastas existem
    os.makedirs(os.path.join("data", "sample_images"), exist_ok=True)
    
    print("Gerando amostras sintéticas de testes...")
    create_synthetic_image("paciente_01_normal.jpg", "normal")
    create_synthetic_image("paciente_02_anomalia_leve.jpg", "leve")
    create_synthetic_image("paciente_03_anomalia_grave.png", "grave")
    print("Concluído. Você pode usar estas imagens na aplicação.")
