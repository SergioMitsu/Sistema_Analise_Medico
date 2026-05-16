import numpy as np
import cv2
import os

# Suprime logs de warnings do tensorflow para o output não ficar poluído
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from utils import logger

class CNNClassifier:
    def __init__(self):
        """
        Inicializa o classificador baseado na ResNet50.
        """
        logger.info("Inicializando ResNet50 (Keras)... Pode levar alguns segundos.")
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Adiciona camada de classificação
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Última camada convolucional para o Grad-CAM
        self.last_conv_layer_name = "conv5_block3_out" 
        
        # A rede já terá pesos na base e pesos aleatórios no topo.
        # Seria treinado aqui, mas usaremos a rede mockada para mostrar pipeline
        self.is_trained = True

    def predict(self, image: np.ndarray) -> tuple[bool, float]:
        """
        Realiza a predição da imagem (deve ser RGB).
        Retorna (is_anomaly, confidence).
        """
        # Redimensiona para o input esperado da ResNet50 (224, 224)
        img_resized = cv2.resize(image, (224, 224))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Pré-processamento padrão da ResNet50
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        is_anomaly = bool(prediction > 0.5)
        
        confidence = float(prediction if is_anomaly else 1 - prediction) * 100
        
        logger.info(f"Classificação CNN: {'ANOMALIA' if is_anomaly else 'NORMAL'} - Confiança: {confidence:.2f}%")
        return is_anomaly, confidence

    def get_gradcam(self, image: np.ndarray) -> np.ndarray:
        """
        Computa o Heatmap do Grad-CAM para a imagem de entrada.
        Retorna um heatmap colorido redimensionado para o tamanho original da imagem,
        sobreposto com a imagem original.
        """
        logger.info("Computando mapa de calor Grad-CAM...")
        orig_shape = image.shape[:2]
        
        img_resized = cv2.resize(image, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            # Para binário com sigmoid, a predição é um único valor
            class_channel = preds[:, 0]
            
        # Gradientes da camada em relação à classe
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Global Average Pooling dos gradientes
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Ponderação dos canais
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU para manter apenas as features positivas
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Preenche zeros e evita NaNs
        heatmap = np.nan_to_num(heatmap)
        
        # Redimensiona para o tamanho original
        heatmap = cv2.resize(heatmap, (orig_shape[1], orig_shape[0]))
        
        # Converte para heatmap colorido usando OpenCV JET colormap
        heatmap_uint8 = np.uint8(255 * heatmap)
        jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Sobrepõe com a imagem original
        superimposed_img = cv2.addWeighted(image, 0.6, jet_heatmap, 0.4, 0)
        
        return superimposed_img
