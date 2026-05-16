import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, List
from utils import logger
import pickle

class AnomalyClassifier:
    def __init__(self):
        """
        Inicializa o classificador Random Forest.
        No mundo real, carregaríamos um modelo pré-treinado aqui (.pkl, .h5, etc).
        Para o nosso MVP, iremos instanciar e treiná-lo com dados sintéticos.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def _dict_to_array(self, features_dict: Dict[str, float]) -> np.ndarray:
        """Converte o dicionário de features em array numpy para o sklearn"""
        return np.array([[
            features_dict.get('num_regions', 0),
            features_dict.get('total_area', 0),
            features_dict.get('total_perimeter', 0),
            features_dict.get('avg_major_axis', 0),
            features_dict.get('avg_minor_axis', 0)
        ]])

    def train_synthetic(self, num_samples: int = 500):
        """
        Treina o modelo em dados sintéticos (Mock) baseado na lógica:
        Muitas regiões E área alta => Anomalia (Classe 1)
        Poucas regiões E área baixa => Normal (Classe 0)
        """
        logger.info("Iniciando treinamento sintético do Random Forest...")
        X_train = []
        y_train = []
        
        for _ in range(num_samples):
            # Normal class (0): Menos contornos, áreas menores
            if np.random.rand() > 0.5:
                num_regions = np.random.randint(0, 5)
                area = np.random.uniform(0, 5000)
                perim = np.random.uniform(0, 1000)
                maj_ax = np.random.uniform(0, 50)
                min_ax = np.random.uniform(0, 30)
                label = 0
            # Anomaly class (1): Mais contornos, áreas grandes (ex: tumores/lesões opacas)
            else:
                num_regions = np.random.randint(3, 20)
                area = np.random.uniform(3000, 50000)
                perim = np.random.uniform(500, 5000)
                maj_ax = np.random.uniform(40, 200)
                min_ax = np.random.uniform(20, 100)
                label = 1
                
            X_train.append([num_regions, area, perim, maj_ax, min_ax])
            y_train.append(label)
            
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Treinamento sintético concluído.")

    def predict(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """
        Recebe o dicionário de features, retorna se é Anomalia (True) ou Normal (False)
        e a probabilidade de confiança na classe predita.
        """
        if not self.is_trained:
            logger.warning("Modelo não foi treinado. Chamando train_synthetic() automaticamente.")
            self.train_synthetic()
            
        x_input = self._dict_to_array(features)
        
        prediction = self.model.predict(x_input)[0]
        probabilities = self.model.predict_proba(x_input)[0]
        
        # prediction = 1 (Anomalia), prediction = 0 (Normal)
        is_anomaly = bool(prediction == 1)
        confidence = float(probabilities[prediction]) * 100
        
        logger.info(f"Classificação: {'ANOMALIA' if is_anomaly else 'NORMAL'} - Confiança: {confidence:.2f}%")
        return is_anomaly, confidence
