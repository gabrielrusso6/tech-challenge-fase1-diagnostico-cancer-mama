import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class BreastCancerPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = None
        
    def load_data(self, file_path):
        """Carrega o dataset do arquivo CSV."""
        df = pd.read_csv(file_path)
        return df
    
    def clean_data(self, df):
        """Limpeza básica dos dados."""
        # Remover coluna ID se existir
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Remover colunasUnnamed se existir
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(unnamed_cols, axis=1)
            print(f"Removidas colunas: {unnamed_cols}")
        
        # Verificar e remover duplicatas
        df = df.drop_duplicates()
        
        # Verificar valores ausentes
        print(f"Valores ausentes antes do tratamento: {df.isnull().sum().sum()}")
        
        return df
    
    def encode_target(self, df):
        """Codifica a variável alvo (diagnosis)."""
        # M = 1 (Maligno), B = 0 (Benigno)
        df['diagnosis'] = self.label_encoder.fit_transform(df['diagnosis'])
        return df
    
    def split_features_target(self, df):
        """Separa features e target."""
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        self.feature_columns = X.columns.tolist()
        return X, y
    
    def handle_missing_values(self, X):
        """Trata valores ausentes nas features."""
        X_imputed = self.imputer.fit_transform(X)
        return pd.DataFrame(X_imputed, columns=self.feature_columns)
    
    def scale_features(self, X):
        """Normaliza as features."""
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=self.feature_columns)
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Divide dados em treino e teste de forma estratificada.
        
        O conjunto de teste é mantido separado e utilizado apenas para avaliação final,
        garantindo que não haja vazamento de dados durante o treinamento e otimização.
        """
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state, stratify=y)
    
    def preprocess_pipeline(self, file_path, test_size=0.2):
        """Pipeline completo de pré-processamento."""
        # Carregar dados
        df = self.load_data(file_path)
        print(f"Dataset carregado: {df.shape}")
        
        # Limpar dados
        df = self.clean_data(df)
        print(f"Dataset após limpeza: {df.shape}")
        
        # Codificar target
        df = self.encode_target(df)
        
        # Separar features e target
        X, y = self.split_features_target(df)
        
        # Tratar valores ausentes
        X = self.handle_missing_values(X)
        
        # Normalizar features
        X = self.scale_features(X)
        
        # Dividir em treino/teste
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        
        print(f"Dados de treino: {X_train.shape}")
        print(f"Dados de teste: {X_test.shape}")
        print(f"Distribuição treino - Benigno(0): {sum(y_train==0)}, Maligno(1): {sum(y_train==1)}")
        print(f"Distribuição teste - Benigno(0): {sum(y_test==0)}, Maligno(1): {sum(y_test==1)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_data(self, X, y):
        """Retorna dados para análise de importância de features."""
        return X, y, self.feature_columns