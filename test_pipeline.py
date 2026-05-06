#!/usr/bin/env python3
"""
Test script para verificar o pipeline do projeto
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from src.preprocessing import BreastCancerPreprocessor

def test_preprocessing():
    print("Testando pré-processamento...")
    
    # Carregar dados diretamente para verificar estrutura
    df = pd.read_csv('data/breast-cancer-wisconsin.csv')
    print(f"Shape original: {df.shape}")
    print(f"Colunas: {list(df.columns)}")
    print(f"Valores nulos: {df.isnull().sum().sum()}")
    
    # Verificar colunas Unnamed
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    print(f"Colunas Unnamed: {unnamed_cols}")
    
    # Testar pré-processamento
    preprocessor = BreastCancerPreprocessor()
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            'data/breast-cancer-wisconsin.csv',
            test_size=0.2
        )
        print("Pré-processamento concluído com sucesso!")
        return True
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return False

if __name__ == "__main__":
    success = test_preprocessing()
    if success:
        print("Teste passou!")
    else:
        print("Teste falhou!")
