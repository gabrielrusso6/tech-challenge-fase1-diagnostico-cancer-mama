#!/usr/bin/env python3
"""
Tech Challenge - Fase 1
Diagnóstico de Câncer de Mama com Machine Learning

Script principal para executar o pipeline completo de ML.
"""

import sys
import os
sys.path.append('src')

from src.preprocessing import BreastCancerPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator

def main():
    print("=" * 60)
    print("Tech Challenge - Fase 1")
    print("Diagnóstico de Câncer de Mama com Machine Learning")
    print("=" * 60)
    
    # 1. Pré-processamento dos dados
    print("\n1. PRÉ-PROCESSAMENTO DOS DADOS")
    print("-" * 40)
    
    preprocessor = BreastCancerPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        'data/breast-cancer-wisconsin.csv',
        test_size=0.2
    )
    
    # 2. Treinamento dos modelos
    print("\n2. TREINAMENTO DOS MODELOS")
    print("-" * 40)
    
    trainer = ModelTrainer()
    best_model, model_results, feature_importance = trainer.train_complete_pipeline(
        X_train, y_train, preprocessor.feature_columns
    )
    
    # 3. Avaliação dos modelos
    print("\n3. AVALIAÇÃO DOS MODELOS")
    print("-" * 40)
    
    evaluator = ModelEvaluator()
    
    # Avaliar todos os modelos treinados
    models_to_evaluate = {name: results['model'] for name, results in model_results.items()}
    comparison_df = evaluator.evaluate_multiple_models(models_to_evaluate, X_test, y_test)
    
    # 4. Visualizações do melhor modelo
    print("\n4. ANÁLISE DETALHADA DO MELHOR MODELO")
    print("-" * 40)
    
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_eval = model_results[best_model_name]['model']
    
    # Obter resultados do melhor modelo
    best_results = evaluator.evaluation_results[best_model_name]
    
    # Plotar visualizações
    evaluator.plot_confusion_matrix(y_test, best_results['y_pred'], best_model_name)
    evaluator.plot_roc_curve(y_test, best_results['y_proba'], best_model_name)
    evaluator.plot_precision_recall_curve(y_test, best_results['y_proba'], best_model_name)
    
    # 5. Importância das features
    if feature_importance is not None:
        print("\n5. IMPORTÂNCIA DAS FEATURES")
        print("-" * 40)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importância')
        plt.title('Top 15 Features Mais Importantes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    # 6. Gerar relatório final
    print("\n6. RELATÓRIO FINAL")
    print("-" * 40)
    
    evaluator.generate_report('evaluation_report.txt')
    
    print(f"\nMelhor modelo: {best_model_name}")
    print(f"F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
    print(f"Acurácia: {comparison_df.iloc[0]['Accuracy']:.4f}")
    
    print("\nPipeline completo executado com sucesso!")
    print("=" * 60)

if __name__ == "__main__":
    main()
