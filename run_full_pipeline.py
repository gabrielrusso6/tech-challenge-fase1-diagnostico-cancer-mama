#!/usr/bin/env python3
"""
Versão corrigida e otimizada do pipeline completo
"""

import sys
import os
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import BreastCancerPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator

def main():
    print("=" * 60)
    print("Tech Challenge - Fase 1")
    print("Diagnóstico de Câncer de Mama com Machine Learning")
    print("=" * 60)
    
    try:
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
        
        # 4. Análise do melhor modelo
        print("\n4. ANÁLISE DETALHADA DO MELHOR MODELO")
        print("-" * 40)
        
        best_model_name = comparison_df.iloc[0]['Model']
        best_results = evaluator.evaluation_results[best_model_name]
        
        print(f"\nMelhor modelo: {best_model_name}")
        print(f"F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
        print(f"Acurácia: {comparison_df.iloc[0]['Accuracy']:.4f}")
        print(f"Precisão: {comparison_df.iloc[0]['Precision']:.4f}")
        print(f"Recall: {comparison_df.iloc[0]['Recall']:.4f}")
        
        # 5. Importância das features (se disponível)
        if feature_importance is not None:
            print("\n5. TOP 10 FEATURES MAIS IMPORTANTES")
            print("-" * 40)
            print(feature_importance.head(10).to_string(index=False))
        
        # 6. Gerar relatório final
        print("\n6. RELATÓRIO FINAL")
        print("-" * 40)
        
        evaluator.generate_report('evaluation_report.txt')
        
        print("\nPipeline completo executado com sucesso!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nERRO durante a execução: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Projeto concluído com sucesso!")
    else:
        print("\n❌ Ocorreram erros durante a execução.")
