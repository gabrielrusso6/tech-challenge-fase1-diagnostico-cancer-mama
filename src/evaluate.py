import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
import pickle

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Avaliação completa de um modelo."""
        print(f"\nAvaliando modelo: {model_name}")
        print("-" * 50)
        
        # Predições
        y_pred = model.predict(X_test)
        y_proba = None
        
        # Probabilidades (se o modelo suportar)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # AUC-ROC (se tiver probabilidades)
        auc_roc = None
        if y_proba is not None:
            auc_roc = roc_auc_score(y_test, y_proba)
        
        # Armazenar resultados
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.evaluation_results[model_name] = results
        
        # Imprimir métricas
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if auc_roc is not None:
            print(f"AUC-ROC: {auc_roc:.4f}")
        
        # Relatório completo
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Benigno(0)', 'Maligno(1)']))
        
        return results
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name="Model"):
        """Plota matriz de confusão."""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benigno', 'Maligno'],
                   yticklabels=['Benigno', 'Maligno'])
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.show()
        
    def plot_roc_curve(self, y_test, y_proba, model_name="Model"):
        """Plota curva ROC."""
        if y_proba is None:
            print("Modelo não suporta previsão de probabilidades")
            return
            
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title(f'Curva ROC - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_precision_recall_curve(self, y_test, y_proba, model_name="Model"):
        """Plota curva Precision-Recall."""
        if y_proba is None:
            print("Modelo não suporta previsão de probabilidades")
            return
            
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Curva Precision-Recall - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def compare_models(self):
        """Compara todos os modelos avaliados."""
        if not self.evaluation_results:
            print("Nenhum modelo avaliado ainda")
            return
            
        # Criar DataFrame comparativo
        comparison_data = []
        for name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'AUC-ROC': results['auc_roc'] if results['auc_roc'] else 'N/A'
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        print("\nComparação de Modelos:")
        print("=" * 80)
        print(df_comparison.to_string(index=False))
        
        # Gráfico comparativo
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(df_comparison))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = df_comparison[metric]
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel('Modelos')
        plt.ylabel('Score')
        plt.title('Comparação de Modelos - Métricas de Desempenho')
        plt.xticks(x + width*1.5, df_comparison['Model'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return df_comparison
        
    def evaluate_multiple_models(self, models_dict, X_test, y_test):
        """Avalia múltiplos modelos."""
        print("Avaliando múltiplos modelos...")
        print("=" * 60)
        
        for name, model in models_dict.items():
            self.evaluate_model(model, X_test, y_test, name)
            
        # Comparação final
        return self.compare_models()
        
    def generate_report(self, save_path='evaluation_report.txt'):
        """Gera relatório detalhado da avaliação."""
        if not self.evaluation_results:
            print("Nenhum modelo avaliado ainda")
            return
            
        report = []
        report.append("RELATÓRIO DE AVALIAÇÃO DE MODELOS")
        report.append("=" * 50)
        report.append(f"Total de modelos avaliados: {len(self.evaluation_results)}")
        report.append("")
        
        for name, results in self.evaluation_results.items():
            report.append(f"Modelo: {name}")
            report.append("-" * 30)
            report.append(f"Acurácia: {results['accuracy']:.4f}")
            report.append(f"Precisão: {results['precision']:.4f}")
            report.append(f"Recall: {results['recall']:.4f}")
            report.append(f"F1-Score: {results['f1_score']:.4f}")
            if results['auc_roc']:
                report.append(f"AUC-ROC: {results['auc_roc']:.4f}")
            report.append("")
        
        # Salvar relatório
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Relatório salvo em: {save_path}")
        
    def load_and_evaluate_model(self, model_path, X_test, y_test, model_name="Loaded Model"):
        """Carrega modelo salvo e avalia."""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            print(f"Modelo carregado de: {model_path}")
            return self.evaluate_model(model, X_test, y_test, model_name)
            
        except FileNotFoundError:
            print(f"Arquivo do modelo não encontrado: {model_path}")
            return None
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return None