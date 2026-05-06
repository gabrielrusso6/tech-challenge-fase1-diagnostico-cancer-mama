import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.model_results = {}
        
    def initialize_models(self):
        """Inicializa os modelos de Machine Learning."""
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'KNN': KNeighborsClassifier(),
            'NaiveBayes': GaussianNB()
        }
        
    def train_baseline_models(self, X_train, y_train):
        """Treina modelos baseline com validação cruzada 5-fold.
        
        A validação cruzada é utilizada para comparar modelos de forma robusta,
        sem utilizar o conjunto de teste que fica reservado para avaliação final.
        """
        print("Treinando modelos baseline com validação cruzada...")
        print("-" * 50)
        
        for name, model in self.models.items():
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Validação cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Armazenar resultados
            self.model_results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Atualizar melhor modelo
            if cv_scores.mean() > self.best_score:
                self.best_score = cv_scores.mean()
                self.best_model = model
                
        print(f"\nMelhor modelo baseline: {type(self.best_model).__name__} "
              f"com score: {self.best_score:.4f}")
        
    def hyperparameter_tuning(self, X_train, y_train):
        """Otimização de hiperparâmetros usando Grid Search com validação cruzada.
        
        A otimização utiliza apenas o conjunto de treino com validação cruzada,
        mantendo o conjunto de teste isolado para avaliação final imparcial.
        """
        print("\nOtimizando hiperparâmetros com Grid Search...")
        print("-" * 50)
        
        # Definir grids de hiperparâmetros
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        # Otimizar apenas os 3 melhores modelos baseline
        top_models = sorted(self.model_results.items(), 
                          key=lambda x: x[1]['cv_mean'], reverse=True)[:3]
        
        for name, results in top_models:
            if name in param_grids:
                print(f"\nOtimizando {name}...")
                
                model = results['model']
                param_grid = param_grids[name]
                
                # Grid Search
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, 
                    scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                # Atualizar resultados
                self.model_results[f"{name}_tuned"] = {
                    'model': grid_search.best_estimator_,
                    'cv_mean': grid_search.best_score_,
                    'best_params': grid_search.best_params_,
                    'cv_scores': None
                }
                
                print(f"Melhor score: {grid_search.best_score_:.4f}")
                print(f"Melhores parâmetros: {grid_search.best_params_}")
                
                # Atualizar melhor modelo se necessário
                if grid_search.best_score_ > self.best_score:
                    self.best_score = grid_search.best_score_
                    self.best_model = grid_search.best_estimator_
                    print(f"Novo melhor modelo: {name} (tuned)")
    
    def feature_importance_analysis(self, X_train, feature_names):
        """Analisa importância das features para modelos que suportam."""
        print("\nAnálise de Importância de Features")
        print("-" * 50)
        
        # Random Forest feature importance
        if 'RandomForest' in self.model_results:
            rf_model = self.model_results['RandomForest']['model']
            importances = rf_model.feature_importances_
            
            # Criar DataFrame com importâncias
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Features mais importantes (Random Forest):")
            print(feature_importance.head(10).to_string(index=False))
            
            return feature_importance
        
        return None
    
    def get_all_results(self):
        """Retorna todos os resultados dos modelos."""
        return self.model_results
    
    def get_best_model(self):
        """Retorna o melhor modelo treinado."""
        return self.best_model, self.best_score
    
    def save_model(self, model, filepath):
        """Salva o modelo treinado."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modelo salvo em: {filepath}")
    
    def train_complete_pipeline(self, X_train, y_train, feature_names):
        """Pipeline completo de treinamento."""
        # Inicializar modelos
        self.initialize_models()
        
        # Treinar modelos baseline
        self.train_baseline_models(X_train, y_train)
        
        # Otimizar hiperparâmetros
        self.hyperparameter_tuning(X_train, y_train)
        
        # Análise de feature importance
        feature_importance = self.feature_importance_analysis(X_train, feature_names)
        
        # Salvar melhor modelo
        self.save_model(self.best_model, 'models/best_model.pkl')
        
        return self.best_model, self.model_results, feature_importance