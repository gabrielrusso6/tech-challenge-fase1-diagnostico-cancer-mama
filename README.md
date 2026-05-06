# Tech Challenge - Fase 1: Diagnóstico de Câncer de Mama

## Resumo do Projeto

Desenvolvimento de um pipeline de Machine Learning para classificação de tumores de mama como benignos ou malignos, utilizando o dataset Wisconsin Breast Cancer com técnicas de aprendizado supervisionado.

## Objetivo

Implementar uma solução educacional de apoio à triagem diagnóstica, demonstrando aplicação de técnicas de Machine Learning em problemas médicos com rigor metodológico na validação e avaliação de modelos.

## Contexto do Problema

O câncer de mama é uma das principais preocupações de saúde pública mundial. O diagnóstico precoce é fundamental para o sucesso do tratamento. Este projeto explora como algoritmos de Machine Learning podem auxiliar na análise de características de tumores, fornecendo uma ferramenta complementar de apoio à decisão clínica.

## Dataset Utilizado

**Wisconsin Breast Cancer Dataset**
- Fonte: UCI Machine Learning Repository
- Amostras: 569 pacientes
- Features: 30 características clínicas (raio, textura, perímetro, etc.)
- Target: Diagnóstico (Benigno/Maligno)
- Formato: Dados tabulares numéricos

## Estrutura do Projeto

```
tech-challenge-fase1-diagnostico-cancer-mama/
├── src/                    # Módulos principais
│   ├── preprocessing.py       # Pipeline de pré-processamento
│   ├── train.py              # Treinamento e otimização
│   └── evaluate.py           # Avaliação e métricas
├── data/                   # Dataset
│   └── breast-cancer-wisconsin.csv
├── notebooks/              # Análises exploratórias
│   └── analysis.ipynb
├── models/                 # Modelos treinados (auto)
├── run_full_pipeline.py    # Pipeline principal
├── main.py                 # Pipeline original
├── test_pipeline.py        # Testes funcionais
├── requirements.txt         # Dependências
├── Dockerfile              # Containerização
└── README.md               # Documentação
```

## Tecnologias Utilizadas

- Python 3.9+ - Linguagem principal
- Scikit-Learn - Biblioteca de Machine Learning
- Pandas & NumPy - Manipulação de dados
- Matplotlib & Seaborn - Visualizações
- Docker - Containerização
- Jupyter - Análises exploratórias

## Como Executar

### Pré-requisitos
```bash
# Python 3.9+ e pip
python --version
pip --version

# Instalar dependências
pip install -r requirements.txt
```

### Execução Local
```bash
# Pipeline completo (recomendado)
python3 run_full_pipeline.py

# Pipeline com visualizações
python3 main.py

# Teste rápido de pré-processamento
python3 test_pipeline.py
```

### Execução com Docker
```bash
# Construir imagem
docker build -t tech-challenge-cancer-mama .

# Executar pipeline
docker run --rm tech-challenge-cancer-mama
```

## Fluxo do Pipeline

1. Carregamento e Exploração: Análise inicial do dataset
2. Pré-processamento: Limpeza e transformação dos dados
3. Divisão Estratificada: Separação treino/teste (80/20)
4. Treinamento: Múltiplos algoritmos com validação cruzada
5. Otimização: Grid Search nos melhores modelos
6. Avaliação Final: Teste em conjunto separado
7. Relatório: Métricas e visualizações

## Pré-processamento Aplicado

- Remoção de colunas: ID e colunas sem informação relevante, como Unnamed
- Tratamento de valores ausentes: Imputação pela média
- Encoding target: M=1 (Maligno), B=0 (Benigno)
- Normalização: StandardScaler (média=0, desvio=1)
- Split estratificado: Mantém proporção das classes

## Modelos Utilizados

1. Random Forest - Ensemble de árvores de decisão
2. SVM (Support Vector Machine) - Máquina de vetores de suporte
3. Logistic Regression - Regressão logística
4. KNN (K-Nearest Neighbors) - K vizinhos mais próximos
5. Naive Bayes - Classificador bayesiano

## Estratégia de Validação

O dataset foi dividido de forma estratificada em treino e teste. O conjunto de teste foi reservado para avaliação final. A validação dos modelos foi realizada por validação cruzada com 5 folds dentro do conjunto de treino, permitindo comparar algoritmos e ajustar hiperparâmetros sem utilizar os dados de teste.

Essa abordagem metodológica garante que:
- O conjunto de teste representa dados nunca vistos
- A seleção do modelo final é baseada apenas em validação cruzada
- A avaliação final é imparcial e confiável
- Não há contaminação entre treinamento e teste

## Métricas de Avaliação

- Accuracy - Taxa de acerto geral
- Precision - Proporção de positivos corretos
- Recall (Sensitivity) - Capacidade de detectar malignos
- F1-Score - Balanceamento precision/recall
- AUC-ROC - Discriminação entre classes
- Matriz de Confusão - Análise detalhada de erros

## Resultados Obtidos

### Performance dos Modelos

**LogisticRegression:**
- Accuracy: 0.9737
- Precision: 0.9756
- Recall: 0.9524
- F1-Score: 0.9639
- AUC-ROC: 0.9964

**RandomForest:**
- Accuracy: 0.9737
- Precision: 1.0000
- Recall: 0.9286
- F1-Score: 0.9630
- AUC-ROC: 0.9929

**SVM:**
- Accuracy: 0.9737
- Precision: 1.0000
- Recall: 0.9286
- F1-Score: 0.9630
- AUC-ROC: 0.9954

**KNN:**
- Accuracy: 0.9561
- Precision: 0.9744
- Recall: 0.9048
- F1-Score: 0.9383
- AUC-ROC: 0.9816

**NaiveBayes:**
- Accuracy: 0.9211
- Precision: 0.9231
- Recall: 0.8571
- F1-Score: 0.8889
- AUC-ROC: 0.9891

### Features Mais Relevantes
1. concave points_worst
2. perimeter_worst
3. radius_worst
4. area_worst
5. concavity_worst

## Interpretação dos Resultados

A Logistic Regression apresentou o melhor equilíbrio geral entre as métricas. O recall de 0.9524 para a classe maligna é relevante em contexto médico, pois reduz o risco de falsos negativos. A AUC-ROC de 0.9964 indica alta capacidade de separação entre casos benignos e malignos. A importância das features indica que características relacionadas à concavidade e perímetro são mais preditivas, A importância das features indica que características relacionadas à concavidade, perímetro, raio e área tiveram maior influência nas previsões do modelo.

## Limitações e Observação Médica/Ética

Este projeto não substitui diagnóstico médico. O modelo foi desenvolvido para fins educacionais e pode ser interpretado como uma ferramenta experimental de apoio à triagem. A decisão clínica final deve sempre ser realizada por profissionais de saúde qualificados.

As decisões médicas devem ser sempre baseadas em:
- Avaliação completa por profissionais qualificados
- Exames clínicos e laboratoriais apropriados
- Histórico completo do paciente
- Contexto clínico individual

## Próximos Passos (Opcional)

- [ ] Validação com datasets externos
- [ ] Análise de interpretabilidade (SHAP, LIME)
- [ ] Experimentos com ensembles avançados
- [ ] Validação prospectiva em contexto clínico
- [ ] Interface para demonstração educacional

## Status do Projeto

Este pipeline de Machine Learning está funcional para fins educacionais e acadêmicos, pronto para documentação final e demonstração das técnicas implementadas.

---

Desenvolvido como parte do Tech Challenge - Fase 1  
Propósito educacional e demonstrativo  
Não substitui avaliação médica profissional