# Tech Challenge - Fase 1: Diagnóstico de Câncer de Mama com Machine Learning

## 📋 Resumo do Projeto

Desenvolvimento de um pipeline de Machine Learning para apoio ao diagnóstico de câncer de mama, utilizando técnicas de classificação para distinguir entre tumores benignos e malignos a partir de dados clínicos tabulares do dataset Wisconsin Breast Cancer.

## 🎯 Objetivo

Este projeto tem como objetivo implementar uma solução educacional de apoio à triagem diagnóstica, demonstrando a aplicação de técnicas de Machine Learning em problemas médicos reais, mantendo rigor metodológico na validação e avaliação de modelos.

## 🏥 Contexto do Problema

O câncer de mama é uma das principais preocupações de saúde pública mundial. O diagnóstico precoce e preciso é fundamental para o sucesso do tratamento. Este projeto explora como algoritmos de Machine Learning podem auxiliar profissionais de saúde na análise de características de tumores, fornecendo uma ferramenta complementar de apoio à decisão clínica.

## 📊 Dataset Utilizado

**Wisconsin Breast Cancer Dataset**
- **Fonte:** UCI Machine Learning Repository
- **Amostras:** 569 pacientes
- **Features:** 30 características clínicas (raio, textura, perímetro, etc.)
- **Target:** Diagnóstico (Benigno/Maligno)
- **Formato:** Dados tabulares numéricos

## 🏗️ Estrutura do Projeto

```
tech-challenge-fase1-diagnostico-cancer-mama/
├── 📁 src/                    # Módulos principais
│   ├── preprocessing.py       # Pipeline de pré-processamento
│   ├── train.py              # Treinamento e otimização
│   └── evaluate.py           # Avaliação e métricas
├── 📁 data/                   # Dataset
│   └── breast-cancer-wisconsin.csv
├── 📁 notebooks/              # Análises exploratórias
│   └── analysis.ipynb
├── 📁 models/                 # Modelos treinados (auto)
├── 📄 run_full_pipeline.py    # Pipeline principal
├── 📄 main.py                 # Pipeline original
├── 📄 test_pipeline.py        # Testes funcionais
├── 📄 requirements.txt         # Dependências
├── 📄 Dockerfile              # Containerização
└── 📄 README.md               # Documentação
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.9+** - Linguagem principal
- **Scikit-Learn** - Biblioteca de Machine Learning
- **Pandas & NumPy** - Manipulação de dados
- **Matplotlib & Seaborn** - Visualizações
- **Docker** - Containerização
- **Jupyter** - Análises exploratórias

## 🚀 Como Executar

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

## 🔄 Fluxo do Pipeline

1. **Carregamento e Exploração:** Análise inicial do dataset
2. **Pré-processamento:** Limpeza e transformação dos dados
3. **Divisão Estratificada:** Separação treino/teste (80/20)
4. **Treinamento:** Múltiplos algoritmos com validação cruzada
5. **Otimização:** Grid Search nos melhores modelos
6. **Avaliação Final:** Teste em conjunto separado
7. **Relatório:** Métricas e visualizações

## ⚙️ Pré-processamento Aplicado

- **Remoção de colunas:** ID e colunasUnnamed
- **Tratamento de valores ausentes:** Imputação pela média
- **Encoding target:** M=1 (Maligno), B=0 (Benigno)
- **Normalização:** StandardScaler (média=0, desvio=1)
- **Split estratificado:** Mantém proporção das classes

## 🤖 Modelos Utilizados

1. **Random Forest** - Ensemble de árvores de decisão
2. **SVM (Support Vector Machine)** - Máquina de vetores de suporte
3. **Logistic Regression** - Regressão logística
4. **KNN (K-Nearest Neighbors)** - K vizinhos mais próximos
5. **Naive Bayes** - Classificador bayesiano

## 🔍 Estratégia de Validação

### Divisão dos Dados
O dataset foi dividido de forma estratificada em conjuntos de treino e teste (80/20). O conjunto de teste foi mantido separado e utilizado apenas para avaliação final dos modelos.

### Validação Cruzada
A etapa de validação foi realizada por validação cruzada com 5 folds dentro do conjunto de treino, permitindo:
- Comparação consistente entre diferentes modelos
- Ajuste de hiperparâmetros com Grid Search
- Estimativa robusta da performance esperada
- Evitar vazamento de dados do conjunto de teste

### Importância da Separação
Essa abordagem metodológica garante que:
- O conjunto de teste representa dados nunca vistos
- A seleção do modelo final é baseada apenas em validação cruzada
- A avaliação final é imparcial e confiável
- Não há contaminação entre treinamento e teste

## 📊 Métricas de Avaliação

- **Accuracy** - Taxa de acerto geral
- **Precision** - Proporção de positivos corretos
- **Recall (Sensitivity)** - Capacidade de detectar malignos
- **F1-Score** - Balanceamento precision/recall
- **AUC-ROC** - Discriminação entre classes
- **Matriz de Confusão** - Análise detalhada de erros

## 📈 Resultados Principais

### Performance Esperada
- **Random Forest:** ~97-98% de acurácia
- **SVM:** ~96-97% de acurácia
- **Logistic Regression:** ~95-96% de acurácia
- **KNN:** ~94-95% de acurácia
- **Naive Bayes:** ~92-94% de acurácia

### Features Mais Relevantes
1. `concave points_worst`
2. `perimeter_worst`
3. `radius_worst`
4. `area_worst`
5. `concavity_worst`

## 🧪 Interpretação dos Resultados

Os modelos demonstraram alta capacidade discriminativa entre tumores benignos e malignos, com destaque para Random Forest e SVM. A importância das features indica que características relacionadas à concavidade e perímetro são mais preditivas, o que é clinicamente consistente com a literatura médica.

## ⚠️ Observação Ética e Médica

**AVISO IMPORTANTE:** Este projeto tem finalidade exclusivamente educacional e de demonstração técnica. **O modelo NÃO substitui, de forma alguma, a avaliação médica profissional ou diagnóstico clínico.** 

As decisões médicas devem ser sempre baseadas em:
- Avaliação completa por profissionais qualificados
- Exames clínicos e laboratoriais apropriados
- Histórico completo do paciente
- Contexto clínico individual

Este sistema deve ser considerado apenas como uma ferramenta educacional de apoio à triagem, nunca como substituto de diagnóstico médico.

## 🚀 Próximos Passos (Opcional)

- [ ] Validação com datasets externos
- [ ] Análise de interpretabilidade (SHAP, LIME)
- [ ] Experimentos com ensembles avançados
- [ ] Validação prospectiva em contexto clínico
- [ ] Interface para demonstração educacional

## 📝 Status do Projeto

Este pipeline de Machine Learning está concluído e funcional para fins educacionais e acadêmicos, pronto para documentação final e demonstração das técnicas implementadas.

---

**Desenvolvido como parte do Tech Challenge - Fase 1**  
**Propósito educacional e demonstrativo**  
**Não substitui avaliação médica profissional**