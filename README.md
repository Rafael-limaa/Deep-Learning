# 🍷 Classificação da Qualidade de Vinhos com Redes Neurais MLP

Este projeto aplica técnicas de Deep Learning com **Redes Neurais Multicamadas (MLP)** para prever a qualidade de vinhos com base em atributos fisico-químicos. Utilizamos o [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) disponível no Kaggle.

---

## 🎯 Objetivo

Construir, treinar e comparar diferentes arquiteturas de redes neurais do tipo MLP (Multi-Layer Perceptron), avaliando o impacto de hiperparâmetros como:
- Número de camadas ocultas
- Quantidade de neurônios por camada
- Função de ativação
- Tamanho do batch
- Uso de técnicas de regularização como **Early Stopping**

---

## 🧪 Base de Dados

- Fonte: Kaggle – [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- Tipo: CSV
- Total de amostras: 1.149 registros
- Atributos:
  - 11 variáveis independentes (ex: pH, álcool, acidez volátil)
  - 1 variável dependente: `quality` (nota de 3 a 8)

---

## 🧠 Modelos MLP Avaliados

| Modelo | Arquitetura               | Observações                         |
|--------|---------------------------|-------------------------------------|
| 1      | 1 camada oculta (16)      | Arquitetura simples (baseline)      |
| 2      | 2 camadas (32 → 16)       | Aprendizado mais profundo           |
| 3      | 2 camadas + batch_size=64 | Variação no tamanho do lote         |
| 4      | 2 camadas + EarlyStopping | Regularização contra overfitting    |

Todos os modelos utilizam:
- Função de ativação: `ReLU` (ocultas) e `softmax` (saída)
- Otimizador: `Adam`
- Perda: `categorical_crossentropy`
- Treino por até 50 épocas (ou até parada antecipada)

---

## ⚙️ Tecnologias Utilizadas

- Python 3.10+
- TensorFlow / Keras
- Scikit-learn
- Pandas
- Matplotlib e Seaborn
- Google Colab (ambiente de desenvolvimento)

---

## 📊 Análises Realizadas

- **Visualização exploratória dos dados** (distribuição da qualidade, correlação, boxplots)
- **Normalização** dos dados com MinMaxScaler
- **Codificação one-hot** da variável target
- **Divisão em treino e teste (70/30)**
- **Treinamento de múltiplos modelos MLP**
- **Comparação com gráficos de acurácia (treino x validação)**
- **Avaliação com classificação multiclasse**:
  - Acurácia
  - Relatório de classificação (precision, recall, f1-score)
  - Matriz de confusão

---

## 📌 Resultados Obtidos

- O modelo com duas camadas ocultas (32 → 16 neurônios) + EarlyStopping obteve o melhor desempenho, equilibrando acurácia e generalização.
- A maioria dos erros ocorre entre classes próximas (ex: qualidade 5 confundida com 6).
- Adições como batch size e parada antecipada impactaram positivamente na estabilidade do modelo.

---

## 📁 Estrutura do Projeto

