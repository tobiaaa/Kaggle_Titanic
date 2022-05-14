# Kaggle_Titanic
Repository for Kaggle Challenge "Titanic - Machine Learning from Disaster"
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tobiaaa/Kaggle_Titanic/]

## Data Preparation
Not available data is filled with a default value.

| Category          | Data Format      | Method                     |
|-------------------|------------------|----------------------------|
| Class             | {1, 2, 3}        | One-Hot                    |
| Sex               | {male, female}   | One-Hot                    |
| Age               | n.a., [0.83, 88] | Grouped One Hot            |
| Siblings, Spouse  | [0, 8]           | Normalization              |
| Parents, Children | [0, 6]           | Normalization              |
| Fare              | [0, 512]         | Group by Quantile, One-Hot |
| Embarked          | {n.a., S, C, Q}  | One-Hot                    |

