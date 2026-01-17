# SmartLearn AI — AI in Personalized Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## Project Overview
**SmartLearn AI** is an AI-driven personalized learning system that adapts educational content based on student performance, engagement, and learning behavior. Unlike traditional platforms with static content, SmartLearn AI recommends the most suitable learning level (**Easy, Medium, Hard**) for each student using a supervised Random Forest classifier.

---

## Demo & Resources

- **Colab Notebook:** [Colab Link](https://colab.research.google.com/drive/1pTaFC6kOhUKxCPIshSADyGBZrOiwQwHe?usp=sharing)
- **Demo Video:** [Watch Demo](https://drive.google.com/file/d/1MPTbBftjsT9UUioxxq0ghKiJ271tA0HC/view?usp=sharing)
- **Presentation (PPT):** [PPT Link](https://docs.google.com/presentation/d/1txNtDAj2tuE3eqUPPvdpXGaUEtmgzF61ZEityTwQ7hE/edit?slide=id.p#slide=id.p)

---

## Abstract
This project demonstrates how AI can support **adaptive, explainable, and ethical personalized learning**. Using a synthetic dataset of 10,000 student records, the model predicts the next learning level with **62.95% accuracy**, capturing real-world learning uncertainty and effectively personalizing recommendations.

---

## Key Features
- Predicts student learning level (**Easy, Medium, Hard**) based on performance and engagement.
- Explainable AI using **feature importance**.
- Robust to realistic, non-deterministic learning progression.
- Supports educators in designing personalized learning paths.

---

## Methodology

### Data
- Synthetic dataset — 10,000 student records.
- Features: `quiz_score`, `engagement_score`, `attempts`, `time_spent`, `learning_style`, `current_difficulty_level`.
- Preprocessing:
```bash
# Encode categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Standardize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

## Model

Algorithm: Random Forest Classifier

Reason: Robust, interpretable, handles non-linearity

Hyperparameter tuning applied to improve generalization:
```bash
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

## Tools & Technologies
- Python 3.10
- Colab Notebook
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn

## Experiments and Results

Overall Accuracy: 0.6295

- Most misclassifications occur between adjacent levels (Easy ↔ Medium).
- Key predictors: quiz_score and engagement_score.

## Conclusion & Future Work

SmartLearn AI demonstrates realistic, ethical, and explainable personalized learning. Accuracy reflects real-world learning uncertainty rather than overfitting.

Future Improvements

- Incorporate reinforcement learning for continuous adaptation
- Use real-world educational datasets
- Integrate large language models for personalized content generation
- Extend recommendations to include learning format suggestions
