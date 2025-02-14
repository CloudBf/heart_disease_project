# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle

# Import des classifieurs
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from ucimlrepo import fetch_ucirepo

# Chargement des données
heart_disease = fetch_ucirepo(id=45) 
X = heart_disease.data.features 
y = heart_disease.data.targets.values.ravel()

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Création des pipelines pour chaque modèle
models = {
    "Logistic Regression": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ]),
    "KNN": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ]),
    "SVM": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    "AdaBoost": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', AdaBoostClassifier(random_state=42))
    ])
}

# Entraînement et évaluation des modèles
results = {}

for name, pipeline in models.items():
    print(f"\nEntraînement de {name}...")
    
    # Entraînement
    pipeline.fit(X_train, y_train)
    
    # Prédictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba)
    }

# Affichage des résultats
print("\nRésultats de l'évaluation :")
print("-" * 50)
results_df = pd.DataFrame(results).round(3)
print(results_df)

# Identification du meilleur modèle (basé sur F1-Score)
best_model_name = max(results.items(), key=lambda x: x[1]['F1-Score'])[0]
print(f"\nMeilleur modèle (F1-Score) : {best_model_name}")

# Sauvegarde du meilleur modèle
best_model = models[best_model_name]
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Sauvegarde des résultats
results_df.to_pickle('model_results.pkl')