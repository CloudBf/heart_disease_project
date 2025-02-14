# Version simplifiée de train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
from ucimlrepo import fetch_ucirepo

# Chargement des données
heart_disease = fetch_ucirepo(id=45) 
X = heart_disease.data.features 
y = heart_disease.data.targets.values.ravel()

# Création du pipeline avec des paramètres fixes optimisés
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=2,
        subsample=0.8,
        random_state=42
    ))
])

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entraînement
pipeline.fit(X_train, y_train)

# Évaluation
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"Score sur l'ensemble d'entraînement : {train_score:.3f}")
print(f"Score sur l'ensemble de test : {test_score:.3f}")

# Importance des features
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': pipeline.named_steps['classifier'].feature_importances_
})
print("\nImportance des features :")
print(feature_importance.sort_values('importance', ascending=False))

# Sauvegarde du modèle
with open('model.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

# Sauvegarde des importances des features