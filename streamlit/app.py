# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Au début du code, après les imports
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Maladie Cardiaque",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .risk-high {
        color: red;
        font-size: 24px;
        font-weight: bold;
    }
    .risk-low {
        color: green;
        font-size: 24px;
        font-weight: bold;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger le modèle
# Chargement du modèle
# Définir le chemin absolu vers le fichier du modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model_optimized.pkl')
MODEL_PATH1 = os.path.join(os.path.dirname(__file__), 'model_info.pkl')

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Modèle non trouvé à l'emplacement : {MODEL_PATH}")
        return None
# # Chargement du modèle
# @st.cache_resource
# def load_model():
#     with open('best_model_optimized.pkl', 'rb') as file:
#         return pickle.load(file)

@st.cache_resource
def load_model_info():
    try:  
        with open(MODEL_PATH1, 'rb') as file1:
            return pickle.load(file1)
    except FileNotFoundError:
        st.error(f"Modèle non trouvé à l'emplacement : {MODEL_PATH1}")
        return None
# Chargement du modèle et des informations
model = load_model()
model_info = load_model_info()

# Sidebar pour les informations sur le modèle
with st.sidebar:
    st.header("📊 Informations sur le modèle")
    st.info("""
    - Précision globale : 89%
    - Score AUC-ROC : 0.951
    - Dernière mise à jour : Février 2024
    """)
    
    # Historique des prédictions
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.header("📝 Historique des prédictions")
    for pred in st.session_state.history[-5:]:  # Afficher les 5 dernières prédictions
        st.write(f"{pred['date']} - Risque: {pred['risk']:.1%}")

# Corps principal
st.title("💗🩺Système de Prédiction des Maladies Cardiaques")

# Onglets
tab1, tab2, tab3 = st.tabs(["📝 Saisie des données", "📊 Statistiques", "ℹ️ Aide"])

with tab1:
    # Interface de saisie
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💁 Informations personnelles")
        with st.container(border=True):
            age = st.slider("Âge", 20, 100, 50)
            sex = st.radio("Sexe", ["Homme", "Femme"])
            weight = st.number_input("Poids (kg)", 40, 200, 70)
            height = st.number_input("Taille (cm)", 140, 220, 170)
            
            # Calcul IMC
            bmi = weight / ((height/100) ** 2)
            st.metric("IMC", f"{bmi:.1f}", 
                     delta="Normal" if 18.5 <= bmi <= 25 else "Hors norme")
    
    with col2:
        st.subheader("🏥 Paramètres médicaux")
        with st.container(border=True):
            cp = st.selectbox("Type de douleur thoracique", 
                ["Angine typique", "Angine atypique", "Douleur non angineuse", "Asymptomatique"])
            trestbps = st.slider("Pression artérielle au repos (mm Hg)", 90, 200, 120)
            chol = st.slider("Cholestérol sérique (mg/dl)", 100, 600, 200)
            fbs = st.checkbox("Glycémie à jeun > 120 mg/dl")
    
    # Paramètres avancés dans un expander
    with st.expander("🔬 Paramètres avancés"):
        col3, col4 = st.columns(2)
        
        with col3:
            restecg = st.selectbox("Résultats ECG au repos", 
                ["Normal", "Anomalie ST-T", "Hypertrophie"])
            thalach = st.slider("Fréquence cardiaque maximale", 70, 220, 150)
            exang = st.checkbox("Angine induite par l'exercice")
        
        with col4:
            oldpeak = st.slider("Dépression ST", 0.0, 6.0, 1.0)
            slope = st.selectbox("Pente du segment ST", 
                ["Ascendante", "Plate", "Descendante"])
            ca = st.slider("Nombre de vaisseaux colorés", 0, 3, 0)
            thal = st.selectbox("Thalassémie", 
                ["Normal", "Défaut fixé", "Défaut réversible"])

    # Bouton de prédiction
    if st.button("🔍 Analyser le risque", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            # Préparation des données
            input_data = {
                'age': age,
                'sex': 1 if sex == "Homme" else 0,
                'cp': ["Angine typique", "Angine atypique", "Douleur non angineuse", "Asymptomatique"].index(cp),
                'trestbps': trestbps,
                'chol': chol,
                'fbs': 1 if fbs else 0,
                'restecg': ["Normal", "Anomalie ST-T", "Hypertrophie"].index(restecg),
                'thalach': thalach,
                'exang': 1 if exang else 0,
                'oldpeak': oldpeak,
                'slope': ["Ascendante", "Plate", "Descendante"].index(slope),
                'ca': ca,
                'thal': ["Normal", "Défaut fixé", "Défaut réversible"].index(thal) + 3
            }
            
            X_pred = pd.DataFrame([input_data])
            
            # Prédiction
            prediction = model.predict(X_pred)[0]
            probability = model.predict_proba(X_pred)[0][1]
            
            # Ajout à l'historique
            st.session_state.history.append({
                'date': datetime.now().strftime("%H:%M:%S"),
                'risk': probability
            })
            
            # Affichage des résultats
            st.header("📋 Résultats de l'analyse")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                if prediction == 1:
                    st.error("⚠️ Risque élevé détecté")
                else:
                    st.success("✅ Risque faible")
            
            with col_res2:
                st.metric(
                    label="Probabilité de maladie cardiaque",
                    value=f"{probability:.1%}"
                )
            
            with col_res3:
                confidence = "Élevée" if abs(probability - 0.5) > 0.3 else "Moyenne"
                st.metric(
                    label="Confiance de la prédiction",
                    value=confidence
                )
            
            # Facteurs de risque
            st.subheader("🔍 Analyse des facteurs de risque")
            
            # Création du graphique
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Importance globale des features
            feature_importance = pd.DataFrame({
                'feature': model_info['feature_names'],
                'importance': model.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(data=feature_importance.head(5), x='importance', y='feature', ax=ax1)
            ax1.set_title("Facteurs les plus influents")
            
            # Comparaison avec les valeurs normales
            normal_ranges = {
                'Pression artérielle': {'min': 90, 'max': 120, 'value': trestbps},
                'Cholestérol': {'min': 150, 'max': 200, 'value': chol},
                'Fréq. cardiaque max': {'min': 100, 'max': 170, 'value': thalach}
            }
            
            for i, (param, values) in enumerate(normal_ranges.items()):
                normalized_value = (values['value'] - values['min']) / (values['max'] - values['min'])
                ax2.barh(i, normalized_value, color='g' if values['min'] <= values['value'] <= values['max'] else 'r')
                
            ax2.set_yticks(range(len(normal_ranges)))
            ax2.set_yticklabels(normal_ranges.keys())
            ax2.set_title("Paramètres par rapport aux normes")
            ax2.set_xlim(0, 1.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Recommandations
            st.subheader("💡 Recommandations")
            with st.container(border=True):
                if prediction == 1:
                    st.warning("""
                    ### Actions recommandées :
                    1. 👨‍⚕️ Consultez rapidement un cardiologue
                    2. 📊 Surveillez votre tension artérielle quotidiennement
                    3. 🏃‍♂️ Limitez les efforts physiques intenses
                    4. 🥗 Suivez un régime alimentaire adapté
                    5. 📱 Planifiez un suivi régulier
                    """)
                else:
                    st.info("""
                    ### Recommandations préventives :
                    1. 🏃‍♂️ Maintenez une activité physique régulière
                    2. 🥗 Conservez une alimentation équilibrée
                    3. 📅 Effectuez un contrôle annuel
                    4. 🚭 Évitez le tabac et limitez l'alcool
                    5. 😴 Assurez-vous d'un bon sommeil
                    """)

            # Après avoir fait la prédiction avec le modèle
        new_prediction = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'age': age,
            'sex': sex,
            'trestbps': trestbps,
            'chol': chol,
            'risk': probability
        }
        st.session_state.predictions_history.append(new_prediction)

with tab2:
    st.header("📊 Statistiques des prédictions")
    
    # Vérification s'il y a des prédictions
    if len(st.session_state.predictions_history) > 0:
        # Conversion de l'historique en DataFrame
        df_history = pd.DataFrame(st.session_state.predictions_history)
        
        # 1. Métriques de base
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Nombre de prédictions",
                len(df_history)
            )
        
        with col2:
            high_risk_count = (df_history['risk'] > 0.5).sum()
            st.metric(
                "Cas à risque élevé",
                f"{high_risk_count}"
            )
        
        with col3:
            avg_risk = df_history['risk'].mean()
            st.metric(
                "Risque moyen",
                f"{avg_risk:.1%}"
            )

        # 2. Graphique d'évolution
        st.subheader("Évolution des prédictions")
        fig_evolution = plt.figure(figsize=(10, 6))
        plt.plot(range(len(df_history)), df_history['risk'], 'o-', color='blue')
        plt.title("Évolution du risque prédit")
        plt.xlabel("Numéro de prédiction")
        plt.ylabel("Niveau de risque")
        plt.grid(True)
        st.pyplot(fig_evolution)

        # 3. Distribution des risques
        st.subheader("Distribution des risques")
        fig_dist = plt.figure(figsize=(10, 6))
        plt.hist(df_history['risk'], bins=10, color='skyblue', edgecolor='black')
        plt.title("Distribution des niveaux de risque")
        plt.xlabel("Niveau de risque")
        plt.ylabel("Nombre de prédictions")
        st.pyplot(fig_dist)

        # 4. Tableau des prédictions
        st.subheader("Historique détaillé")
        df_display = df_history.copy()
        df_display['risk'] = df_display['risk'].apply(lambda x: f"{x:.1%}")
        st.dataframe(
            df_display,
            hide_index=True
        )

        # 5. Bouton de téléchargement
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Télécharger l'historique",
            csv,
            "predictions_history.csv",
            "text/csv"
        )
        
    else:
        st.info("Aucune prédiction n'a encore été effectuée.")

with tab3:
    st.header("ℹ️ Aide et Informations")
    st.markdown("""
    ### Comment utiliser l'application
    1. Remplissez vos informations personnelles
    2. Complétez les paramètres médicaux
    3. Si disponible, ajoutez les paramètres avancés
    4. Cliquez sur "Analyser le risque"
    
    ### Comprendre les résultats
    - La prédiction est basée sur un modèle entraîné sur des données réelles
    - Le score de risque va de 0% à 100%
    - Les recommandations sont générées automatiquement
    - IMC : Indice de Masse Corporelle
                
    ### Limitations
    Cette application est un outil d'aide à la décision et ne remplace pas l'avis d'un professionnel de santé.
    """)

# Création d'un espace avant le footer
st.markdown("<br><br>", unsafe_allow_html=True)

# Création du footer avec CSS personnalisé
footer = """
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        padding: 10px;
        text-align: center;
        border-top: 1px solid #cccccc;
    }
    .footer p {
        margin: 0;
        color: #666666;
        font-size: 14px;
    }
    .footer a {
        color: #009688;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
<div class="footer">
    <p>
        Développé par Combarri Christelle & Zongo Ibrahim <br><br>
        Version 1.0 • 
        © 2024
    </p>
</div>
"""
# Affichage du footer
st.markdown(footer, unsafe_allow_html=True)