# ============================================================================
# 🥑 MODÈLE PRÉDICTIF POUR LE PRIX DES AVOCATS
# ============================================================================
# Ce script prédit le prix moyen des avocats aux États-Unis en utilisant
# un modèle de machine learning (XGBoost).
# ============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

print("✅ Bibliothèques importées avec succès !")

# =============================================================================
# ÉTAPE 1 : IMPORTER ET EXPLORER LES DONNÉES
# =============================================================================
print("\n" + "=" * 60)
print("📊 ÉTAPE 1 : IMPORTER ET EXPLORER LES DONNÉES")
print("=" * 60)

# 1.1 Charger les données
df = pd.read_csv('avocado.csv')
print(f"\n📊 Dimensions du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print("\n📋 Aperçu des 5 premières lignes :")
print(df.head())

# 1.2 Informations sur le dataset
print("\n📋 Informations sur les colonnes :")
print(df.info())

# 1.3 Supprimer les colonnes inutiles
colonnes_a_supprimer = ['Unnamed: 0', 'Total Volume', 'Total Bags']
print(f"\n🗑️ Suppression des colonnes : {colonnes_a_supprimer}")
df = df.drop(columns=colonnes_a_supprimer)
print(f"✅ Nouvelles dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# 1.4 Renommer les colonnes
renommage = {
    '4046': 'Quality1',
    '4225': 'Quality2',
    '4770': 'Quality3'
}
df = df.rename(columns=renommage)
print(f"\n✅ Colonnes renommées : {renommage}")
print(f"📋 Nouvelles colonnes : {list(df.columns)}")

# 1.5 Convertir les dates
print(f"\n📅 Type actuel de 'Date' : {df['Date'].dtype}")
df['Date'] = pd.to_datetime(df['Date'])
print(f"✅ Type après conversion : {df['Date'].dtype}")
print(f"📆 Période couverte : du {df['Date'].min().strftime('%d/%m/%Y')} au {df['Date'].max().strftime('%d/%m/%Y')}")

# 1.6 Vérification des valeurs manquantes
print("\n🔍 Analyse des valeurs manquantes :")
valeurs_manquantes = df.isnull().sum()
print(valeurs_manquantes)
total_manquants = valeurs_manquantes.sum()
if total_manquants == 0:
    print("✅ Aucune valeur manquante dans le dataset !")
else:
    print(f"⚠️ Total de valeurs manquantes : {total_manquants}")

# 1.7 Vérification et suppression des doublons
nb_doublons = df.duplicated().sum()
print(f"\n🔍 Nombre de lignes dupliquées : {nb_doublons}")
if nb_doublons > 0:
    print(f"⚠️ {nb_doublons} doublons détectés ! Suppression en cours...")
    df = df.drop_duplicates()
    print(f"✅ Doublons supprimés. Nouvelles dimensions : {df.shape}")
else:
    print("✅ Aucun doublon détecté !")

# 1.8 Résumé du dataset nettoyé
print("\n" + "=" * 60)
print("📊 RÉSUMÉ DU DATASET NETTOYÉ")
print("=" * 60)
print(f"📈 Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"🎯 Variable cible : AveragePrice")
print(f"   - Min : {df['AveragePrice'].min():.2f} $")
print(f"   - Max : {df['AveragePrice'].max():.2f} $")
print(f"   - Moyenne : {df['AveragePrice'].mean():.2f} $")
print(f"🏷️ Types d'avocats : {df['type'].unique().tolist()}")
print(f"🌍 Nombre de régions : {df['region'].nunique()}")

# =============================================================================
# ÉTAPE 2 : PRÉPARER LES DONNÉES POUR LE MODÈLE
# =============================================================================
print("\n" + "=" * 60)
print("🔧 ÉTAPE 2 : PRÉPARER LES DONNÉES POUR LE MODÈLE")
print("=" * 60)

# 2.1 Définition des features (X) et de la cible (y)
X = df.drop(columns=['AveragePrice', 'Date'])
y = df['AveragePrice']

print(f"\n🎯 Variable cible (y) : AveragePrice")
print(f"📊 Dimensions de X : {X.shape}")
print(f"📊 Dimensions de y : {y.shape}")

# 2.2 Définition des colonnes numériques et catégoriques
colonnes_numeriques = ['Quality1', 'Quality2', 'Quality3', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year']
colonnes_categoriques = ['type', 'region']

print(f"\n🔢 Colonnes numériques ({len(colonnes_numeriques)}) : {colonnes_numeriques}")
print(f"🏷️ Colonnes catégoriques ({len(colonnes_categoriques)}) : {colonnes_categoriques}")

# 2.3 Création du ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), colonnes_numeriques),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), colonnes_categoriques)
    ],
    remainder='drop'
)

print("\n✅ ColumnTransformer créé avec succès !")
print("   - Colonnes numériques → StandardScaler")
print("   - Colonnes catégoriques → OneHotEncoder")

# 2.4 Division des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Conversion explicite en DataFrame/Series pandas
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

print(f"\n📊 Division des données :")
print(f"   - Ensemble d'entraînement : {len(X_train)} échantillons (80%)")
print(f"   - Ensemble de test : {len(X_test)} échantillons (20%)")

# =============================================================================
# ÉTAPE 3 : CONSTRUIRE ET ENTRAÎNER LE MODÈLE
# =============================================================================
print("\n" + "=" * 60)
print("🤖 ÉTAPE 3 : CONSTRUIRE ET ENTRAÎNER LE MODÈLE")
print("=" * 60)

# 3.1 Création du modèle XGBRegressor
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

# 3.2 Création du pipeline complet
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb_model)
])

print("\n✅ Pipeline créé avec succès !")
print("   - Étape 1 : Prétraitement (ColumnTransformer)")
print("   - Étape 2 : Modèle (XGBRegressor)")

# 3.3 Entraînement du modèle
print("\n🚀 Entraînement du modèle en cours...")
start_time = time.time()

pipeline.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"✅ Modèle entraîné avec succès !")
print(f"⏱️ Temps d'entraînement : {training_time:.2f} secondes")

# =============================================================================
# ÉTAPE 4 : ÉVALUATION ET SAUVEGARDE DU MODÈLE
# =============================================================================
print("\n" + "=" * 60)
print("📈 ÉTAPE 4 : ÉVALUATION ET SAUVEGARDE DU MODÈLE")
print("=" * 60)

# 4.1 Prédictions
y_pred = pipeline.predict(X_test)
y_train_pred = pipeline.predict(X_train)

print("\n✅ Prédictions effectuées !")

# 4.2 Calcul des métriques
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test = r2_score(y_test, y_pred)

print("\n" + "-" * 50)
print("📊 PERFORMANCES DU MODÈLE")
print("-" * 50)
print(f"\n🎓 Ensemble d'ENTRAÎNEMENT :")
print(f"   - RMSE : {rmse_train:.4f} $")
print(f"   - R²   : {r2_train:.4f} ({r2_train*100:.2f}%)")

print(f"\n🧪 Ensemble de TEST :")
print(f"   - RMSE : {rmse_test:.4f} $")
print(f"   - R²   : {r2_test:.4f} ({r2_test*100:.2f}%)")

print(f"\n📈 Interprétation :")
print(f"   - Le modèle explique {r2_test*100:.1f}% de la variance des prix")
print(f"   - L'erreur moyenne de prédiction est de ±{rmse_test:.3f} $")

# Vérification du surapprentissage
if r2_train - r2_test > 0.1:
    print("\n⚠️ Attention : Possible surapprentissage détecté !")
else:
    print("\n✅ Pas de surapprentissage significatif détecté")

# 4.3 Comparaison des prédictions
print("\n📋 Comparaison des 10 premières prédictions :")
y_test_array = np.array(y_test)
comparaison = pd.DataFrame({
    'Prix Réel ($)': np.round(y_test_array[:10], 2),
    'Prix Prédit ($)': np.round(y_pred[:10], 2),
    'Erreur ($)': np.round(y_test_array[:10] - y_pred[:10], 3)
})
print(comparaison.to_string(index=False))

# 4.4 Visualisations
print("\n📊 Génération des graphiques...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : Scatter plot
ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
y_min, y_max = float(np.min(y_test)), float(np.max(y_test))
ax1.plot([y_min, y_max], [y_min, y_max], 'r--', lw=2, label='Prédiction parfaite')
ax1.set_xlabel('Prix Réel ($)', fontsize=12)
ax1.set_ylabel('Prix Prédit ($)', fontsize=12)
ax1.set_title(f'Prédictions vs Valeurs Réelles\nR² = {r2_test:.4f}', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graphique 2 : Distribution des erreurs
ax2 = axes[1]
erreurs = y_test - y_pred
ax2.hist(erreurs, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erreur = 0')
ax2.set_xlabel('Erreur de prédiction ($)', fontsize=12)
ax2.set_ylabel('Fréquence', fontsize=12)
ax2.set_title(f'Distribution des Erreurs\nRMSE = {rmse_test:.4f} $', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Graphiques sauvegardés : model_evaluation.png")

# 4.5 Sauvegarde du modèle
nom_fichier = 'avocado_price_model.pkl'
joblib.dump(pipeline, nom_fichier)

taille_fichier = os.path.getsize(nom_fichier) / (1024 * 1024)

print("\n" + "-" * 50)
print("💾 SAUVEGARDE DU MODÈLE")
print("-" * 50)
print(f"✅ Pipeline sauvegardé avec succès !")
print(f"📁 Fichier : {nom_fichier}")
print(f"📦 Taille : {taille_fichier:.2f} MB")

# 4.6 Vérification du chargement
pipeline_charge = joblib.load(nom_fichier)
y_pred_verif = pipeline_charge.predict(X_test.head(5))

if np.allclose(y_pred[:5], y_pred_verif):
    print("✅ Vérification : Le modèle se charge et fonctionne correctement !")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "🥑" * 30)
print("\n" + "=" * 60)
print("📊 RÉSUMÉ FINAL DU PROJET")
print("=" * 60)

print(f"\n📁 DONNÉES :")
print(f"   - Dataset original : {len(df)} échantillons")
print(f"   - Features : {X.shape[1]} colonnes")
print(f"   - Entraînement : {len(X_train)} échantillons (80%)")
print(f"   - Test : {len(X_test)} échantillons (20%)")

print(f"\n🔧 PRÉTRAITEMENT :")
print(f"   - StandardScaler sur {len(colonnes_numeriques)} colonnes numériques")
print(f"   - OneHotEncoder sur {len(colonnes_categoriques)} colonnes catégoriques")

print(f"\n🤖 MODÈLE : XGBRegressor")
print(f"   - n_estimators : 100")
print(f"   - max_depth : 6")
print(f"   - learning_rate : 0.1")

print(f"\n📈 PERFORMANCES (Test) :")
print(f"   - RMSE : {rmse_test:.4f} $")
print(f"   - R²   : {r2_test:.4f} ({r2_test*100:.2f}%)")

print(f"\n💾 FICHIERS GÉNÉRÉS :")
print(f"   - {nom_fichier} (modèle)")
print(f"   - model_evaluation.png (graphiques)")

print("\n" + "=" * 60)
print("🚀 Le modèle est prêt à être utilisé !")
print("=" * 60)
print("\n" + "🥑" * 30)

# =============================================================================
# EXEMPLE D'UTILISATION DU MODÈLE
# =============================================================================
print("\n" + "=" * 60)
print("🔮 EXEMPLE D'UTILISATION DU MODÈLE")
print("=" * 60)

# Exemple de prédiction
exemple = pd.DataFrame({
    'Quality1': [5000],
    'Quality2': [10000],
    'Quality3': [2000],
    'Small Bags': [3000],
    'Large Bags': [500],
    'XLarge Bags': [100],
    'year': [2023],
    'type': ['organic'],
    'region': ['LosAngeles']
})

prix_predit = pipeline_charge.predict(exemple)[0]

print("\n📋 Caractéristiques de l'avocat :")
for col in exemple.columns:
    print(f"   - {col}: {exemple[col].values[0]}")

print(f"\n💰 Prix prédit : {prix_predit:.2f} $")
print("=" * 60)

