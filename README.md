# 🥑 Avocado Price Prediction - Full Stack Application

Application de prédiction du prix des avocats utilisant XGBoost avec backend Flask et frontend Streamlit.

## 📁 Structure du projet

```
application/
├── model/
│   ├── avocado_prediction.py   # Script de création du modèle
│   ├── avocado.csv             # Dataset
│   └── avocado_price_model.pkl # Modèle généré (après exécution)
├── back/
│   └── back.py                 # API Flask (Backend)
├── front/
│   └── front.py                # Interface Streamlit (Frontend)
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

## 🚀 Installation et Exécution

### Étape 1 : Créer un environnement virtuel Python

Ouvrez un terminal dans le dossier du projet :

```bash
# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement (Windows)
.venv\Scripts\activate

# Activer l'environnement (Linux/Mac)
source .venv/bin/activate
```

### Étape 2 : Installer les dépendances

```bash
# Naviguer vers le dossier application
cd application

# Installer les dépendances
pip install -r requirements.txt
```

### Étape 3 : Générer le modèle pickle

```bash
# Naviguer vers le dossier model
cd model

# Exécuter le script pour générer le modèle
python avocado_prediction.py
```

Cela génère le fichier `avocado_price_model.pkl`.

### Étape 4 : Lancer le backend Flask

```bash
# Naviguer vers le dossier back
cd ../back

# Lancer l'API Flask
python back.py
```

Le serveur démarre sur : **http://localhost:5000**

### Étape 5 : Lancer le frontend Streamlit

Ouvrir un **NOUVEAU terminal** (garder le backend actif), puis :

```bash
# Activer l'environnement virtuel
.venv\Scripts\activate

# Naviguer vers le dossier front
cd application/front

# Lancer Streamlit
streamlit run front.py
```

Le frontend démarre sur : **http://localhost:8501**

## 🧪 Tester l'API

### Test avec PowerShell

```powershell
# Test de la page d'accueil
Invoke-RestMethod -Uri "http://localhost:5000/" -Method Get

# Test de santé
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method Get

# Test de prédiction
$body = @{
    Quality1 = 5000
    Quality2 = 10000
    Quality3 = 2000
    "Small Bags" = 3000
    "Large Bags" = 500
    "XLarge Bags" = 100
    year = 2023
    type = "organic"
    region = "LosAngeles"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -Body $body -ContentType "application/json"
```

### Test avec curl (Git Bash)

```bash
# Test de prédiction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Quality1": 5000,
    "Quality2": 10000,
    "Quality3": 2000,
    "Small Bags": 3000,
    "Large Bags": 500,
    "XLarge Bags": 100,
    "year": 2023,
    "type": "organic",
    "region": "LosAngeles"
  }'
```

## 📋 Endpoints de l'API

| Méthode | Endpoint         | Description                 |
| ------- | ---------------- | --------------------------- |
| GET     | `/`              | Page d'accueil              |
| GET     | `/health`        | Vérification de santé       |
| GET     | `/features`      | Liste des features requises |
| POST    | `/predict`       | Prédiction du prix          |
| POST    | `/predict_batch` | Prédiction par lot          |

## 📊 Features requises

| Feature     | Type   | Description                 |
| ----------- | ------ | --------------------------- |
| Quality1    | float  | Volume avocats calibre 4046 |
| Quality2    | float  | Volume avocats calibre 4225 |
| Quality3    | float  | Volume avocats calibre 4770 |
| Small Bags  | float  | Nombre de petits sacs       |
| Large Bags  | float  | Nombre de grands sacs       |
| XLarge Bags | float  | Nombre de très grands sacs  |
| year        | int    | Année                       |
| type        | string | "conventional" ou "organic" |
| region      | string | Région (ex: "LosAngeles")   |

## 🎯 Exemple de réponse

```json
{
    "status": "success",
    "prediction": 1.45,
    "unit": "USD",
    "message": "Prix prédit : 1.45 $"
}
```

## 📦 Fichiers Python

| Fichier                       | Description                                          |
| ----------------------------- | ---------------------------------------------------- |
| `model/avocado_prediction.py` | Entraîne le modèle XGBoost et génère le fichier .pkl |
| `back/back.py`                | API Flask pour les prédictions (port 5000)           |
| `front/front.py`              | Interface Streamlit (port 8501)                      |

## 🔄 Architecture du flux

```
┌─────────────────┐      POST /predict       ┌─────────────────┐
│                 │  ─────────────────────►  │                 │
│   FRONTEND      │      JSON data           │    BACKEND      │
│   (Streamlit)   │                          │    (Flask)      │
│   Port 8501     │  ◄─────────────────────  │    Port 5000    │
│                 │      prediction          │                 │
└─────────────────┘                          └─────────────────┘
```

**Flux de données :**

1. L'utilisateur remplit le formulaire dans le **Frontend** (Streamlit)
2. Les données sont envoyées en **JSON** au **Backend** (Flask) via `POST /predict`
3. Le Backend utilise le **modèle XGBoost** pour prédire le prix
4. La **prédiction** est renvoyée au Frontend et affichée à l'utilisateur
