# ============================================================================
# 🥑 BACKEND API - PRÉDICTION DU PRIX DES AVOCATS
# ============================================================================
# API Flask pour prédire le prix des avocats en utilisant le modèle XGBoost
# ============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)  # Permet les requêtes cross-origin pour le frontend

# =============================================================================
# CHARGEMENT DU MODÈLE
# =============================================================================

# Chemin vers le fichier pickle du modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'avocado_price_model.pkl')

# Chargement du modèle au démarrage
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modèle chargé avec succès depuis : {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Erreur : Le fichier modèle n'a pas été trouvé à : {MODEL_PATH}")
    print("   Veuillez d'abord exécuter le script avocado_prediction.py pour générer le modèle.")
    model = None

# =============================================================================
# ROUTES DE L'API
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    """Route d'accueil - Vérifie que l'API fonctionne"""
    return jsonify({
        'status': 'success',
        'message': '🥑 API de prédiction du prix des avocats',
        'version': '1.0',
        'endpoints': {
            '/': 'Page d\'accueil (GET)',
            '/health': 'Vérification de santé (GET)',
            '/predict': 'Prédiction du prix (POST)',
            '/features': 'Liste des features requises (GET)'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Route de vérification de santé de l'API"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'message': 'Le modèle est prêt' if model_loaded else 'Le modèle n\'est pas chargé'
    })


@app.route('/features', methods=['GET'])
def get_features():
    """Retourne la liste des features requises pour la prédiction"""
    return jsonify({
        'status': 'success',
        'features': {
            'Quality1': 'Volume d\'avocats calibre 4046 (float)',
            'Quality2': 'Volume d\'avocats calibre 4225 (float)',
            'Quality3': 'Volume d\'avocats calibre 4770 (float)',
            'Small Bags': 'Nombre de petits sacs (float)',
            'Large Bags': 'Nombre de grands sacs (float)',
            'XLarge Bags': 'Nombre de très grands sacs (float)',
            'year': 'Année (int)',
            'type': 'Type d\'avocat : "conventional" ou "organic"',
            'region': 'Région (ex: "LosAngeles", "NewYork", "Albany", etc.)'
        },
        'example': {
            'Quality1': 5000,
            'Quality2': 10000,
            'Quality3': 2000,
            'Small Bags': 3000,
            'Large Bags': 500,
            'XLarge Bags': 100,
            'year': 2023,
            'type': 'organic',
            'region': 'LosAngeles'
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Route de prédiction du prix des avocats
    
    Attend un JSON avec les features suivantes :
    - Quality1, Quality2, Quality3: volumes par calibre
    - Small Bags, Large Bags, XLarge Bags: quantités de sacs
    - year: année
    - type: "conventional" ou "organic"
    - region: région des États-Unis
    
    Retourne le prix prédit en dollars
    """
    
    # Vérification que le modèle est chargé
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Le modèle n\'est pas chargé. Veuillez d\'abord générer le fichier pickle.'
        }), 500
    
    try:
        # Récupération des données JSON
        data = request.get_json()
        
        if data is None:
            return jsonify({
                'status': 'error',
                'message': 'Aucune donnée JSON reçue'
            }), 400
        
        # Liste des features requises
        required_features = ['Quality1', 'Quality2', 'Quality3', 'Small Bags', 
                           'Large Bags', 'XLarge Bags', 'year', 'type', 'region']
        
        # Vérification des features manquantes
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({
                'status': 'error',
                'message': f'Features manquantes : {missing_features}'
            }), 400
        
        # Création du DataFrame pour la prédiction
        input_data = pd.DataFrame({
            'Quality1': [float(data['Quality1'])],
            'Quality2': [float(data['Quality2'])],
            'Quality3': [float(data['Quality3'])],
            'Small Bags': [float(data['Small Bags'])],
            'Large Bags': [float(data['Large Bags'])],
            'XLarge Bags': [float(data['XLarge Bags'])],
            'year': [int(data['year'])],
            'type': [str(data['type'])],
            'region': [str(data['region'])]
        })
        
        # Prédiction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': round(float(prediction), 2),
            'unit': 'USD',
            'message': f'Prix prédit : {prediction:.2f} $',
            'input_data': data
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Erreur de valeur : {str(e)}'
        }), 400
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erreur interne : {str(e)}'
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Route de prédiction par lot
    
    Attend un JSON avec une liste d'objets contenant les features
    """
    
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Le modèle n\'est pas chargé.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({
                'status': 'error',
                'message': 'Les données doivent être une liste d\'objets'
            }), 400
        
        predictions = []
        
        for i, item in enumerate(data):
            input_data = pd.DataFrame({
                'Quality1': [float(item['Quality1'])],
                'Quality2': [float(item['Quality2'])],
                'Quality3': [float(item['Quality3'])],
                'Small Bags': [float(item['Small Bags'])],
                'Large Bags': [float(item['Large Bags'])],
                'XLarge Bags': [float(item['XLarge Bags'])],
                'year': [int(item['year'])],
                'type': [str(item['type'])],
                'region': [str(item['region'])]
            })
            
            pred = model.predict(input_data)[0]
            predictions.append({
                'index': i,
                'prediction': round(float(pred), 2),
                'input': item
            })
        
        return jsonify({
            'status': 'success',
            'count': len(predictions),
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erreur : {str(e)}'
        }), 500


# =============================================================================
# LANCEMENT DU SERVEUR
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🥑 DÉMARRAGE DU BACKEND - API AVOCADO PRICE PREDICTION")
    print("=" * 60)
    print(f"\n🌐 URL : http://localhost:5000")
    print("\n📋 Endpoints disponibles :")
    print("   - GET  /          : Page d'accueil")
    print("   - GET  /health    : Vérification de santé")
    print("   - GET  /features  : Liste des features")
    print("   - POST /predict   : Prédiction du prix")
    print("   - POST /predict_batch : Prédiction par lot")
    print("\n" + "=" * 60)
    
    # Lancement du serveur Flask
    app.run(host='0.0.0.0', port=5000, debug=True)

