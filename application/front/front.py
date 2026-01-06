# ============================================================================
# 🥑 FRONTEND STREAMLIT - PRÉDICTION DU PRIX DES AVOCATS
# ============================================================================
# Interface utilisateur pour prédire le prix des avocats
# ============================================================================

import streamlit as st
import requests
import json

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="🥑 Avocado Price Predictor",
    page_icon="🥑",
    layout="centered",
    initial_sidebar_state="expanded"
)

# URL de l'API Backend
API_URL = "http://localhost:5000"

# =============================================================================
# STYLES CSS PERSONNALISÉS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #81C784 0%, #4CAF50 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-price {
        font-size: 3rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #388E3C;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# EN-TÊTE
# =============================================================================
st.markdown('<p class="main-header">🥑 Avocado Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prédisez le prix moyen des avocats aux États-Unis</p>', unsafe_allow_html=True)

# =============================================================================
# VÉRIFICATION DE LA CONNEXION AU BACKEND
# =============================================================================
def check_api_health():
    """Vérifie si l'API backend est disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Affichage du statut de l'API
api_status = check_api_health()
if api_status:
    st.sidebar.success("✅ Backend connecté")
else:
    st.sidebar.error("❌ Backend non disponible")
    st.error("⚠️ Le backend n'est pas accessible. Assurez-vous que le serveur Flask est lancé sur http://localhost:5000")
    st.code("cd application/back\npython back.py", language="bash")
    st.stop()

# =============================================================================
# SIDEBAR - INFORMATIONS
# =============================================================================
st.sidebar.markdown("## 📊 À propos")
st.sidebar.markdown("""
Cette application utilise un modèle **XGBoost** entraîné sur des données 
historiques de ventes d'avocats aux États-Unis.

**Métriques du modèle :**
- 📈 R² ≈ 0.85
- 📉 RMSE ≈ 0.15 $
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔗 API Endpoints")
st.sidebar.code(f"""
GET  {API_URL}/
GET  {API_URL}/health
GET  {API_URL}/features
POST {API_URL}/predict
""")

# =============================================================================
# FORMULAIRE DE PRÉDICTION
# =============================================================================
st.markdown("### 📝 Entrez les caractéristiques de l'avocat")

# Création de colonnes pour un meilleur layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📦 Volumes par calibre")
    quality1 = st.number_input(
        "Quality1 (calibre 4046)", 
        min_value=0.0, 
        max_value=1000000.0, 
        value=5000.0,
        help="Volume d'avocats de petit calibre"
    )
    quality2 = st.number_input(
        "Quality2 (calibre 4225)", 
        min_value=0.0, 
        max_value=1000000.0, 
        value=10000.0,
        help="Volume d'avocats de calibre moyen"
    )
    quality3 = st.number_input(
        "Quality3 (calibre 4770)", 
        min_value=0.0, 
        max_value=1000000.0, 
        value=2000.0,
        help="Volume d'avocats de gros calibre"
    )

with col2:
    st.markdown("#### 🛍️ Quantité de sacs")
    small_bags = st.number_input(
        "Small Bags", 
        min_value=0.0, 
        max_value=1000000.0, 
        value=3000.0,
        help="Nombre de petits sacs"
    )
    large_bags = st.number_input(
        "Large Bags", 
        min_value=0.0, 
        max_value=1000000.0, 
        value=500.0,
        help="Nombre de grands sacs"
    )
    xlarge_bags = st.number_input(
        "XLarge Bags", 
        min_value=0.0, 
        max_value=1000000.0, 
        value=100.0,
        help="Nombre de très grands sacs"
    )

st.markdown("---")

# Deuxième ligne de paramètres
col3, col4, col5 = st.columns(3)

with col3:
    year = st.selectbox(
        "📅 Année",
        options=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        index=8,
        help="Année de la prédiction"
    )

with col4:
    avocado_type = st.selectbox(
        "🏷️ Type d'avocat",
        options=["conventional", "organic"],
        format_func=lambda x: "🌱 Bio (organic)" if x == "organic" else "🥑 Conventionnel",
        help="Type d'avocat : conventionnel ou biologique"
    )

with col5:
    # Liste des régions disponibles
    regions = [
        "Albany", "Atlanta", "BaltimoreWashington", "Boise", "Boston",
        "BuffaloRochester", "California", "Charlotte", "Chicago", "CincinnatiDayton",
        "Columbus", "DallasFtWorth", "Denver", "Detroit", "GrandRapids",
        "GreatLakes", "HarrisburgScranton", "HartfordSpringfield", "Houston", "Indianapolis",
        "Jacksonville", "LasVegas", "LosAngeles", "Louisville", "MiamiFtLauderdale",
        "Midsouth", "Nashville", "NewOrleansMobile", "NewYork", "Northeast",
        "NorthernNewEngland", "Orlando", "Philadelphia", "PhoenixTucson", "Pittsburgh",
        "Plains", "Portland", "RaleighGreensboro", "RichmondNorfolk", "Roanoke",
        "Sacramento", "SanDiego", "SanFrancisco", "Seattle", "SouthCarolina",
        "SouthCentral", "Southeast", "Spokane", "StLouis", "Syracuse",
        "Tampa", "TotalUS", "West", "WestTexNewMexico"
    ]
    
    region = st.selectbox(
        "🌍 Région",
        options=regions,
        index=regions.index("LosAngeles") if "LosAngeles" in regions else 0,
        help="Région des États-Unis"
    )

# =============================================================================
# BOUTON DE PRÉDICTION
# =============================================================================
st.markdown("---")

if st.button("🔮 Prédire le prix", use_container_width=True):
    # Préparation des données
    data = {
        "Quality1": quality1,
        "Quality2": quality2,
        "Quality3": quality3,
        "Small Bags": small_bags,
        "Large Bags": large_bags,
        "XLarge Bags": xlarge_bags,
        "year": year,
        "type": avocado_type,
        "region": region
    }
    
    # Appel à l'API
    with st.spinner("🔄 Calcul en cours..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", 0)
                
                # Affichage du résultat
                st.markdown(f"""
                <div class="prediction-box">
                    <p style="margin:0; font-size: 1.2rem;">💰 Prix prédit</p>
                    <p class="prediction-price">{prediction:.2f} $</p>
                    <p style="margin:0; font-size: 0.9rem;">par avocat</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Détails de la prédiction
                with st.expander("📋 Détails de la requête"):
                    st.json(data)
                    st.json(result)
                
                # Interprétation
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                if prediction < 1.0:
                    st.success(f"✅ Prix bas ! Les avocats sont abordables à {prediction:.2f} $")
                elif prediction < 1.5:
                    st.info(f"ℹ️ Prix moyen. Les avocats coûtent {prediction:.2f} $")
                else:
                    st.warning(f"⚠️ Prix élevé ! Les avocats coûtent {prediction:.2f} $")
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.error(f"❌ Erreur API : {response.status_code}")
                st.json(response.json())
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de se connecter au backend. Vérifiez que le serveur est lancé.")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    🥑 Avocado Price Predictor | Projet EPSI - Atelier IA Générative<br>
    Modèle XGBoost | Backend Flask | Frontend Streamlit
</div>
""", unsafe_allow_html=True)

