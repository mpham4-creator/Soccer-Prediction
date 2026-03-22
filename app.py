import streamlit as st
import pandas as pd
import joblib

# --- SPRINT 6: THE MODEL SHOWDOWN DEPLOYMENT ---

st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="wide")

# 1. Load the Models and Team Mappings
@st.cache_resource 
def load_assets():
    # Load all 3 models we just generated in our MLOps pipeline
    models = {
        "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "XGBoost": joblib.load("xgboost_model.pkl")
    }
    
    team_mapping = joblib.load("team_mapping.pkl")
    reverse_mapping = {v: k for k, v in team_mapping.items()}
    return models, team_mapping, reverse_mapping

models_dict, team_mapping, reverse_mapping = load_assets()
team_names = list(team_mapping.values())

# --- TEAM LOGO URLs ---
# We are using high-quality SVG links, which often have transparent backgrounds.
team_logos = {
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "Aston Villa": "https://upload.wikimedia.org/wikipedia/en/9/9f/Aston_Villa_logo.svg",
    "Bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
    "Brentford": "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",
    "Brighton": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
    "Burnley": "https://upload.wikimedia.org/wikipedia/en/6/6d/Burnley_FC_Logo.svg",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "Crystal Palace": "https://upload.wikimedia.org/wikipedia/en/0/0c/Crystal_Palace_FC_logo.svg",
    "Everton": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",
    "Fulham": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "Man City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "Man United": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "Newcastle": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "Nott'm Forest": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",
    "Tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
    "West Ham": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
    "Wolves": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers_FC_crest.svg",
    "Leicester": "https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg",
    "Southampton": "https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg",
    "Ipswich": "https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg"
}

# --- STANDARD TRANSPARENT LOGO FUNCTION ---
def display_standard_logo(team_name):
    if team_name in team_logos:
        logo_url = team_logos[team_name]
        # MODIFIED: I changed background-color to transparent and removed the shadow.
        # This keeps the fixed size bounding box but removes the "white part."
        html_code = f"""
        <div style="display: flex; justify-content: center; align-items: center; 
                    width: 100px; height: 100px; 
                    background-color: transparent; padding: 5px;
                    margin-top: 10px;">
            <img src="{logo_url}" style="max-width: 100%; max-height: 100%; object-fit: contain;">
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)
    else:
        # Fallback placeholder (also transparent)
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; 
                    width: 100px; height: 100px; background-color: transparent; 
                    margin-top: 10px; font-size: 50px;">
            ⚽
        </div>
        """, unsafe_allow_html=True)

# 2. Build the User Interface
st.title("⚽ Premier League Match Predictor - Advanced Edition")
st.markdown("Now powered by an automated MLOps pipeline and three distinct ML algorithms.")

# THE NEW FEATURE: The Model Selector Dropdown!
st.sidebar.header("🧠 AI Brain Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose your Prediction Engine:", 
    ["Logistic Regression", "Random Forest", "XGBoost"]
)
st.sidebar.markdown(f"*Currently using: **{selected_model_name}***")

# Set the active model based on the user's dropdown choice
active_model = models_dict[selected_model_name]

st.divider()

# Top Row: Team Selection
col_home, col_away = st.columns(2)

with col_home:
    st.header("🏠 Home Team")
    # Mini-columns: [1 part for standard card, 2 parts for dropdown]
    logo_col, drop_col = st.columns([1, 2]) 
    
    with drop_col:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True) # Spacer for alignment
        home_team = st.selectbox("Select Home Team", team_names, index=0)
        home_code = reverse_mapping[home_team]
        
    with logo_col:
        display_standard_logo(home_team)

with col_away:
    st.header("✈️ Away Team")
    logo_col, drop_col = st.columns([1, 2])
    
    with drop_col:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True) # Spacer
        away_team = st.selectbox("Select Away Team", team_names, index=1)
        away_code = reverse_mapping[away_team]
        
    with logo_col:
        display_standard_logo(away_team)

if home_team == away_team:
    st.warning("⚠️ A team cannot play itself! Please select different teams.")

# Middle Row: Form Sliders
st.subheader("📊 Adjust Recent Form (Last 3 Matches Avg)")
form_col1, form_col2 = st.columns(2)

with form_col1:
    st.markdown(f"**{home_team} (Home Form)**")
    fthg_rolling = st.slider("Goals Scored", 0.0, 5.0, 1.5, key='h_goals')
    hs_rolling = st.slider("Shots Taken", 5.0, 30.0, 14.0, key='h_shots')
    hst_rolling = st.slider("Shots on Target", 0.0, 15.0, 5.0, key='h_sot')

with form_col2:
    st.markdown(f"**{away_team} (Away Form)**")
    ftag_rolling = st.slider("Goals Scored", 0.0, 5.0, 1.0, key='a_goals')
    as_rolling = st.slider("Shots Taken", 5.0, 30.0, 10.0, key='a_shots')
    ast_rolling = st.slider("Shots on Target", 0.0, 15.0, 3.0, key='a_sot')

# Bottom Row: Match Context & Prediction
st.divider()
match_day = st.radio("Match Day", ["Saturday", "Sunday"], horizontal=True)
day_code = 5 if match_day == "Saturday" else 6

# 3. The Prediction Engine
if st.button(f"🔮 Predict using {selected_model_name}", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("Please fix the team selection first.")
    else:
        input_data = pd.DataFrame({
            "HomeTeam_Code": [home_code],
            "AwayTeam_Code": [away_code],
            "Day_Code": [day_code],
            "FTHG_rolling_3": [fthg_rolling],
            "HS_rolling_3": [hs_rolling],
            "HST_rolling_3": [hst_rolling],
            "FTAG_rolling_3": [ftag_rolling],
            "AS_rolling_3": [as_rolling],
            "AST_rolling_3": [ast_rolling]
        })
        
        prediction = active_model.predict(input_data)[0]
        prob_home_win = active_model.predict_proba(input_data)[0][1] 
        
        st.write("---")
        st.subheader("Prediction Results:")
        
        if prediction == 1:
            st.success(f"🏆 {selected_model_name} predicts **{home_team}** will WIN at home!")
            st.progress(float(prob_home_win))
            st.write(f"Confidence: **{prob_home_win * 100:.1f}%**")
        else:
            st.info(f"⚖️ {selected_model_name} predicts a **Draw** or a win for **{away_team}**.")
            st.progress(float(1 - prob_home_win))
            st.write(f"Confidence (Away/Draw): **{(1 - prob_home_win) * 100:.1f}%**")