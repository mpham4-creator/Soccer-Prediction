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
    home_team = st.selectbox("Select Home Team", team_names, index=0)
    home_code = reverse_mapping[home_team]

with col_away:
    st.header("✈️ Away Team")
    away_team = st.selectbox("Select Away Team", team_names, index=1)
    away_code = reverse_mapping[away_team]

if home_team == away_team:
    st.warning("⚠️ A team cannot play itself! Please select different teams.")

st.divider()

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
# Notice how the button dynamically changes its text based on your model choice!
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