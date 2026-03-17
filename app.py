import streamlit as st
import pandas as pd
import joblib

# --- SPRINT 5: PREMIER LEAGUE DEPLOYMENT ---

st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="wide")

# 1. Load the Model and Team Mappings
@st.cache_resource 
def load_assets():
    model = joblib.load("pl_model.pkl")
    team_mapping = joblib.load("team_mapping.pkl")
    # Reverse the mapping to easily get the ID from the team name
    reverse_mapping = {v: k for k, v in team_mapping.items()}
    return model, team_mapping, reverse_mapping

rf_model, team_mapping, reverse_mapping = load_assets()
team_names = list(team_mapping.values())

# 2. Build the User Interface
st.title("⚽ Premier League Match Predictor")
st.markdown("""
    Welcome to the ML Predictor! This model uses **Random Forest** trained on historical Premier League seasons to predict upcoming matches.
    Adjust the recent form of both teams to see how it impacts the win probability.
""")
st.divider()

# Top Row: Team Selection
col_home, col_away = st.columns(2)

with col_home:
    st.header("🏠 Home Team")
    home_team = st.selectbox("Select Home Team", team_names, index=0)
    home_code = reverse_mapping[home_team]

with col_away:
    st.header("✈️ Away Team")
    # Default away team to a different index so they don't play themselves
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
if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("Please fix the team selection first.")
    else:
        # Package the data exactly how the model expects it
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
        
        # Get predictions
        prediction = rf_model.predict(input_data)[0]
        # predict_proba returns [Probability of 0 (Draw/Away), Probability of 1 (Home Win)]
        prob_home_win = rf_model.predict_proba(input_data)[0][1] 
        
        st.write("---")
        st.subheader("Prediction Results:")
        
        # We mapped Target=1 to Home Win during training
        if prediction == 1:
            st.success(f"🏆 The AI predicts **{home_team}** will WIN at home!")
            st.progress(float(prob_home_win))
            st.write(f"Confidence: **{prob_home_win * 100:.1f}%**")
        else:
            st.info(f"⚖️ The AI predicts a **Draw** or a win for **{away_team}**.")
            st.progress(float(1 - prob_home_win))
            st.write(f"Confidence (Away/Draw): **{(1 - prob_home_win) * 100:.1f}%**")