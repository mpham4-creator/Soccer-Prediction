import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score
import joblib

# --- SPRINT 6: THE MODEL SHOWDOWN ---

def train_and_compare_models(data_path="full_league_data.csv"):
    print("Loading full Premier League dataset...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}.")
        return

    df["Date"] = pd.to_datetime(df["Date"])
    
    predictors = [
        "HomeTeam_Code", "AwayTeam_Code", "Day_Code",
        "FTHG_rolling_3", "HS_rolling_3", "HST_rolling_3",
        "FTAG_rolling_3", "AS_rolling_3", "AST_rolling_3"
    ]
    
    # Chronological Split
    train = df[df["Date"] < "2024-08-01"]
    test = df[df["Date"] >= "2024-08-01"]
    
    # Initialize our three competitors
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42, eval_metric='logloss')
    }
    
    print("\n⚔️ --- THE MODEL SHOWDOWN (Test on 24/25 Season) --- ⚔️")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train[predictors], train["Target"])
        
        preds = model.predict(test[predictors])
        
        acc = accuracy_score(test["Target"], preds)
        precision = precision_score(test["Target"], preds, zero_division=0)
        
        print(f"Results for {name}:")
        print(f" - Accuracy: {acc * 100:.1f}%")
        print(f" - Precision: {precision * 100:.1f}%")
        
        # Save each model so our Streamlit app can use them
        filename = f"{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, filename)
    
    # Save team mappings
    team_mapping = dict(enumerate(df['HomeTeam'].astype('category').cat.categories))
    joblib.dump(team_mapping, "team_mapping.pkl")
    print("\n✅ All models trained and saved successfully!")

if __name__ == "__main__":
    train_and_compare_models()