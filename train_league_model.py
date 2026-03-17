import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
import joblib

# --- SPRINT 4: FULL LEAGUE MODEL TRAINING ---

def train_full_league_model(data_path="full_league_data.csv"):
    print("Loading full Premier League dataset...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Run full_league_pipeline.py first.")
        return

    # Ensure Date is datetime for our chronological split
    df["Date"] = pd.to_datetime(df["Date"])
    
    # 1. Define our Predictors
    # We now use the engineered features for BOTH teams
    predictors = [
        "HomeTeam_Code", "AwayTeam_Code", "Day_Code",
        "FTHG_rolling_3", "HS_rolling_3", "HST_rolling_3", # Home Form
        "FTAG_rolling_3", "AS_rolling_3", "AST_rolling_3"  # Away Form
    ]
    
    # 2. Chronological Train/Test Split
    # We will train the model on the 22/23 and 23/24 seasons, 
    # and test its predictive power on the 24/25 season.
    # This simulates how you would deploy it in the real world today.
    train = df[df["Date"] < "2024-08-01"]
    test = df[df["Date"] >= "2024-08-01"]
    
    if train.empty or test.empty:
        print("Data split error. Check your dates!")
        return

    # 3. Initialize the Model
    # We increase n_estimators (number of trees) because the dataset is much more complex now.
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    
    # 4. Train the Model
    print("Training the Random Forest on historical seasons...")
    rf.fit(train[predictors], train["Target"])
    
    # 5. Make Predictions on the Test Set (The current season)
    preds = rf.predict(test[predictors])
    
    # 6. Evaluate the Model
    acc = accuracy_score(test["Target"], preds)
    precision = precision_score(test["Target"], preds, zero_division=0)
    
    print("\n--- Real-World Model Evaluation (24/25 Season) ---")
    print(f"Accuracy: {acc * 100:.1f}%")
    print(f"Precision (Home Win Confidence): {precision * 100:.1f}%")
    print("--------------------------------------------------\n")
    
    # 7. Save the Model
    model_filename = "pl_model.pkl"
    joblib.dump(rf, model_filename)
    
    # Let's also save the team encodings so our web app knows which number maps to which team!
    team_mapping = dict(enumerate(df['HomeTeam'].astype('category').cat.categories))
    joblib.dump(team_mapping, "team_mapping.pkl")
    
    print(f"Success! Model frozen to {model_filename}")
    print("Team mappings saved to team_mapping.pkl")

if __name__ == "__main__":
    train_full_league_model()
    