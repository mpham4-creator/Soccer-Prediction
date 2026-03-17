import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import joblib

# --- SPRINT 4: MODEL TRAINING & EVALUATION ---

def train_and_save_model(data_path="processed_match_data.csv"):
    """
    Loads processed data, trains a Random Forest model chronologically,
    evaluates its precision, and saves it for deployment.
    """
    print("Loading data...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Run data_pipeline.py first.")
        return

    # Ensure Date is a datetime object for our chronological split
    df["Date"] = pd.to_datetime(df["Date"])
    
    # 1. Define our Predictors (Features) and Target
    # We use the categorical codes and the rolling averages we engineered
    predictors = [
        "Venue_Code", "Opponent_Code", "Day_Code",
        "GF_rolling_3", "GA_rolling_3", "Sh_rolling_3", "SoT_rolling_3", "Poss_rolling_3"
    ]
    
    # 2. Chronological Train/Test Split
    # In a full-scale app, you'd train on previous seasons and test on the current one.
    # For this MVP, we will split the season down the middle (e.g., Jan 1st).
    train = df[df["Date"] < "2024-01-01"]
    test = df[df["Date"] >= "2024-01-01"]
    
    if train.empty or test.empty:
        print("Not enough data to split. Ensure your CSV has matches before and after Jan 1, 2024.")
        return

    # 3. Initialize the Model
    # Random Forest is great here because it can pick up non-linear patterns.
    # n_estimators=50 (50 decision trees), min_samples_split=10 (prevents overfitting)
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=42)
    
    # 4. Train the Model
    print("Training the Random Forest model...")
    rf.fit(train[predictors], train["Target"])
    
    # 5. Make Predictions on the Test Set
    preds = rf.predict(test[predictors])
    
    # 6. Evaluate the Model
    acc = accuracy_score(test["Target"], preds)
    precision = precision_score(test["Target"], preds, zero_division=0)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {acc * 100:.1f}%")
    print(f"Precision: {precision * 100:.1f}%")
    print("------------------------\n")
    
    # 7. Save the Model (Pickling)
    model_filename = "soccer_model.pkl"
    joblib.dump(rf, model_filename)
    print(f"Success! Model frozen and saved to {model_filename}")

if __name__ == "__main__":
    # Note: To get reliable accuracy, make sure your data_pipeline.py 
    # scraped more than just one team's data before running this!
    train_and_save_model()