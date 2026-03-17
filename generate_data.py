import pandas as pd
import numpy as np
from datetime import timedelta

def create_local_dataset():
    print("Generating local dataset to bypass firewall...")
    
    # Generate 100 match dates across 2023 and 2024
    start_date = pd.to_datetime("2023-08-01")
    dates = [start_date + timedelta(days=7*i) for i in range(100)]
    
    np.random.seed(42) # Keeps the "randomness" consistent
    
    # Build the dataframe with the exact columns our ML model expects
    data = {
        "Date": dates,
        "Team": ["Arsenal"] * 100,
        "Venue_Code": np.random.choice([0, 1], size=100), 
        "Opponent_Code": np.random.randint(0, 19, size=100), 
        "Day_Code": np.random.choice([5, 6], size=100), # Sat/Sun
        "GF_rolling_3": np.random.uniform(0.5, 3.0, size=100).round(2),
        "GA_rolling_3": np.random.uniform(0.5, 2.5, size=100).round(2),
        "Sh_rolling_3": np.random.uniform(8.0, 18.0, size=100).round(1),
        "SoT_rolling_3": np.random.uniform(3.0, 8.0, size=100).round(1),
        "Poss_rolling_3": np.random.uniform(40.0, 65.0, size=100).round(1),
        "Target": np.random.choice([0, 1], size=100, p=[0.3, 0.7]) # 70% win rate
    }
    
    df = pd.DataFrame(data)
    
    # Save it to the exact filename the training script is looking for
    df.to_csv("processed_match_data.csv", index=False)
    print("Success! 'processed_match_data.csv' generated and ready for training.")

if __name__ == "__main__":
    create_local_dataset()