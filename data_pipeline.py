import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from io import StringIO

# --- SPRINT 2: DATA ACQUISITION ---
def scrape_fbref_data(team_url, team_name):
    """
    Efficiently scrapes match logs using Pandas read_html.
    Includes a sleep timer to respect server rate limits.
    """
    print(f"Scraping data for {team_name}...")
    
    # Spoofing a standard browser header to help bypass basic bot detection
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(team_url, headers=headers)
    
    # Using StringIO to prevent Pandas FutureWarning
    try:
        matches = pd.read_html(StringIO(response.text), match="Scores & Fixtures")[0]
    except ValueError:
        return pd.DataFrame() # Return empty DF if the table isn't found or blocked
    
    # Clean up the raw scrape
    matches['Team'] = team_name
    
    # Be polite to the server
    time.sleep(3) 
    return matches

# --- SPRINT 3: FEATURE ENGINEERING ---
def add_rolling_features(df, metrics, window=3):
    """
    Highly efficient, vectorized rolling average calculation.
    """
    # 1. We MUST sort by date first
    df = df.sort_values("Date")
    
    # 2. Calculate rolling averages using groupby. No loops!
    rolling_stats = (df.groupby("Team")[metrics]
                     .rolling(window, min_periods=window)
                     .mean()
                     .reset_index(level=0, drop=True))
    
    # 3. CRITICAL: Shift the data down by 1 row to prevent data leakage.
    shifted_stats = df.groupby("Team")[rolling_stats.columns].shift(1)
    
    # Rename columns to indicate they are rolling averages
    shifted_stats.columns = [f"{col}_rolling_{window}" for col in metrics]
    
    # Combine the new features with the original dataframe
    df = pd.concat([df, shifted_stats], axis=1)
    
    # Drop rows with NaN values (the first few games won't have a history)
    return df.dropna(subset=shifted_stats.columns)

def clean_and_prepare_data(df):
    """Transforms raw strings into machine-readable numeric features."""
    
    # Filter only Premier League matches (drop cup games)
    if 'Comp' in df.columns:
        df = df[df['Comp'] == 'Premier League'].copy()
    
    # Convert Dates to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert categorical strings to numeric codes for the ML model
    df['Venue_Code'] = df['Venue'].astype('category').cat.codes # Home=1, Away=0
    df['Opponent_Code'] = df['Opponent'].astype('category').cat.codes
    df['Day_Code'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
    
    # Create the Target Variable: 1 if we won, 0 if draw/loss
    df['Target'] = (df['Result'] == 'W').astype(int)
    
    return df

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Example URL for Arsenal's 2023-2024 season
    arsenal_url = "https://fbref.com/en/squads/18bb7c10/2023-2024/matchlogs/c9/schedule/Arsenal-Scores-and-Fixtures-Premier-League"
    
    # 1. Extract
    raw_df = scrape_fbref_data(arsenal_url, "Arsenal")
    
    if not raw_df.empty:
        # 2. Clean
        cleaned_df = clean_and_prepare_data(raw_df)
        
        # 3. Transform (Feature Engineering)
        # Goals For, Goals Against, Shots, Shots on Target, Possession
        metrics_to_roll = ['GF', 'GA', 'Sh', 'SoT', 'Poss'] 
        
        # Convert string metrics to numeric first
        for col in metrics_to_roll:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
        final_df = add_rolling_features(cleaned_df, metrics_to_roll)
        
        # 4. Load (Save for the ML model)
        final_df.to_csv("processed_match_data.csv", index=False)
        print("Data pipeline complete. File saved as processed_match_data.csv")
    else:
        print("Failed to fetch data. The server might be blocking the request.")