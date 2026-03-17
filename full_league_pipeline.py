import pandas as pd
import numpy as np

# --- SCALING TO FULL PREMIER LEAGUE ---

def fetch_pl_data(seasons):
    """
    Downloads full season CSVs directly from football-data.co.uk.
    Seasons format should be list of strings like '2223', '2324'
    """
    print(f"Fetching Premier League data for seasons: {seasons}...")
    all_seasons = []
    
    for season in seasons:
        # E0 is the code for the English Premier League
        url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
        print(f"Downloading {season}...")
        try:
            df = pd.read_csv(url)
            df['Season'] = season
            all_seasons.append(df)
        except Exception as e:
            print(f"Failed to download {season}: {e}")
            
    # Combine all seasons into one massive dataframe
    full_data = pd.concat(all_seasons, ignore_index=True)
    
    # Drop rows where the match hasn't happened yet (NaNs in result)
    full_data = full_data.dropna(subset=['FTR']) 
    return full_data

def clean_and_engineer(df):
    """Maps columns and engineers rolling features for BOTH teams."""
    print("Cleaning data and engineering features...")
    
    # 1. Keep only the columns we need for the model
    # FTHG = Home Goals, FTAG = Away Goals, HS = Home Shots, AS = Away Shots, 
    # HST = Home Shots on Target, AST = Away Shots Target
    cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'FTR']
    df = df[cols_to_keep].copy()
    
    # Convert dates to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df = df.sort_values("Date")
    
    # 2. Encode categorical data
    df['HomeTeam_Code'] = df['HomeTeam'].astype('category').cat.codes
    df['AwayTeam_Code'] = df['AwayTeam'].astype('category').cat.codes
    df['Day_Code'] = df['Date'].dt.dayofweek
    
    # Target: 1 if Home Team wins, 0 if Draw or Away Team wins
    df['Target'] = (df['FTR'] == 'H').astype(int)
    
    # 3. Calculate rolling averages for Home and Away independently
    # Note: For a true production app, you'd calculate team form regardless of home/away,
    # but for this iteration, we will look at "Home Form" and "Away Form" to keep the logic clean.
    
    home_metrics = ['FTHG', 'HS', 'HST']
    away_metrics = ['FTAG', 'AS', 'AST']
    
    # Calculate rolling 3-game stats for home teams
    home_rolling = (df.groupby('HomeTeam')[home_metrics]
                    .rolling(3, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True))
    home_shifted = df.groupby('HomeTeam')[home_rolling.columns].shift(1)
    home_shifted.columns = [f"{col}_rolling_3" for col in home_metrics]
    
    # Calculate rolling 3-game stats for away teams
    away_rolling = (df.groupby('AwayTeam')[away_metrics]
                    .rolling(3, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True))
    away_shifted = df.groupby('AwayTeam')[away_rolling.columns].shift(1)
    away_shifted.columns = [f"{col}_rolling_3" for col in away_metrics]
    
    # Combine it all
    final_df = pd.concat([df, home_shifted, away_shifted], axis=1)
    
    # Drop the first few weeks of the season where teams have no rolling history
    return final_df.dropna()

if __name__ == "__main__":
    # Let's pull the 2022-2023, 2023-2024, and 2024-2025 seasons
    raw_league_data = fetch_pl_data(['2223', '2324', '2425'])
    
    if not raw_league_data.empty:
        processed_league_data = clean_and_engineer(raw_league_data)
        processed_league_data.to_csv("full_league_data.csv", index=False)
        print("Success! 'full_league_data.csv' is ready.")
        
        # Print out the teams we captured just to verify!
        teams = processed_league_data['HomeTeam'].unique()
        print(f"\nCaptured {len(teams)} teams, including: {', '.join(teams[:5])}...")