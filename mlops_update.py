import os
import subprocess

# --- SPRINT 6: MLOPS AUTOMATED PIPELINE ---

def run_pipeline():
    print("🚀 Initiating Full MLOps Pipeline Update...")
    
    # 1. Trigger the data pipeline (fetch new data & engineer features)
    print("\n--- PHASE 1: DATA INGESTION & ENGINEERING ---")
    data_script = "full_league_pipeline.py"
    if os.path.exists(data_script):
        # We use subprocess to run the other Python file programmatically
        subprocess.run(["c:\\python314\\python.exe", data_script], check=True)
    else:
        print(f"Error: Cannot find {data_script}")
        return

    # 2. Trigger the model showdown (train & evaluate)
    print("\n--- PHASE 2: MODEL RETRAINING & EVALUATION ---")
    train_script = "train_league_model.py"
    if os.path.exists(train_script):
        subprocess.run(["c:\\python314\\python.exe", train_script], check=True)
    else:
        print(f"Error: Cannot find {train_script}")
        return
        
    print("\n🎉 MLOps Pipeline Complete! App is ready for production with the latest data.")

if __name__ == "__main__":
    run_pipeline()