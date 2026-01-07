import pandas as pd
import os

# --- UPDATED PATHS (Pointing to the Data folder) ---
FILE_IVAN = os.path.join('Data', 'ivan_adjusted_data.csv')
FILE_MAX = os.path.join('Data', 'max.csv')

# --- FEATURES (Inputs for Random Forest) ---
FEATURES = [
    'batch_size',
    'buffer_size',
    'learning_rate',
    'beta',
    'epsilon',
    'lambd',
    'num_epoch',
    'hidden_units',
    'num_layers',
    'time_horizon',
    'max_steps'
]

# --- TARGETS (Outputs to Predict) ---
TARGETS = [
    'Time Elapsed/s',
    '3DBall.Environment.CumulativeReward.mean'
]

# --- DEFAULTS (To fill missing values) ---
DEFAULTS = {
    'hidden_units': 128,
    'num_layers': 2,
    'gamma': 0.99,
    'max_steps': 500000,
    'time_horizon': 64,
    'batch_size': 64,
    'buffer_size': 10240,
    'learning_rate': 3e-4,
    'beta': 0.001,
    'epsilon': 0.2,
    'lambd': 0.99,
    'num_epoch': 3
}

def generate_files():
    # 1. Load Data
    print(f"Looking for files in: {os.getcwd()}/Data/")
    
    try:
        df_ivan = pd.read_csv(FILE_IVAN)
        print(f"Loaded Ivan's data: {len(df_ivan)} rows")
    except FileNotFoundError:
        print(f" ERROR: Could not find {FILE_IVAN}")
        return

    try:
        df_max = pd.read_csv(FILE_MAX)
        print(f"Loaded Max's data: {len(df_max)} rows")
    except FileNotFoundError:
        print(f" Warning: Could not find {FILE_MAX} (Processing only Ivan's data)")
        df_max = pd.DataFrame()

    # 2. Merge
    full_df = pd.concat([df_ivan, df_max], ignore_index=True)
    
    # 3. Clean & Fill
    for col in FEATURES:
        # Create column if missing
        if col not in full_df.columns:
            full_df[col] = DEFAULTS.get(col, 0)
        # Fill NaNs
        full_df[col] = full_df[col].fillna(DEFAULTS.get(col, 0))
    
    # Drop rows where Targets are missing
    full_df.dropna(subset=TARGETS, inplace=True)

    # 4. Save Outputs (Saving these to Root is fine)
    X = full_df[FEATURES]
    y = full_df[TARGETS]

    X.to_csv('X_features.csv', index=False)
    y.to_csv('y_targets.csv', index=False)
    
    print("\n SUCCESS! Files Generated in root folder:")
    print(f"  -> X_features.csv ({len(X)} rows)")
    print(f"  -> y_targets.csv  ({len(y)} rows)")

if __name__ == "__main__":
    generate_files()