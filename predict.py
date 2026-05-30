import pandas as pd
import numpy as np
import joblib
import json

def parse_time(t):
    """
    Safely parse 'HH:MM' or 'HH:MM:SS' strings into total minutes from midnight.
    Returns NaN if invalid.
    """
    if pd.isna(t):
        return np.nan
    
    t_str = str(t).strip()
    parts = t_str.split(':')
    
    if len(parts) == 1:
        h = int(parts[0])
        m = 0
    elif len(parts) == 2:
        h, m = parts
    elif len(parts) >= 3:
        h, m, _ = parts[0], parts[1], parts[2]
    else:
        return np.nan
    
    try:
        h = int(h)
        m = int(m)
    except:
        return np.nan
    
    return h * 60 + m

def main():
    # 1) Load your saved models
    late_model = joblib.load('late_arrival_model.pkl')
    early_model = joblib.load('early_leave_model.pkl')
    
    # 2) Read CSV again for predictions
    df = pd.read_csv('employee_attendance.csv')
    
    # 3) Convert date & times to match train.py
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    df['Time_In_Minutes'] = df['Time_In'].apply(parse_time)
    df['Time_Out_Minutes'] = df['Time_Out'].apply(parse_time)
    
    # Remove any pre-existing Late_Arrival / Early_Leave columns
    for col in ['Late_Arrival','Early_Leave']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # Drop rows with missing crucial data
    df.dropna(subset=['Email','DayOfWeek','Time_In_Minutes','Time_Out_Minutes'], inplace=True)
    
    # 4) Prepare input features
    X = df[['DayOfWeek','Time_In_Minutes','Time_Out_Minutes']].values
    
    # 5) Predict with both models
    late_preds = late_model.predict(X)
    early_preds = early_model.predict(X)
    
    # Convert numeric predictions to Yes/No
    late_str = ["Yes" if p == 1 else "No" for p in late_preds]
    early_str = ["Yes" if p == 1 else "No" for p in early_preds]
    
    # 6) Construct a results DataFrame
    results = pd.DataFrame({
        'Name': df['Name'],
        'Predicted_Late': late_str,
        'Predicted_EarlyLeave': early_str
    })
    
    # 7) Keep only rows with at least one "Yes"
    late_early_only = results[
        (results['Predicted_Late'] == "Yes") | (results['Predicted_EarlyLeave'] == "Yes")
    ]
    
    # 8) Group by Name to merge multiple records. If any record is "Yes" for that person, final is "Yes."
    def any_yes(series):
        return "Yes" if "Yes" in series.values else "No"
    
    merged = late_early_only.groupby('Name', as_index=False).agg({
        'Predicted_Late': any_yes,
        'Predicted_EarlyLeave': any_yes
    })
    
    # 9) Print each row in horizontal form, one below another
    print("\nEmployees flagged as late or early (one per line):")
    for _, row in merged.iterrows():
        print(f"{row['Name']} -> Late: {row['Predicted_Late']} | Early: {row['Predicted_EarlyLeave']}")
    
    # 10) Save merged results to JSON
    json_output = merged.to_json(orient='records')
    with open('late_early_results.json', 'w') as f:
        f.write(json_output)
    
    print(f"\nTotal unique employees flagged: {len(merged)}")
    print("Merged results have been saved to late_early_results.json.")

if __name__ == '__main__':
    main()
