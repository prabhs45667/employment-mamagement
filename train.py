import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def parse_time(t):
    """
    Safely parse 'HH:MM' or 'HH:MM:SS' strings into total minutes from midnight.
    If the string is invalid or NaN, returns NaN.
    """
    if pd.isna(t):
        return np.nan
    
    # Convert to string in case it's numerical
    t_str = str(t).strip()
    
    # Split by ':'
    parts = t_str.split(':')
    
    # Handle different possible lengths
    if len(parts) == 1:
        # Possibly only an hour, e.g. '9'
        h = int(parts[0])
        m = 0
        s = 0
    elif len(parts) == 2:
        # HH:MM
        h, m = parts
        s = 0
    elif len(parts) >= 3:
        # HH:MM:SS (if there's more than 3 parts, ignore extras)
        h, m, s = parts[0], parts[1], parts[2]
    else:
        return np.nan  # Unexpected format
    
    try:
        h = int(h)
        m = int(m)
    except:
        return np.nan
    
    # Try parsing seconds
    try:
        s = int(s) if 's' in locals() else 0
    except:
        s = 0
    
    # Convert total time to minutes
    total_minutes = h * 60 + m
    return total_minutes

def main():
    # 1) Load the CSV dataset
    data = pd.read_csv('employee_attendance.csv')  # Replace with your actual filename
    
    # 2) Before any cleaning, examine class distribution
    print("========== BEFORE CLEANING ==========")
    print("Rows in dataset:", len(data))
    if 'Late_Arrival' in data.columns:
        print("Late_Arrival (raw):\n", data['Late_Arrival'].value_counts(dropna=False))
    if 'Early_Leave' in data.columns:
        print("Early_Leave (raw):\n", data['Early_Leave'].value_counts(dropna=False))
    
    # 3) Basic data cleaning & feature engineering
    #    Adjust this format if your CSV date is stored differently (DD-MM-YYYY or YYYY-MM-DD, etc.)
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
    
    # Convert date to day of week
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    
    # Parse times
    data['Time_In_Minutes'] = data['Time_In'].apply(parse_time)
    data['Time_Out_Minutes'] = data['Time_Out'].apply(parse_time)
    
    # Drop rows that have NaN in crucial columns
    print("\nDropping rows with NaN in crucial columns...")
    print("Rows before dropna:", len(data))
    data.dropna(subset=['DayOfWeek','Time_In_Minutes','Time_Out_Minutes','Late_Arrival','Early_Leave'], inplace=True)
    print("Rows after dropna:", len(data))
    
    # Calculate employee historical averages to fix data leakage!
    emp_averages = data.groupby('Email').agg({
        'Time_In_Minutes': 'mean',
        'Time_Out_Minutes': 'mean'
    }).reset_index()
    emp_averages.columns = ['Email', 'Emp_Avg_Time_In', 'Emp_Avg_Time_Out']
    
    data = data.merge(emp_averages, on='Email')
    
    # Convert "Yes"/"No" to numeric if needed
    le_late = LabelEncoder()
    le_early = LabelEncoder()
    
    data['Late_Arrival'] = le_late.fit_transform(data['Late_Arrival'])   # "Yes"->1, "No"->0
    data['Early_Leave'] = le_early.fit_transform(data['Early_Leave'])   # "Yes"->1, "No"->0
    
    # Confirm distribution after label encoding
    print("\nLate_Arrival distribution (encoded):\n", data['Late_Arrival'].value_counts())
    print("Early_Leave distribution (encoded):\n", data['Early_Leave'].value_counts())
    
    # Drop columns irrelevant to modeling
    data.drop(['Date','Time_In','Time_Out','Email','Employee_ID', 'Time_In_Minutes', 'Time_Out_Minutes'], axis=1, errors='ignore', inplace=True)
    
    # 4) Separate features and labels
    feature_cols = ['DayOfWeek', 'Emp_Avg_Time_In', 'Emp_Avg_Time_Out']
    X = data[feature_cols].values
    
    y_late = data['Late_Arrival'].values
    y_early = data['Early_Leave'].values
    
    # 5) Train separate models
    
    # ===== Late Arrival =====
    X_train_late, X_test_late, y_train_late, y_test_late = train_test_split(
        X, y_late, test_size=0.2, random_state=42
    )
    late_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    late_model.fit(X_train_late, y_train_late)
    late_accuracy = late_model.score(X_test_late, y_test_late)
    print("\n===== LATE ARRIVAL MODEL =====")
    print("Late_Arrival Model Accuracy: {:.2f}%".format(late_accuracy * 100))
    late_pred_test = late_model.predict(X_test_late)
    print("Confusion Matrix (Late_Arrival):\n", confusion_matrix(y_test_late, late_pred_test))
    print("Classification Report (Late_Arrival):\n", classification_report(y_test_late, late_pred_test))
    
    # ===== Early Leave =====
    X_train_early, X_test_early, y_train_early, y_test_early = train_test_split(
        X, y_early, test_size=0.2, random_state=42
    )
    early_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    early_model.fit(X_train_early, y_train_early)
    early_accuracy = early_model.score(X_test_early, y_test_early)
    print("\n===== EARLY LEAVE MODEL =====")
    print("Early_Leave Model Accuracy: {:.2f}%".format(early_accuracy * 100))
    early_pred_test = early_model.predict(X_test_early)
    print("Confusion Matrix (Early_Leave):\n", confusion_matrix(y_test_early, early_pred_test))
    print("Classification Report (Early_Leave):\n", classification_report(y_test_early, early_pred_test))
    
    # 6) Save the trained models
    joblib.dump(late_model, 'late_arrival_model.pkl')
    joblib.dump(early_model, 'early_leave_model.pkl')
    print("\nModels saved successfully: late_arrival_model.pkl, early_leave_model.pkl")
    
    # Save the employee averages so the prediction script/web app can use them
    emp_averages.to_json('employee_historical_averages.json', orient='records')
    print("Employee averages saved to employee_historical_averages.json")

if __name__ == '__main__':
    main()
