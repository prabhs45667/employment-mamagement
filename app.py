import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# Paths to data and models
DATA_PATH = 'employee_attendance.csv'
LATE_MODEL_PATH = 'late_arrival_model.pkl'
EARLY_MODEL_PATH = 'early_leave_model.pkl'

# Helper functions for time parsing and calculation
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
        try:
            h = int(parts[0])
            m = 0
        except:
            return np.nan
    elif len(parts) == 2:
        try:
            h, m = int(parts[0]), int(parts[1])
        except:
            return np.nan
    elif len(parts) >= 3:
        try:
            h, m = int(parts[0]), int(parts[1])
        except:
            return np.nan
    else:
        return np.nan
    
    return h * 60 + m

def minutes_to_time_str(minutes):
    """Convert minutes since midnight to 'HH:MM' format."""
    if pd.isna(minutes):
        return "--:--"
    hours = int(minutes) // 60
    mins = int(minutes) % 60
    return f"{hours:02d}:{mins:02d}"

def calculate_working_hours(time_in_minutes, time_out_minutes):
    """Calculate working hours from time_in and time_out in minutes."""
    if pd.isna(time_in_minutes) or pd.isna(time_out_minutes):
        return np.nan
    if time_out_minutes < time_in_minutes:
        return np.nan
    return (time_out_minutes - time_in_minutes) / 60

# Data and model loader
def get_data_and_models():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # Process the data
    df['Date_Parsed'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df['DayOfWeek'] = df['Date_Parsed'].dt.dayofweek
    df['Month'] = df['Date_Parsed'].dt.month
    df['Year'] = df['Date_Parsed'].dt.year
    df['Time_In_Minutes'] = df['Time_In'].apply(parse_time)
    df['Time_Out_Minutes'] = df['Time_Out'].apply(parse_time)
    df['Working_Hours'] = df.apply(
        lambda row: calculate_working_hours(row['Time_In_Minutes'], row['Time_Out_Minutes']), 
        axis=1
    )
    df['Team'] = (df['Employee_ID'] // 10).astype(str)
    
    late_model = None
    early_model = None
    
    try:
        late_model = joblib.load(LATE_MODEL_PATH)
        early_model = joblib.load(EARLY_MODEL_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        
    return df, late_model, early_model

# HTML Route
@app.route('/')
def index():
    return render_template('index.html')

# API Endpoints
@app.route('/api/employees', methods=['GET'])
def get_employees():
    try:
        df, _, _ = get_data_and_models()
        employees = df.groupby('Email').agg({
            'Employee_ID': 'first',
            'Name': 'first',
            'Team': 'first'
        }).reset_index()
        
        employees_list = employees.to_dict(orient='records')
        return jsonify(employees_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/employee/<email>', methods=['GET'])
def get_employee_data(email):
    try:
        df, late_model, early_model = get_data_and_models()
        emp_df = df[df['Email'] == email].copy()
        
        if emp_df.empty:
            return jsonify({'error': f"Employee with email {email} not found"}), 404
        
        emp_name = emp_df['Name'].iloc[0]
        emp_id = int(emp_df['Employee_ID'].iloc[0])
        emp_team = emp_df['Team'].iloc[0]
        
        # Summary statistics
        total_days = len(emp_df)
        avg_working_hours = float(emp_df['Working_Hours'].mean()) if not emp_df['Working_Hours'].dropna().empty else 0.0
        avg_late_rate = float((emp_df['Late_Arrival'] == 'Yes').mean() * 100)
        avg_early_rate = float((emp_df['Early_Leave'] == 'Yes').mean() * 100)
        cumulative_hours = float(emp_df['Working_Hours'].sum())
        expected_hours = total_days * 9.0
        hours_loss = expected_hours - cumulative_hours
        percentage_loss = (hours_loss / expected_hours * 100) if expected_hours > 0 else 0.0
        
        # Monthly overview
        monthly_df = emp_df.groupby(['Year', 'Month']).agg({
            'Late_Arrival': lambda x: float((x == 'Yes').mean() * 100),
            'Early_Leave': lambda x: float((x == 'Yes').mean() * 100),
            'Working_Hours': ['mean', 'sum'],
            'Date': 'count'
        }).reset_index()
        
        monthly_df.columns = ['Year', 'Month', 'Late_Arrival_Rate', 'Early_Leave_Rate', 'Avg_Working_Hours', 'Working_Hours_Sum', 'Days']
        monthly_df['Expected_Hours'] = monthly_df['Days'] * 9.0
        monthly_df['Hours_Loss'] = monthly_df['Expected_Hours'] - monthly_df['Working_Hours_Sum']
        monthly_df['Percentage_Loss'] = (monthly_df['Hours_Loss'] / monthly_df['Expected_Hours'] * 100)
        monthly_df['Month_Year'] = monthly_df.apply(lambda row: f"{int(row['Month'])}/{int(row['Year'])}", axis=1)
        
        # Day of week patterns
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_df = emp_df.groupby('DayOfWeek').agg({
            'Time_In_Minutes': 'mean',
            'Time_Out_Minutes': 'mean',
            'Working_Hours': 'mean'
        }).reindex(range(7)).reset_index()
        
        dow_df['Day_Name'] = dow_df['DayOfWeek'].apply(lambda d: day_names[int(d)])
        dow_df['Avg_Time_In'] = dow_df['Time_In_Minutes'].apply(lambda m: minutes_to_time_str(m) if pd.notna(m) else "--:--")
        dow_df['Avg_Time_Out'] = dow_df['Time_Out_Minutes'].apply(lambda m: minutes_to_time_str(m) if pd.notna(m) else "--:--")
        dow_df['Avg_Working_Hours'] = dow_df['Working_Hours'].fillna(0.0)
        
        # Time Series details (sorted by date)
        ts_df = emp_df.sort_values('Date_Parsed').copy()
        ts_df['Date_Str'] = ts_df['Date_Parsed'].dt.strftime('%d-%m-%Y')
        ts_data = ts_df[['Date_Str', 'Time_In', 'Time_Out', 'Working_Hours', 'Late_Arrival', 'Early_Leave']].to_dict(orient='records')
        
        # Rolling averages (7 days window)
        ts_df['Late_Numeric'] = ts_df['Late_Arrival'].apply(lambda x: 1 if x == 'Yes' else 0)
        ts_df['Early_Numeric'] = ts_df['Early_Leave'].apply(lambda x: 1 if x == 'Yes' else 0)
        ts_df['Rolling_Late'] = ts_df['Late_Numeric'].rolling(window=min(7, len(ts_df)), min_periods=1).mean() * 100
        ts_df['Rolling_Early'] = ts_df['Early_Numeric'].rolling(window=min(7, len(ts_df)), min_periods=1).mean() * 100
        
        rolling_data = ts_df[['Date_Str', 'Rolling_Late', 'Rolling_Early']].to_dict(orient='records')
        
        # Improvement Score
        first_half = ts_df.iloc[:len(ts_df)//2]
        second_half = ts_df.iloc[len(ts_df)//2:]
        if len(first_half) > 0 and len(second_half) > 0:
            late_improvement = float(first_half['Late_Numeric'].mean() - second_half['Late_Numeric'].mean())
            early_improvement = float(first_half['Early_Numeric'].mean() - second_half['Early_Numeric'].mean())
            improvement_score = (late_improvement + early_improvement) * 50.0
        else:
            late_improvement = 0.0
            early_improvement = 0.0
            improvement_score = 0.0
            
        improvement_metrics = {
            'late_improvement': late_improvement * 100,
            'early_improvement': early_improvement * 100,
            'improvement_score': improvement_score,
            'trend_status': "Employee is improving!" if improvement_score > 10 else ("Employee needs attention." if improvement_score < -10 else "No significant change.")
        }
        
        # Predictions (from model on historical records using averages)
        predicted_late_count = 0
        predicted_early_count = 0
        
        if late_model is not None and early_model is not None:
            emp_avg_in = emp_df['Time_In_Minutes'].mean()
            emp_avg_out = emp_df['Time_Out_Minutes'].mean()
            
            features = emp_df[['DayOfWeek']].copy()
            features['Emp_Avg_Time_In'] = emp_avg_in if pd.notna(emp_avg_in) else 540.0
            features['Emp_Avg_Time_Out'] = emp_avg_out if pd.notna(emp_avg_out) else 1020.0
            features.dropna(inplace=True)
            
            if not features.empty:
                late_preds = late_model.predict(features[['DayOfWeek', 'Emp_Avg_Time_In', 'Emp_Avg_Time_Out']].values)
                early_preds = early_model.predict(features[['DayOfWeek', 'Emp_Avg_Time_In', 'Emp_Avg_Time_Out']].values)
                predicted_late_count = int(np.sum(late_preds))
                predicted_early_count = int(np.sum(early_preds))
                
        # Deviation indicators
        emp_deviations = df.groupby('Name')['Working_Hours'].apply(
            lambda x: ((9 * len(x) - x.sum()) / (9 * len(x))) * 100 if len(x) > 0 else 0.0
        ).reset_index()
        emp_deviations.columns = ['Name', 'Deviation_Percent']
        high_deviation_threshold_met = bool(abs(emp_deviations[emp_deviations['Name'] == emp_name]['Deviation_Percent'].iloc[0]) > 15) if not emp_deviations[emp_deviations['Name'] == emp_name].empty else False
        
        response = {
            'info': {
                'email': email,
                'name': emp_name,
                'employee_id': emp_id,
                'team': emp_team,
                'high_deviation': high_deviation_threshold_met,
                'deviation_val': float(emp_deviations[emp_deviations['Name'] == emp_name]['Deviation_Percent'].iloc[0]) if not emp_deviations[emp_deviations['Name'] == emp_name].empty else 0.0
            },
            'summary': {
                'total_days': total_days,
                'avg_working_hours': avg_working_hours,
                'avg_late_rate': avg_late_rate,
                'avg_early_rate': avg_early_rate,
                'cumulative_hours': cumulative_hours,
                'expected_hours': expected_hours,
                'hours_loss': hours_loss,
                'percentage_loss': percentage_loss,
                'predicted_late_count': predicted_late_count,
                'predicted_early_count': predicted_early_count
            },
            'monthly': monthly_df.to_dict(orient='records'),
            'patterns': dow_df.to_dict(orient='records'),
            'timeseries': ts_data,
            'rolling': rolling_data,
            'improvement': improvement_metrics
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/team-comparison', methods=['GET'])
def get_team_comparison():
    try:
        df, _, _ = get_data_and_models()
        team_data = df.groupby('Team').agg({
            'Late_Arrival': lambda x: float((x == 'Yes').mean() * 100),
            'Early_Leave': lambda x: float((x == 'Yes').mean() * 100),
            'Working_Hours': 'mean',
            'Name': 'nunique',
            'Employee_ID': 'count'
        }).reset_index()
        
        team_data.columns = ['Team', 'Late_Rate', 'Early_Rate', 'Avg_Hours', 'Members', 'Records']
        
        teams_list = team_data.to_dict(orient='records')
        return jsonify(teams_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cumulative-dashboard', methods=['GET'])
def get_cumulative_dashboard():
    try:
        df, _, _ = get_data_and_models()
        
        total_employees = int(df['Name'].nunique())
        total_records = int(len(df))
        avg_late = float((df['Late_Arrival'] == 'Yes').mean() * 100)
        avg_early = float((df['Early_Leave'] == 'Yes').mean() * 100)
        avg_hours = float(df['Working_Hours'].mean())
        
        # Monthly overview trend
        monthly_trend = df.groupby(['Year', 'Month']).agg({
            'Late_Arrival': lambda x: float((x == 'Yes').mean() * 100),
            'Early_Leave': lambda x: float((x == 'Yes').mean() * 100),
            'Working_Hours': 'sum',
            'Date': 'count'
        }).reset_index()
        monthly_trend['Month_Year'] = monthly_trend.apply(lambda row: f"{int(row['Month'])}/{int(row['Year'])}", axis=1)
        monthly_trend['Expected_Hours'] = monthly_trend['Date'] * 9.0
        monthly_trend['Loss_Percent'] = ((monthly_trend['Expected_Hours'] - monthly_trend['Working_Hours']) / monthly_trend['Expected_Hours']) * 100
        
        # Employee Performance Scores
        emp_perf = df.groupby(['Name', 'Email']).agg({
            'Late_Arrival': lambda x: float((x == 'Yes').mean() * 100),
            'Early_Leave': lambda x: float((x == 'Yes').mean() * 100),
            'Working_Hours': 'mean'
        }).reset_index()
        emp_perf['Score'] = 100.0 - (emp_perf['Late_Arrival'] + emp_perf['Early_Leave']) / 2.0
        emp_perf = emp_perf.sort_values('Score', ascending=False)
        
        # Employee Deviations
        employee_deviations = df.groupby(['Name', 'Email'])['Working_Hours'].apply(
            lambda x: ((9 * len(x) - x.sum()) / (9 * len(x))) * 100 if len(x) > 0 else 0.0
        ).reset_index()
        employee_deviations.columns = ['Name', 'Email', 'Deviation_Percent']
        
        high_deviation_employees = employee_deviations[abs(employee_deviations['Deviation_Percent']) > 15].to_dict(orient='records')
        
        # Histogram data for deviation
        df['Hours_Deviation'] = 9.0 - df['Working_Hours']
        hist_counts, bin_edges = np.histogram(df['Hours_Deviation'].dropna(), bins=10)
        hist_data = {
            'counts': hist_counts.tolist(),
            'bins': [float(b) for b in bin_edges]
        }
        
        response = {
            'summary': {
                'total_employees': total_employees,
                'total_records': total_records,
                'avg_late_rate': avg_late,
                'avg_early_rate': avg_early,
                'avg_working_hours': avg_hours
            },
            'monthly_trend': monthly_trend.to_dict(orient='records'),
            'performance': emp_perf.to_dict(orient='records'),
            'high_deviations': high_deviation_employees,
            'deviation_histogram': hist_data
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-custom', methods=['POST'])
def predict_custom():
    try:
        _, late_model, early_model = get_data_and_models()
        if late_model is None or early_model is None:
            return jsonify({'error': "Machine learning models are not trained or loaded. Please run train.py first."}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': "Missing request data"}), 400
            
        day_of_week = int(data.get('day_of_week', 0)) # 0=Monday, 6=Sunday
        time_in_str = data.get('time_in', '09:00')
        time_out_str = data.get('time_out', '17:00')
        
        time_in_min = parse_time(time_in_str)
        time_out_min = parse_time(time_out_str)
        
        if pd.isna(time_in_min) or pd.isna(time_out_min):
            return jsonify({'error': "Invalid time formats. Use HH:MM."}), 400
            
        # Features: [DayOfWeek, Time_In_Minutes, Time_Out_Minutes]
        features = np.array([[day_of_week, time_in_min, time_out_min]])
        
        late_pred = int(late_model.predict(features)[0])
        early_pred = int(early_model.predict(features)[0])
        
        # Probabilities if supported
        try:
            late_prob = float(late_model.predict_proba(features)[0][1] * 100)
            early_prob = float(early_model.predict_proba(features)[0][1] * 100)
        except:
            late_prob = 100.0 if late_pred == 1 else 0.0
            early_prob = 100.0 if early_pred == 1 else 0.0
            
        return jsonify({
            'late_arrival': "Yes" if late_pred == 1 else "No",
            'late_probability': late_prob,
            'early_leave': "Yes" if early_pred == 1 else "No",
            'early_probability': early_prob,
            'working_hours': (time_out_min - time_in_min) / 60
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Try running on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
