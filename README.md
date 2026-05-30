# Employee Management Analytics System

This system provides enhanced analytics capabilities for employee attendance data, integrating with machine learning models to analyze patterns and predict behaviors.

## Features

- **Monthly Overview**: Visualize late arrivals, early departures, and working hours by month
- **Attendance Patterns**: Analyze arrival and departure times by day of week
- **Working Hours Analysis**: Track working hours trends, cumulative monthly hours, and percentage loss hours
- **Deviation Analysis**: Identify significant deviations from baseline expectations with special highlighting for employees exceeding 15% deviation

## Components

- `train.py`: Trains machine learning models for predicting late arrivals and early departures
- `predict.py`: Uses trained models to predict employee attendance behaviors
- `employee_analytics_ui.py`: Interactive UI for detailed employee analytics

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training Models

Run the training script to build prediction models:

```
python train.py
```

This will create two model files: `late_arrival_model.pkl` and `early_leave_model.pkl`

### Making Predictions

Run the prediction script to identify employees likely to arrive late or leave early:

```
python predict.py
```

### Analytics Dashboard

Launch the interactive analytics dashboard:

```
python employee_analytics_ui.py
```

The dashboard provides:
- Individual employee profiles
- Monthly attendance averages
- Time pattern analysis
- Working hours calculations including cumulative monthly hours and percentage loss hours
- Deviation analysis with identification of employees exceeding 15% deviation from baseline
- Visual highlighting of employees with significant working hours deviation

## Data Format

The system expects a CSV file with the following columns:
- Employee_ID: Unique identifier for each employee
- Name: Employee name
- Date: Date in DD-MM-YYYY format
- Time_In: Arrival time in HH:MM format
- Time_Out: Departure time in HH:MM format
- Late_Arrival: "Yes" or "No" indicating if arrival was late
- Early_Leave: "Yes" or "No" indicating if departure was early