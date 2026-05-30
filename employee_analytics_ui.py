import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Utility functions
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
    elif len(parts) == 2:
        # HH:MM
        h, m = parts
    else:
        # Unexpected format
        return np.nan
    
    try:
        h = int(h)
        m = int(m)
    except:
        return np.nan
    
    # Convert total time to minutes
    total_minutes = h * 60 + m
    return total_minutes

def minutes_to_time_str(minutes):
    """
    Convert minutes since midnight to 'HH:MM' format
    """
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

def calculate_working_hours(time_in_minutes, time_out_minutes):
    """
    Calculate working hours from time_in and time_out in minutes
    """
    if pd.isna(time_in_minutes) or pd.isna(time_out_minutes):
        return np.nan
    
    # Handle cases where time_out is earlier than time_in (should not happen in normal cases)
    if time_out_minutes < time_in_minutes:
        return np.nan
    
    return (time_out_minutes - time_in_minutes) / 60  # Convert to hours

class EmployeeAnalyticsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Employee Attendance Analytics")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize data
        self.df = None
        self.current_employee = None
        self.late_model = None
        self.early_model = None
        
        # Try to load models
        try:
            self.late_model = joblib.load('late_arrival_model.pkl')
            self.early_model = joblib.load('early_leave_model.pkl')
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
        
        # Create main frames
        self.create_widgets()
        
        # Load data if available
        if os.path.exists('employee_attendance.csv'):
            self.load_data('employee_attendance.csv')
    
    def create_widgets(self):
        # Create main frames
        self.sidebar_frame = tk.Frame(self.root, width=250, bg="#2c3e50")
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_frame.pack_propagate(False)
        
        self.content_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Sidebar widgets
        self.title_label = tk.Label(self.sidebar_frame, text="Employee Analytics", 
                                   font=("Arial", 16, "bold"), bg="#2c3e50", fg="white")
        self.title_label.pack(pady=20)
        
        # Load data button
        self.load_button = tk.Button(self.sidebar_frame, text="Load Data", 
                                    command=self.browse_file, bg="#3498db", fg="white",
                                    font=("Arial", 12), width=20)
        self.load_button.pack(pady=10)
        
        # Employee selection
        self.employee_label = tk.Label(self.sidebar_frame, text="Select Employee:", 
                                     font=("Arial", 12), bg="#2c3e50", fg="white")
        self.employee_label.pack(pady=(20, 5))
        
        self.employee_var = tk.StringVar()
        self.employee_dropdown = ttk.Combobox(self.sidebar_frame, textvariable=self.employee_var, 
                                            state="readonly", font=("Arial", 12), width=18)
        self.employee_dropdown.pack(pady=5)
        self.employee_dropdown.bind("<<ComboboxSelected>>", self.on_employee_selected)
        
        # Analysis type selection
        self.analysis_label = tk.Label(self.sidebar_frame, text="Analysis Type:", 
                                     font=("Arial", 12), bg="#2c3e50", fg="white")
        self.analysis_label.pack(pady=(20, 5))
        
        self.analysis_types = ["Monthly Overview", "Attendance Patterns", 
                              "Working Hours Analysis", "Deviation Analysis",
                              "Improvement Trends", "Team Comparisons", "Cumulative Dashboard"]
        self.analysis_var = tk.StringVar(value=self.analysis_types[0])
        
        for analysis in self.analysis_types:
            rb = tk.Radiobutton(self.sidebar_frame, text=analysis, variable=self.analysis_var, 
                               value=analysis, command=self.update_analysis,
                               bg="#2c3e50", fg="white", selectcolor="#34495e",
                               activebackground="#2c3e50", activeforeground="white",
                               font=("Arial", 12))
            rb.pack(anchor=tk.W, padx=20, pady=5)
        
        # Content area - initially empty
        self.content_title = tk.Label(self.content_frame, text="Please load data and select an employee", 
                                    font=("Arial", 16, "bold"), bg="#f0f0f0")
        self.content_title.pack(pady=20)
        
        # Frame for plots
        self.plot_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.load_data(file_path)
    
    def load_data(self, file_path):
        try:
            self.df = pd.read_csv(file_path)
            
            # Process the data
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y', errors='coerce')
            self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
            self.df['Month'] = self.df['Date'].dt.month
            self.df['Year'] = self.df['Date'].dt.year
            
            self.df['Time_In_Minutes'] = self.df['Time_In'].apply(parse_time)
            self.df['Time_Out_Minutes'] = self.df['Time_Out'].apply(parse_time)
            
            # Calculate working hours
            self.df['Working_Hours'] = self.df.apply(
                lambda row: calculate_working_hours(row['Time_In_Minutes'], row['Time_Out_Minutes']), 
                axis=1
            )
            
            # Add Team field
            self.df['Team'] = (self.df['Employee_ID'] // 10).astype(str)
            
            # Update employee dropdown
            employees = sorted(self.df['Email'].unique())
            self.employee_dropdown['values'] = employees
            
            if len(employees) > 0:
                self.employee_var.set(employees[0])
                self.on_employee_selected(None)
            
            messagebox.showinfo("Success", f"Data loaded successfully with {len(self.df)} records")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def on_employee_selected(self, event):
        self.current_employee = self.employee_var.get()
        self.update_analysis()
    
    def update_analysis(self):
        if self.df is None:
            return
        
        analysis_type = self.analysis_var.get()
        
        # For employee-specific analyses, check if employee is selected
        if analysis_type in ["Monthly Overview", "Attendance Patterns", "Working Hours Analysis", "Deviation Analysis", "Improvement Trends"]:
            if self.current_employee is None:
                return
            # Filter data for the selected employee
            employee_data = self.df[self.df['Email'] == self.current_employee].copy()
            if employee_data.empty:
                messagebox.showinfo("No Data", f"No data available for {self.current_employee}")
                return
            # Update content title
            employee_name = employee_data['Name'].iloc[0] if not employee_data.empty else self.current_employee
            self.content_title.config(text=f"{analysis_type} for {employee_name}")
        else:
            # For non-employee specific, use all data
            self.content_title.config(text=f"{analysis_type}")
        
        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Call the appropriate analysis function
        if analysis_type == "Monthly Overview":
            self.show_monthly_overview(employee_data)
        elif analysis_type == "Attendance Patterns":
            self.show_attendance_patterns(employee_data)
        elif analysis_type == "Working Hours Analysis":
            self.show_working_hours_analysis(employee_data)
        elif analysis_type == "Deviation Analysis":
            self.show_deviation_analysis(employee_data)
        elif analysis_type == "Improvement Trends":
            self.show_improvement_trends(employee_data)
        elif analysis_type == "Team Comparisons":
            self.show_team_comparisons()
        elif analysis_type == "Cumulative Dashboard":
            self.show_cumulative_dashboard()
    
    def show_monthly_overview(self, employee_data):
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(10, 8))
        fig.subplots_adjust(hspace=0.4)
        
        # Group by month and calculate metrics
        monthly_data = employee_data.groupby(['Year', 'Month']).agg({
            'Late_Arrival': lambda x: (x == 'Yes').mean() * 100,
            'Early_Leave': lambda x: (x == 'Yes').mean() * 100,
            'Working_Hours': 'mean'
        }).reset_index()
        
        # Format month-year for display
        monthly_data['Month-Year'] = monthly_data.apply(
            lambda row: f"{row['Month']}-{row['Year']}", axis=1
        )
        
        # Plot 1: Late Arrivals by Month
        ax1 = fig.add_subplot(3, 1, 1)
        sns.barplot(x='Month-Year', y='Late_Arrival', data=monthly_data, ax=ax1, color='#3498db')
        ax1.set_title('Monthly Late Arrival Percentage')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Plot 2: Early Leaves by Month
        ax2 = fig.add_subplot(3, 1, 2)
        sns.barplot(x='Month-Year', y='Early_Leave', data=monthly_data, ax=ax2, color='#e74c3c')
        ax2.set_title('Monthly Early Leave Percentage')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: Average Working Hours by Month
        ax3 = fig.add_subplot(3, 1, 3)
        sns.barplot(x='Month-Year', y='Working_Hours', data=monthly_data, ax=ax3, color='#2ecc71')
        ax3.set_title('Average Working Hours by Month')
        ax3.set_ylabel('Hours')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        
        # Add the figure to the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add summary statistics
        stats_frame = tk.Frame(self.plot_frame, bg="#f0f0f0")
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Calculate overall statistics
        avg_late = employee_data['Late_Arrival'].apply(lambda x: 1 if x == 'Yes' else 0).mean() * 100
        avg_early = employee_data['Early_Leave'].apply(lambda x: 1 if x == 'Yes' else 0).mean() * 100
        avg_hours = employee_data['Working_Hours'].mean()
        
        stats_text = f"Overall Statistics:\n"
        stats_text += f"Average Late Arrival Rate: {avg_late:.2f}%\n"
        stats_text += f"Average Early Leave Rate: {avg_early:.2f}%\n"
        stats_text += f"Average Working Hours: {avg_hours:.2f} hours per day"
        
        stats_label = tk.Label(stats_frame, text=stats_text, font=("Arial", 12), 
                              bg="#f0f0f0", justify=tk.LEFT)
        stats_label.pack(padx=20, pady=10, anchor=tk.W)
    
    def show_attendance_patterns(self, employee_data):
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(10, 8))
        fig.subplots_adjust(hspace=0.4)
        
        # Plot 1: Arrival Time by Day of Week
        ax1 = fig.add_subplot(2, 1, 1)
        sns.boxplot(x='DayOfWeek', y='Time_In_Minutes', data=employee_data, ax=ax1, palette='viridis')
        ax1.set_title('Arrival Time by Day of Week')
        ax1.set_xlabel('Day of Week (0=Monday, 6=Sunday)')
        ax1.set_ylabel('Time (minutes from midnight)')
        
        # Add a secondary y-axis with hour labels
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylim(ax1.get_ylim())
        hour_ticks = np.arange(8*60, 11*60, 30)  # From 8:00 to 11:00 in 30-min intervals
        ax1_twin.set_yticks(hour_ticks)
        ax1_twin.set_yticklabels([minutes_to_time_str(m) for m in hour_ticks])
        ax1_twin.set_ylabel('Time')
        
        # Plot 2: Departure Time by Day of Week
        ax2 = fig.add_subplot(2, 1, 2)
        sns.boxplot(x='DayOfWeek', y='Time_Out_Minutes', data=employee_data, ax=ax2, palette='viridis')
        ax2.set_title('Departure Time by Day of Week')
        ax2.set_xlabel('Day of Week (0=Monday, 6=Sunday)')
        ax2.set_ylabel('Time (minutes from midnight)')
        
        # Add a secondary y-axis with hour labels
        ax2_twin = ax2.twinx()
        ax2_twin.set_ylim(ax2.get_ylim())
        hour_ticks = np.arange(16*60, 19*60, 30)  # From 16:00 to 19:00 in 30-min intervals
        ax2_twin.set_yticks(hour_ticks)
        ax2_twin.set_yticklabels([minutes_to_time_str(m) for m in hour_ticks])
        ax2_twin.set_ylabel('Time')
        
        # Add the figure to the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add pattern analysis
        pattern_frame = tk.Frame(self.plot_frame, bg="#f0f0f0")
        pattern_frame.pack(fill=tk.X, pady=10)
        
        # Calculate morning/evening patterns
        morning_avg = employee_data.groupby('DayOfWeek')['Time_In_Minutes'].mean()
        evening_avg = employee_data.groupby('DayOfWeek')['Time_Out_Minutes'].mean()
        
        # Find the day with earliest arrival and latest departure
        earliest_day = morning_avg.idxmin()
        latest_day = evening_avg.idxmax()
        
        # Calculate time zone deviation (consistency in arrival/departure times)
        time_in_std = employee_data['Time_In_Minutes'].std() / 60  # Convert to hours
        time_out_std = employee_data['Time_Out_Minutes'].std() / 60  # Convert to hours
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        pattern_text = f"Attendance Pattern Analysis:\n"
        pattern_text += f"Most punctual day: {day_names[earliest_day]} (avg. arrival: {minutes_to_time_str(int(morning_avg[earliest_day]))})\n"
        pattern_text += f"Latest departure day: {day_names[latest_day]} (avg. departure: {minutes_to_time_str(int(evening_avg[latest_day]))})\n"
        pattern_text += f"Time consistency (lower is better):\n"
        pattern_text += f"  - Arrival time variation: {time_in_std:.2f} hours\n"
        pattern_text += f"  - Departure time variation: {time_out_std:.2f} hours"
        
        pattern_label = tk.Label(pattern_frame, text=pattern_text, font=("Arial", 12), 
                                bg="#f0f0f0", justify=tk.LEFT)
        pattern_label.pack(padx=20, pady=10, anchor=tk.W)
    
    def show_working_hours_analysis(self, employee_data):
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(hspace=0.5)
        
        # Plot 1: Working Hours by Day of Week
        ax1 = fig.add_subplot(3, 1, 1)
        sns.boxplot(x='DayOfWeek', y='Working_Hours', data=employee_data, ax=ax1, palette='viridis')
        ax1.set_title('Working Hours by Day of Week')
        ax1.set_xlabel('Day of Week (0=Monday, 6=Sunday)')
        ax1.set_ylabel('Hours')
        ax1.axhline(y=9, color='r', linestyle='--', alpha=0.7)  # Standard 9-hour workday
        
        # Plot 2: Working Hours Trend Over Time
        ax2 = fig.add_subplot(3, 1, 2)
        sns.lineplot(x='Date', y='Working_Hours', data=employee_data, ax=ax2, marker='o')
        ax2.set_title('Working Hours Trend')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Hours')
        ax2.axhline(y=9, color='r', linestyle='--', alpha=0.7)  # Standard 9-hour workday
        
        # Plot 3: Cumulative Monthly Working Hours
        ax3 = fig.add_subplot(3, 1, 3)
        
        # Group by month and calculate cumulative hours
        monthly_data = employee_data.groupby(['Year', 'Month'])['Working_Hours'].sum().reset_index()
        monthly_data['Month-Year'] = monthly_data.apply(
            lambda row: f"{row['Month']}-{row['Year']}", axis=1
        )
        
        # Calculate percentage loss hours (assuming 9-hour workday)
        workdays_per_month = employee_data.groupby(['Year', 'Month'])['Date'].count().reset_index()
        workdays_per_month.columns = ['Year', 'Month', 'Workdays']
        monthly_data = monthly_data.merge(workdays_per_month, on=['Year', 'Month'])
        monthly_data['Expected_Hours'] = monthly_data['Workdays'] * 9
        monthly_data['Hours_Loss'] = monthly_data['Expected_Hours'] - monthly_data['Working_Hours']
        monthly_data['Percentage_Loss'] = (monthly_data['Hours_Loss'] / monthly_data['Expected_Hours']) * 100
        
        # Plot cumulative hours
        sns.barplot(x='Month-Year', y='Working_Hours', data=monthly_data, ax=ax3, color='#2ecc71')
        ax3.set_title('Cumulative Monthly Working Hours')
        ax3.set_ylabel('Total Hours')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        
        # Add a secondary y-axis for percentage loss
        ax3_twin = ax3.twinx()
        sns.lineplot(x=range(len(monthly_data)), y='Percentage_Loss', data=monthly_data, 
                    ax=ax3_twin, color='#e74c3c', marker='o')
        ax3_twin.set_ylabel('Percentage Loss (%)', color='#e74c3c')
        ax3_twin.tick_params(axis='y', colors='#e74c3c')
        ax3_twin.grid(False)
        
        # Format the date axis
        fig.autofmt_xdate()
        
        # Add the figure to the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add working hours statistics
        stats_frame = tk.Frame(self.plot_frame, bg="#f0f0f0")
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Calculate statistics
        avg_hours = employee_data['Working_Hours'].mean()
        std_hours = employee_data['Working_Hours'].std()
        max_hours = employee_data['Working_Hours'].max()
        min_hours = employee_data['Working_Hours'].min()
        
        # Calculate cumulative working hours
        total_hours = employee_data['Working_Hours'].sum()
        
        # Calculate percentage of days with less than 8 hours
        less_than_8 = (employee_data['Working_Hours'] < 8).mean() * 100
        
        # Calculate overall percentage loss
        total_expected_hours = len(employee_data) * 9
        total_loss_hours = total_expected_hours - total_hours
        percentage_loss = (total_loss_hours / total_expected_hours) * 100 if total_expected_hours > 0 else 0
        
        stats_text = f"Working Hours Analysis:\n"
        stats_text += f"Average working hours: {avg_hours:.2f} hours per day\n"
        stats_text += f"Variation in working hours: {std_hours:.2f} hours (standard deviation)\n"
        stats_text += f"Maximum hours worked in a day: {max_hours:.2f} hours\n"
        stats_text += f"Minimum hours worked in a day: {min_hours:.2f} hours\n"
        stats_text += f"Cumulative working hours: {total_hours:.2f} hours\n"
        stats_text += f"Total expected hours: {total_expected_hours:.2f} hours\n"
        stats_text += f"Total hours loss: {total_loss_hours:.2f} hours\n"
        stats_text += f"Overall percentage loss: {percentage_loss:.2f}%\n"
        stats_text += f"Percentage of days with less than 9 hours: {(employee_data['Working_Hours'] < 9).mean() * 100:.2f}%"
        
        stats_label = tk.Label(stats_frame, text=stats_text, font=("Arial", 12), 
                              bg="#f0f0f0", justify=tk.LEFT)
        stats_label.pack(padx=20, pady=10, anchor=tk.W)
    
    def show_deviation_analysis(self, employee_data):
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Calculate baseline metrics (expected values)
        baseline_arrival = 9 * 60  # 9:00 AM in minutes
        baseline_departure = 18 * 60  # 6:00 PM in minutes
        baseline_hours = 9  # 9 hours workday
        
        # Calculate deviations
        employee_data['Arrival_Deviation_Minutes'] = employee_data['Time_In_Minutes'] - baseline_arrival
        employee_data['Departure_Deviation_Minutes'] = baseline_departure - employee_data['Time_Out_Minutes']
        employee_data['Hours_Deviation'] = baseline_hours - employee_data['Working_Hours']
        employee_data['Hours_Deviation_Percent'] = (employee_data['Hours_Deviation'] / baseline_hours) * 100
        
        # Plot 1: Arrival Time Deviation
        ax1 = fig.add_subplot(3, 2, 1)
        sns.histplot(employee_data['Arrival_Deviation_Minutes'], ax=ax1, kde=True, color='#3498db')
        ax1.set_title('Arrival Time Deviation from 9:00 AM')
        ax1.set_xlabel('Minutes (negative = early, positive = late)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # Plot 2: Departure Time Deviation
        ax2 = fig.add_subplot(3, 2, 3)
        sns.histplot(employee_data['Departure_Deviation_Minutes'], ax=ax2, kde=True, color='#e74c3c')
        ax2.set_title('Departure Time Deviation from 6:00 PM')
        ax2.set_xlabel('Minutes (negative = late departure, positive = early departure)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # Plot 3: Working Hours Deviation
        ax3 = fig.add_subplot(3, 2, 5)
        sns.histplot(employee_data['Hours_Deviation'], ax=ax3, kde=True, color='#2ecc71')
        ax3.set_title('Working Hours Deviation from 9-hour Workday')
        ax3.set_xlabel('Hours (negative = overtime, positive = undertime)')
        ax3.set_ylabel('Frequency')
        ax3.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # Plot 4: Percentage Deviation Over Time
        ax4 = fig.add_subplot(3, 2, (2, 4))
        sns.lineplot(x='Date', y='Hours_Deviation_Percent', data=employee_data, ax=ax4, marker='o')
        ax4.set_title('Percentage Deviation Over Time')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Deviation (%)')
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax4.axhline(y=15, color='r', linestyle='-', alpha=0.5)  # 15% threshold line
        ax4.axhline(y=-15, color='r', linestyle='-', alpha=0.5)  # -15% threshold line
        fig.autofmt_xdate()
        
        # Add the figure to the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add deviation statistics
        stats_frame = tk.Frame(self.plot_frame, bg="#f0f0f0")
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Calculate statistics
        avg_arrival_dev = employee_data['Arrival_Deviation_Minutes'].mean()
        avg_departure_dev = employee_data['Departure_Deviation_Minutes'].mean()
        avg_hours_dev = employee_data['Hours_Deviation'].mean()
        
        # Calculate percentage of significant deviations (>15% from baseline)
        arrival_sig_dev = (abs(employee_data['Arrival_Deviation_Minutes']) > 0.15 * 60).mean() * 100
        departure_sig_dev = (abs(employee_data['Departure_Deviation_Minutes']) > 0.15 * 60).mean() * 100
        hours_sig_dev = (abs(employee_data['Hours_Deviation_Percent']) > 15).mean() * 100
        
        # Check if this employee has >15% deviation
        has_significant_deviation = (abs(employee_data['Hours_Deviation_Percent']) > 15).any()
        
        stats_text = f"Deviation Analysis:\n"
        stats_text += f"Average arrival deviation: {avg_arrival_dev:.2f} minutes from 9:00 AM\n"
        stats_text += f"Average departure deviation: {avg_departure_dev:.2f} minutes from 5:00 PM\n"
        stats_text += f"Average working hours deviation: {avg_hours_dev:.2f} hours from 9-hour workday\n\n"
        
        stats_text += f"Significant Deviations (>15% from baseline):\n"
        stats_text += f"Arrival time: {arrival_sig_dev:.2f}% of days\n"
        stats_text += f"Departure time: {departure_sig_dev:.2f}% of days\n"
        stats_text += f"Working hours: {hours_sig_dev:.2f}% of days"
        
        stats_label = tk.Label(stats_frame, text=stats_text, font=("Arial", 12), 
                              bg="#f0f0f0", justify=tk.LEFT)
        stats_label.pack(padx=20, pady=10, anchor=tk.W)
        
        # Add a section for employees with significant deviations
        if self.df is not None:
            # Create a frame for the high deviation employees list
            high_dev_frame = tk.Frame(self.plot_frame, bg="#f0f0f0")
            high_dev_frame.pack(fill=tk.X, pady=10)
            
            # Calculate average deviation percentage for each employee
            employee_deviations = self.df.groupby('Name')['Working_Hours'].apply(
                lambda x: ((9 * len(x) - x.sum()) / (9 * len(x))) * 100
            ).reset_index()
            employee_deviations.columns = ['Name', 'Deviation_Percent']
            
            # Filter employees with >15% deviation
            high_deviation_employees = employee_deviations[abs(employee_deviations['Deviation_Percent']) > 15]
            
            # Create a title for the section
            high_dev_title = tk.Label(high_dev_frame, text="Employees with >15% Working Hours Deviation:", 
                                    font=("Arial", 12, "bold"), bg="#f0f0f0")
            high_dev_title.pack(anchor=tk.W, padx=20, pady=(10, 5))
            
            # Create a frame for the list
            list_frame = tk.Frame(high_dev_frame, bg="#f0f0f0")
            list_frame.pack(fill=tk.X, padx=20, pady=5)
            
            # Add column headers
            tk.Label(list_frame, text="Employee Name", font=("Arial", 11, "bold"), 
                    bg="#f0f0f0", width=30, anchor="w").grid(row=0, column=0, sticky="w")
            tk.Label(list_frame, text="Deviation %", font=("Arial", 11, "bold"), 
                    bg="#f0f0f0", width=15, anchor="w").grid(row=0, column=1, sticky="w")
            
            # Add a separator
            separator = ttk.Separator(list_frame, orient='horizontal')
            separator.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
            
            # Add each employee to the list
            if len(high_deviation_employees) > 0:
                for i, (_, row) in enumerate(high_deviation_employees.iterrows()):
                    name_label = tk.Label(list_frame, text=row['Name'], font=("Arial", 11), 
                                        bg="#f0f0f0", width=30, anchor="w")
                    name_label.grid(row=i+2, column=0, sticky="w")
                    
                    # Highlight the current employee
                    if row['Name'] == self.current_employee:
                        name_label.config(font=("Arial", 11, "bold"), fg="#e74c3c")
                    
                    # Format the deviation percentage
                    dev_text = f"{row['Deviation_Percent']:.2f}%"
                    dev_label = tk.Label(list_frame, text=dev_text, font=("Arial", 11), 
                                        bg="#f0f0f0", width=15, anchor="w")
                    dev_label.grid(row=i+2, column=1, sticky="w")
                    
                    # Highlight positive/negative deviations
                    if row['Deviation_Percent'] > 0:
                        dev_label.config(fg="#e74c3c")  # Red for loss
                    else:
                        dev_label.config(fg="#2ecc71")  # Green for gain
            else:
                tk.Label(list_frame, text="No employees with significant deviation", 
                        font=("Arial", 11, "italic"), bg="#f0f0f0", 
                        width=45).grid(row=2, column=0, columnspan=2, sticky="w")
            
            # If current employee has significant deviation, highlight it
            if has_significant_deviation:
                highlight_frame = tk.Frame(high_dev_frame, bg="#ffcccc", padx=10, pady=10)
                highlight_frame.pack(fill=tk.X, padx=20, pady=10)
                
                highlight_text = f"⚠️ {self.current_employee} has significant working hours deviation! ⚠️"
                tk.Label(highlight_frame, text=highlight_text, font=("Arial", 12, "bold"), 
                        bg="#ffcccc", fg="#e74c3c").pack()

    def show_improvement_trends(self, employee_data):
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(12, 8))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        # Calculate improvement metrics
        employee_data = employee_data.sort_values('Date')
        employee_data['Late_Arrival_Num'] = employee_data['Late_Arrival'].apply(lambda x: 1 if x == 'Yes' else 0)
        employee_data['Early_Leave_Num'] = employee_data['Early_Leave'].apply(lambda x: 1 if x == 'Yes' else 0)

        # Rolling average for trends
        window = min(7, len(employee_data))
        employee_data['Late_Rolling'] = employee_data['Late_Arrival_Num'].rolling(window=window).mean()
        employee_data['Early_Rolling'] = employee_data['Early_Leave_Num'].rolling(window=window).mean()

        # Plot 1: Late Arrival Trend
        ax1 = fig.add_subplot(2, 2, 1)
        sns.lineplot(x='Date', y='Late_Rolling', data=employee_data, ax=ax1, marker='o', color='#3498db')
        ax1.set_title('Late Arrival Trend (7-day rolling average)')
        ax1.set_ylabel('Late Rate')
        ax1.set_xlabel('Date')
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)  # 50% threshold
        fig.autofmt_xdate()

        # Add tooltips (annotations for key points)
        for i, row in employee_data.iterrows():
            if row['Late_Arrival_Num'] == 1:
                ax1.annotate(f"{row['Date'].strftime('%d-%m')} Late", (row['Date'], row['Late_Rolling']),
                             xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')

        # Plot 2: Early Leave Trend
        ax2 = fig.add_subplot(2, 2, 2)
        sns.lineplot(x='Date', y='Early_Rolling', data=employee_data, ax=ax2, marker='o', color='#e74c3c')
        ax2.set_title('Early Leave Trend (7-day rolling average)')
        ax2.set_ylabel('Early Rate')
        ax2.set_xlabel('Date')
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        fig.autofmt_xdate()

        for i, row in employee_data.iterrows():
            if row['Early_Leave_Num'] == 1:
                ax2.annotate(f"{row['Date'].strftime('%d-%m')} Early", (row['Date'], row['Early_Rolling']),
                             xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')

        # Plot 3: Improvement Score
        ax3 = fig.add_subplot(2, 2, 3)
        # Calculate improvement score: decrease in late/early over time
        first_half = employee_data.iloc[:len(employee_data)//2]
        second_half = employee_data.iloc[len(employee_data)//2:]
        late_improvement = first_half['Late_Arrival_Num'].mean() - second_half['Late_Arrival_Num'].mean()
        early_improvement = first_half['Early_Leave_Num'].mean() - second_half['Early_Leave_Num'].mean()
        improvement_score = (late_improvement + early_improvement) * 50  # Scale to 0-100

        bars = ax3.bar(['Late Improvement', 'Early Improvement', 'Overall Score'],
                       [late_improvement * 100, early_improvement * 100, improvement_score], color=['#3498db', '#e74c3c', '#2ecc71'])
        ax3.set_title('Improvement Metrics')
        ax3.set_ylabel('Improvement %')
        ax3.set_ylim(-100, 100)

        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

        # Plot 4: Specific Late Times
        ax4 = fig.add_subplot(2, 2, 4)
        late_days = employee_data[employee_data['Late_Arrival'] == 'Yes'].copy()
        if not late_days.empty:
            late_days['Time_In_Str'] = late_days['Time_In_Minutes'].apply(minutes_to_time_str)
            sns.scatterplot(x='Date', y='Time_In_Minutes', data=late_days, ax=ax4, color='#ff6b6b', s=50)
            ax4.set_title('Specific Late Arrival Times')
            ax4.set_ylabel('Arrival Time (minutes)')
            ax4.set_xlabel('Date')
            fig.autofmt_xdate()

            # Add time labels
            for i, row in late_days.iterrows():
                ax4.annotate(row['Time_In_Str'], (row['Date'], row['Time_In_Minutes']),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Add the figure to the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add improvement statistics
        stats_frame = tk.Frame(self.plot_frame, bg="#f0f0f0")
        stats_frame.pack(fill=tk.X, pady=10)

        # Calculate stats
        total_late = employee_data['Late_Arrival_Num'].sum()
        total_early = employee_data['Early_Leave_Num'].sum()
        recent_late = second_half['Late_Arrival_Num'].sum()
        recent_early = second_half['Early_Leave_Num'].sum()

        improvement_text = f"Improvement Trends:\n"
        improvement_text += f"Total Late Days: {total_late}\n"
        improvement_text += f"Total Early Days: {total_early}\n"
        improvement_text += f"Recent Late Days (2nd half): {recent_late}\n"
        improvement_text += f"Recent Early Days (2nd half): {recent_early}\n"
        improvement_text += f"Late Improvement: {late_improvement*100:.1f}%\n"
        improvement_text += f"Early Improvement: {early_improvement*100:.1f}%\n"
        improvement_text += f"Overall Improvement Score: {improvement_score:.1f}%\n"

        if improvement_score > 10:
            improvement_text += "Employee is improving!"
        elif improvement_score < -10:
            improvement_text += "Employee needs attention."
        else:
            improvement_text += "No significant change."

        stats_label = tk.Label(stats_frame, text=improvement_text, font=("Arial", 12), 
                              bg="#f0f0f0", justify=tk.LEFT)
        stats_label.pack(padx=20, pady=10, anchor=tk.W)

    def show_team_comparisons(self):
        if self.df is None:
            return

        # Simulate teams by grouping Employee_ID
        self.df['Team'] = (self.df['Employee_ID'] // 10).astype(str)  # Group into teams of 10

        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(12, 8))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        # Group by team
        team_data = self.df.groupby('Team').agg({
            'Late_Arrival': lambda x: (x == 'Yes').mean() * 100,
            'Early_Leave': lambda x: (x == 'Yes').mean() * 100,
            'Working_Hours': 'mean',
            'Name': 'count'
        }).reset_index()
        team_data.columns = ['Team', 'Late_Rate', 'Early_Rate', 'Avg_Hours', 'Members']

        # Plot 1: Late Rate by Team
        ax1 = fig.add_subplot(2, 2, 1)
        sns.barplot(x='Team', y='Late_Rate', data=team_data, ax=ax1, color='#3498db')
        ax1.set_title('Late Arrival Rate by Team')
        ax1.set_ylabel('Late Rate (%)')

        for i, row in team_data.iterrows():
            ax1.annotate(f"{row['Late_Rate']:.1f}%", (i, row['Late_Rate']),
                         xytext=(0, 3), textcoords='offset points', ha='center')

        # Plot 2: Early Rate by Team
        ax2 = fig.add_subplot(2, 2, 2)
        sns.barplot(x='Team', y='Early_Rate', data=team_data, ax=ax2, color='#e74c3c')
        ax2.set_title('Early Leave Rate by Team')
        ax2.set_ylabel('Early Rate (%)')

        for i, row in team_data.iterrows():
            ax2.annotate(f"{row['Early_Rate']:.1f}%", (i, row['Early_Rate']),
                         xytext=(0, 3), textcoords='offset points', ha='center')

        # Plot 3: Avg Hours by Team
        ax3 = fig.add_subplot(2, 2, 3)
        sns.barplot(x='Team', y='Avg_Hours', data=team_data, ax=ax3, color='#2ecc71')
        ax3.set_title('Average Working Hours by Team')
        ax3.set_ylabel('Hours')

        for i, row in team_data.iterrows():
            ax3.annotate(f"{row['Avg_Hours']:.1f}h", (i, row['Avg_Hours']),
                         xytext=(0, 3), textcoords='offset points', ha='center')

        # Plot 4: Team Members
        ax4 = fig.add_subplot(2, 2, 4)
        sns.barplot(x='Team', y='Members', data=team_data, ax=ax4, color='#9b59b6')
        ax4.set_title('Team Size')
        ax4.set_ylabel('Members')

        for i, row in team_data.iterrows():
            ax4.annotate(f"{row['Members']}", (i, row['Members']),
                         xytext=(0, 3), textcoords='offset points', ha='center')

        # Add the figure to the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add team statistics
        stats_frame = tk.Frame(self.plot_frame, bg="#f0f0f0")
        stats_frame.pack(fill=tk.X, pady=10)

        # Find best and worst teams
        best_team_late = team_data.loc[team_data['Late_Rate'].idxmin()]['Team']
        worst_team_late = team_data.loc[team_data['Late_Rate'].idxmax()]['Team']
        best_team_hours = team_data.loc[team_data['Avg_Hours'].idxmax()]['Team']

        stats_text = f"Team Comparisons:\n"
        stats_text += f"Best Late Rate Team: {best_team_late}\n"
        stats_text += f"Worst Late Rate Team: {worst_team_late}\n"
        stats_text += f"Best Avg Hours Team: {best_team_hours}\n"

        stats_label = tk.Label(stats_frame, text=stats_text, font=("Arial", 12), 
                              bg="#f0f0f0", justify=tk.LEFT)
        stats_label.pack(padx=20, pady=10, anchor=tk.W)

    def show_cumulative_dashboard(self):
        if self.df is None:
            return

        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(14, 10))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        # Overall stats
        total_employees = self.df['Name'].nunique()
        total_records = len(self.df)
        avg_late = (self.df['Late_Arrival'] == 'Yes').mean() * 100
        avg_early = (self.df['Early_Leave'] == 'Yes').mean() * 100
        avg_hours = self.df['Working_Hours'].mean()

        # Plot 1: Overall Late and Early Rates
        ax1 = fig.add_subplot(3, 2, 1)
        rates = [avg_late, avg_early]
        labels = ['Late Arrival', 'Early Leave']
        colors = ['#3498db', '#e74c3c']
        ax1.pie(rates, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Overall Attendance Rates')

        # Plot 2: Monthly Average Trend
        ax2 = fig.add_subplot(3, 2, 2)
        monthly_trend = self.df.groupby(['Year', 'Month']).agg({
            'Late_Arrival': lambda x: (x == 'Yes').mean() * 100,
            'Early_Leave': lambda x: (x == 'Yes').mean() * 100
        }).reset_index()
        monthly_trend['Month-Year'] = monthly_trend.apply(lambda row: f"{row['Month']}-{row['Year']}", axis=1)
        sns.lineplot(x='Month-Year', y='Late_Arrival', data=monthly_trend, ax=ax2, marker='o', label='Late')
        sns.lineplot(x='Month-Year', y='Early_Leave', data=monthly_trend, ax=ax2, marker='o', label='Early')
        ax2.set_title('Monthly Average Trend')
        ax2.set_ylabel('Rate (%)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.legend()

        # Plot 3: Cumulative Working Hours
        ax3 = fig.add_subplot(3, 2, 3)
        cumulative_hours = self.df.groupby(['Year', 'Month'])['Working_Hours'].sum().reset_index()
        cumulative_hours['Month-Year'] = cumulative_hours.apply(lambda row: f"{row['Month']}-{row['Year']}", axis=1)
        sns.barplot(x='Month-Year', y='Working_Hours', data=cumulative_hours, ax=ax3, color='#2ecc71')
        ax3.set_title('Cumulative Working Hours')
        ax3.set_ylabel('Total Hours')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

        # Plot 4: Loss Hours Percentage
        ax4 = fig.add_subplot(3, 2, 4)
        workdays = self.df.groupby(['Year', 'Month'])['Date'].count().reset_index()
        workdays.columns = ['Year', 'Month', 'Workdays']
        cumulative_hours = cumulative_hours.merge(workdays, on=['Year', 'Month'])
        cumulative_hours['Expected_Hours'] = cumulative_hours['Workdays'] * 9
        cumulative_hours['Loss_Hours'] = cumulative_hours['Expected_Hours'] - cumulative_hours['Working_Hours']
        cumulative_hours['Loss_Percent'] = (cumulative_hours['Loss_Hours'] / cumulative_hours['Expected_Hours']) * 100
        sns.barplot(x='Month-Year', y='Loss_Percent', data=cumulative_hours, ax=ax4, color='#e74c3c')
        ax4.set_title('Loss Hours Percentage')
        ax4.set_ylabel('Loss %')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)

        # Plot 5: Deviation Analysis
        ax5 = fig.add_subplot(3, 2, 5)
        baseline_hours = 9
        self.df['Hours_Deviation'] = baseline_hours - self.df['Working_Hours']
        sns.histplot(self.df['Hours_Deviation'], ax=ax5, kde=True, color='#9b59b6')
        ax5.set_title('Working Hours Deviation')
        ax5.set_xlabel('Deviation (hours)')
        ax5.axvline(x=0, color='r', linestyle='--')

        # Plot 6: Employee Performance
        ax6 = fig.add_subplot(3, 2, 6)
        emp_perf = self.df.groupby('Name').agg({
            'Late_Arrival': lambda x: (x == 'Yes').mean() * 100,
            'Early_Leave': lambda x: (x == 'Yes').mean() * 100,
            'Working_Hours': 'mean'
        }).reset_index()
        emp_perf['Score'] = 100 - (emp_perf['Late_Arrival'] + emp_perf['Early_Leave']) / 2
        sns.barplot(x='Name', y='Score', data=emp_perf.sort_values('Score', ascending=False), ax=ax6, palette='viridis')
        ax6.set_title('Employee Performance Score')
        ax6.set_ylabel('Score')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=90)

        # Add the figure to the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add dashboard statistics
        stats_frame = tk.Frame(self.plot_frame, bg="#f0f0f0")
        stats_frame.pack(fill=tk.X, pady=10)

        stats_text = f"Cumulative Dashboard:\n"
        stats_text += f"Total Employees: {total_employees}\n"
        stats_text += f"Total Records: {total_records}\n"
        stats_text += f"Avg Late Rate: {avg_late:.1f}%\n"
        stats_text += f"Avg Early Rate: {avg_early:.1f}%\n"
        stats_text += f"Avg Working Hours: {avg_hours:.1f}h\n"

        stats_label = tk.Label(stats_frame, text=stats_text, font=("Arial", 12), 
                              bg="#f0f0f0", justify=tk.LEFT)
        stats_label.pack(padx=20, pady=10, anchor=tk.W)

# Main function to run the application
def main():
    root = tk.Tk()
    app = EmployeeAnalyticsApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
