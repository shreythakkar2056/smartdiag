import pandas as pd
import random
from datetime import datetime

# File path to append data
csv_path = "smartdiag_dynamic_writer.csv"

# Column headers
columns = [
    "Timestamp",
    "Battery_V",
    "Battery_C",
    "Battery_T",
    "Coolant_T",
    "RPM",
    "Motor_Current",
    "Wheel_FL_Speed",
    "Brake_Pressure",
    "Fault_Label"
]

# Generate a single row of data
def generate_data_row():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    battery_v = round(random.uniform(11.5, 13.0), 2)
    battery_c = round(random.uniform(10, 60), 2)
    battery_t = round(random.uniform(30, 50), 2)
    coolant_t = round(random.uniform(85, 110), 2)
    rpm = round(random.uniform(1500, 3000), 2)
    motor_current = round(random.uniform(20, 70), 2)
    wheel_speed = round(random.uniform(50, 65), 2)
    brake_pressure = round(random.uniform(0, 4), 2)

    # Decide fault condition
    if battery_t > 44:
        fault = "Overheat_Battery"
    elif coolant_t > 100:
        fault = "Coolant_Issue"
    elif battery_v < 11.8:
        fault = "Battery_LowVoltage"
    elif motor_current > 60 and rpm > 2500:
        fault = "Motor_Overload"
    elif brake_pressure > 3.5 and rpm < 1000:
        fault = "Brake_Anomaly"
    elif abs(wheel_speed - (rpm / 40)) > 5:
        fault = "Wheel_Speed_Mismatch"
    else:
        fault = "Normal"

    return [timestamp, battery_v, battery_c, battery_t, coolant_t, rpm, motor_current, wheel_speed, brake_pressure, fault]

# Create 50 new rows
new_data = [generate_data_row() for _ in range(50)]
df_new = pd.DataFrame(new_data, columns=columns)

# If file doesn't exist, write with headers; otherwise, append without headers
try:
    with open(csv_path, 'x') as f:
        df_new.to_csv(f, header=True, index=False)
except FileExistsError:
    df_new.to_csv(csv_path, mode='a', header=False, index=False)

csv_path
