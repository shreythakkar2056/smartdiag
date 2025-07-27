import pandas as pd
import random
from datetime import datetime

# File path
csv_path = "smartdiag_dynamic_writer_EV.csv"

# Column headers
columns = [
    "Timestamp",
    "Vehicle_Type",
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

def generate_ev_row():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    vehicle_type = "EV"

    battery_v = round(random.uniform(320, 430), 2)
    battery_c = round(random.uniform(-30, 250), 2)
    battery_t = round(random.uniform(5, 65), 2)
    coolant_t = round(random.uniform(40, 120), 2)
    rpm = round(random.uniform(0, 9500), 2)
    motor_current = round(random.uniform(0, 250), 2)
    wheel_speed = round(random.uniform(0, 120), 2)
    brake_pressure = round(random.uniform(0, 60), 2)

    # Fault detection logic
    if battery_v < 330 or battery_v > 420:
        fault = "Battery_Voltage_Abnormal"
    elif battery_c > 200:
        fault = "Battery_Current_Overload"
    elif battery_t < 10 or battery_t > 60:
        fault = "Battery_Temperature_Abnormal"
    elif coolant_t > 110 or coolant_t < 50:
        fault = "Coolant_Temperature_Fault"
    elif rpm > 9000:
        fault = "Motor_RPM_Overlimit"
    elif motor_current > 200:
        fault = "Motor_Current_Overdraw"
    elif abs(wheel_speed - (rpm / 40)) > 5:
        fault = "Wheel_Speed_Mismatch"
    elif brake_pressure == 0 and rpm > 1000:
        fault = "Brake_Failure"
    else:
        fault = "Normal"

    return [
        timestamp,
        vehicle_type,
        battery_v,
        battery_c,
        battery_t,
        coolant_t,
        rpm,
        motor_current,
        wheel_speed,
        brake_pressure,
        fault
    ]

# Generate and write 50 rows
data = [generate_ev_row() for _ in range(50)]
df = pd.DataFrame(data, columns=columns)

try:
    with open(csv_path, 'x') as f:
        df.to_csv(f, header=True, index=False)
except FileExistsError:
    df.to_csv(csv_path, mode='a', header=False, index=False)

print("EV vehicle data written to", csv_path)
