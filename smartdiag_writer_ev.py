import pandas as pd
import random
from datetime import datetime
import os

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

    # Multi-label fault detection logic
    faults = []
    if battery_v < 330 or battery_v > 420:
        faults.append("Battery_Voltage_Abnormal")
    if battery_c > 200:
        faults.append("Battery_Current_Overload")
    if battery_t < 10 or battery_t > 60:
        faults.append("Battery_Temperature_Abnormal")
    if coolant_t > 110 or coolant_t < 50:
        faults.append("Coolant_Temperature_Fault")
    if rpm > 9000:
        faults.append("Motor_RPM_Overlimit")
    if motor_current > 200:
        faults.append("Motor_Current_Overdraw")
    if abs(wheel_speed - (rpm / 40)) > 5:
        faults.append("Wheel_Speed_Mismatch")
    if brake_pressure == 0 and rpm > 1000:
        faults.append("Brake_Failure")
    if not faults:
        faults.append("Normal")

    fault_label = ";".join(faults)

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
        fault_label
    ]

# Generate and write 50 rows
data = [generate_ev_row() for _ in range(50)]
df = pd.DataFrame(data, columns=columns)

print(df.head())
print(f"Number of rows: {len(df)}")

if not os.path.exists(csv_path):
    df.to_csv(csv_path, header=True, index=False, lineterminator='\n')
else:
    df.to_csv(csv_path, mode='a', header=False, index=False, lineterminator='\n')

print("EV vehicle data written to", csv_path)
