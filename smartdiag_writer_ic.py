import pandas as pd
import random
from datetime import datetime
import os

# File path
csv_path = "smartdiag_dynamic_writer_IC.csv"

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

def generate_ic_row():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    vehicle_type = "IC"

    battery_v = round(random.uniform(11.0, 15.5), 2)
    battery_c = round(random.uniform(0, 80), 2)
    battery_t = round(random.uniform(5, 60), 2)
    coolant_t = round(random.uniform(40, 120), 2)
    rpm = round(random.uniform(500, 7000), 2)
    motor_current = round(random.uniform(0, 100), 2)
    wheel_speed = round(random.uniform(0, 120), 2)
    brake_pressure = round(random.uniform(0, 60), 2)

    # Multi-label fault detection logic
    faults = []
    if battery_v < 11.5 or battery_v > 15:
        faults.append("Battery_Voltage_Abnormal")
    if battery_c == 0:
        faults.append("Battery_NoCurrentDraw")
    if battery_t > 50:
        faults.append("Battery_Overheat")
    if coolant_t > 110 or coolant_t < 50:
        faults.append("Coolant_Temperature_Fault")
    if rpm > 6000:
        faults.append("RPM_Exceeds_Redline")
    if brake_pressure == 0 and rpm > 800:
        faults.append("Brake_Failure")
    if abs(wheel_speed - (rpm / 40)) > 5:
        faults.append("Wheel_Speed_Mismatch")
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
data = [generate_ic_row() for _ in range(50)]
df = pd.DataFrame(data, columns=columns)

print(df.head())
print(f"Number of rows: {len(df)}")

if not os.path.exists(csv_path):
    df.to_csv(csv_path, header=True, index=False, lineterminator='\n')
else:
    df.to_csv(csv_path, mode='a', header=False, index=False, lineterminator='\n')

print("IC engine vehicle data written to", csv_path)
