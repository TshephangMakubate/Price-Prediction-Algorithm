# Importing Required Packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

def compute_model_output(x, w, b):
    return w * x + b

def compute_cost(x, y, w, b):
    m = x.shape[0]
    f_wb = compute_model_output(x, w, b)
    cost = np.sum((f_wb - y) ** 2)
    total_cost = (1 / (2 * m)) * cost
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    f_wb = w * x + b
    error = f_wb - y
    dj_dw = (1 / m) * np.dot(error, x)
    dj_db = (1 / m) * np.sum(error)
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha=1.0e-3, num_iters=10000, tolerance=1.0e-6):
    b = b_in
    w = w_in
    prev_cost = float('inf')

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        b -= alpha * dj_db
        w -= alpha * dj_dw
        current_cost = compute_cost(x, y, w, b)
        if abs(prev_cost - current_cost) < tolerance:
            break
        prev_cost = current_cost

    return w, b

# Load data
nat_gas_df = pd.read_csv("Nat_Gas.csv")
nat_gas_df["Dates"] = pd.to_datetime(nat_gas_df["Dates"], format='%m/%d/%y')
nat_gas_df['Numeric_dates'] = (nat_gas_df['Dates'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(nat_gas_df['Numeric_dates'].values.reshape(-1, 1)).flatten()
y_train = nat_gas_df["Prices"].to_numpy()

# Train model
w, b = gradient_descent(x_train, y_train, 0, 0)
model_prediction = compute_model_output(x_train, w, b)

# Accuracy metrics
r2 = r2_score(y_train, model_prediction)
mae = mean_absolute_error(y_train, model_prediction)
print(f"Model R² score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(nat_gas_df["Dates"], y_train, marker='x', c='r', label='Actual Values')
plt.plot(nat_gas_df["Dates"], model_prediction, c='b', label='Our Prediction')
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45, fontweight="bold")
plt.yticks(fontweight="bold")
plt.title("Natural Gas Price Prediction")
plt.legend()
plt.tight_layout()
plt.show()

# Forecasting
date_input = input("Enter the date (mm/dd/yy) for price estimation: ")
try:
    date_obj = pd.to_datetime(date_input, format='%m/%d/%y')
    temp_date_input = (date_obj - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
    scaled_date_input = scaler.transform([[temp_date_input]])[0][0]
    price_estimation = (scaled_date_input * w) + b
    print(f"The estimated price on {date_input} is ${price_estimation:.2f}")
except ValueError:
    print("Invalid date format. Please use mm/dd/yy.")
