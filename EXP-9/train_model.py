import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CSV_PATH = "house_data.csv"   # columns: Area, Bedrooms, Bathrooms, Age, Distance_City, Garage, Price

# ---------- 1. Data Collection ----------
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    # Simulated housing data (same schema as a real dataset) used when no CSV is present
    rng = np.random.default_rng(11)
    n = 1500
    area = rng.uniform(500, 3500, n).round(0)                       # square feet
    bedrooms = np.clip(np.round(area / 750 + rng.normal(0, 0.6, n)), 1, 5).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 4)
    age = rng.integers(0, 31, n)                                    # years
    distance = rng.uniform(1, 30, n).round(1)                       # km from city centre
    garage = rng.integers(0, 2, n)
    price = (10 + 0.045 * area + 4 * bedrooms + 3 * bathrooms - 0.6 * age
             - 1.2 * distance + 5 * garage + rng.normal(0, 8, n))   # price in lakhs
    df = pd.DataFrame({"Area": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms, "Age": age,
                       "Distance_City": distance, "Garage": garage, "Price": price.round(2)})
    df.to_csv(CSV_PATH, index=False)
    return df

df = load_data()

# ---------- 2. Data Preprocessing ----------
print("Missing values before cleaning:", int(df.isnull().sum().sum()))
df = df.fillna(df.median(numeric_only=True))       # handle missing values
df = df.drop_duplicates().reset_index(drop=True)   # remove duplicate rows

# ---------- 3. Feature Selection ----------
FEATURES = ["Area", "Bedrooms", "Bathrooms", "Age", "Distance_City", "Garage"]
X, y = df[FEATURES], df["Price"]

# ---------- 4. Train-Test Split (80-20) ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 5. Model Training ----------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------- 6. Prediction and Evaluation ----------
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("Total records      :", len(df))
print("Training records   :", len(X_train))
print("Testing records    :", len(X_test))
print("Intercept          :", round(model.intercept_, 4))
for name, coef in zip(FEATURES, model.coef_):
    print(f"Coefficient {name:<13}: {coef:.4f}")
print(f"MAE                : {mae:.4f} lakhs")
print(f"MSE                : {mse:.4f}")
print(f"RMSE               : {rmse:.4f} lakhs")
print(f"R2 Score           : {r2:.4f}")

# ---------- 7. Visualization ----------
plt.figure(figsize=(6.5, 5))
plt.scatter(y_test, pred, alpha=0.5, color="#2e7d32", edgecolors="none")
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, color="#d32f2f", linestyle="--", label="Perfect prediction")
plt.title("House Price: Actual vs Predicted (Test Set)")
plt.xlabel("Actual Price (lakhs)")
plt.ylabel("Predicted Price (lakhs)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=200)

# ---------- 8. Save model for the web app ----------
joblib.dump(model, "model.pkl")
json.dump({"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "train": len(X_train), "test": len(X_test)},
          open("metrics.json", "w"))
