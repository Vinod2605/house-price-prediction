# House Price Prediction (King County, USA)
# Goal: predict house prices using Linear, Polynomial and Ridge Regression.
# Tip: each chart opens in a window. Close the window to continue the program.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Folder to save the charts (used later in the README)
os.makedirs("images", exist_ok=True)

# ---------------------------------------------------------
# Step 1: Load the data
# ---------------------------------------------------------
url = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/"
    "FinalModule_Coursera/data/kc_house_data_NaN.csv"
)
df = pd.read_csv(url)

# Remove columns we do not need
df = df.drop(columns=["id", "Unnamed: 0"], errors="ignore")
print(df.describe())

# ---------------------------------------------------------
# Step 2: Fill missing values with the average
# ---------------------------------------------------------
print("\nMissing bedrooms:", df["bedrooms"].isna().sum())
print("Missing bathrooms:", df["bathrooms"].isna().sum())

df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].mean())
df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].mean())

print("\nHouses by number of floors:")
print(df["floors"].value_counts().to_frame())

# ---------------------------------------------------------
# Step 3: Explore the data with charts
# ---------------------------------------------------------
# Chart 1: price with and without a waterfront view
sns.boxplot(x="waterfront", y="price", data=df)
plt.xlabel("Waterfront view (0 = no, 1 = yes)")
plt.ylabel("Price")
plt.title("Price vs Waterfront View")
plt.savefig("images/house_price_vs_waterfront.png", dpi=150)
plt.show()

# Chart 2: house size vs price
sns.regplot(x="sqft_above", y="price", data=df)
plt.ylim(0, None)
plt.title("Square Feet Above Ground vs Price")
plt.savefig("images/house_price_vs_sqft_above.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# Step 4: Which columns are most related to price?
# ---------------------------------------------------------
numeric_columns = df.select_dtypes(include=[np.number])
print("\nCorrelation with price:")
print(numeric_columns.corr()["price"].sort_values(ascending=False))

# ---------------------------------------------------------
# Step 5: Split the data into training and test sets
# The model learns from the training data (85%).
# We then check it on the test data (15%) that it has never seen.
# ---------------------------------------------------------
features = [
    "floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
    "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living",
]

train_data, test_data = train_test_split(df, test_size=0.15, random_state=1)
y_train = train_data["price"]
y_test = test_data["price"]

print("\nTraining samples:", len(train_data))
print("Test samples:", len(test_data))

# ---------------------------------------------------------
# Step 6: Build and test the models
# R2 = how well the model fits (closer to 1 is better)
# RMSE = average error in dollars (smaller is better)
# ---------------------------------------------------------
print("\n--- Model Results (on test data) ---")

# Model 1: Simple Linear Regression using longitude only
model_1 = LinearRegression()
model_1.fit(train_data[["long"]], y_train)
pred_1 = model_1.predict(test_data[["long"]])
print("1. Linear (long only)    R2:", round(r2_score(y_test, pred_1), 3),
      " RMSE:", round(mean_squared_error(y_test, pred_1) ** 0.5))

# Model 2: Simple Linear Regression using sqft_living only
model_2 = LinearRegression()
model_2.fit(train_data[["sqft_living"]], y_train)
pred_2 = model_2.predict(test_data[["sqft_living"]])
print("2. Linear (sqft_living)  R2:", round(r2_score(y_test, pred_2), 3),
      " RMSE:", round(mean_squared_error(y_test, pred_2) ** 0.5))

# Model 3: Multiple Linear Regression using all 11 features
model_3 = LinearRegression()
model_3.fit(train_data[features], y_train)
pred_3 = model_3.predict(test_data[features])
print("3. Linear (all features) R2:", round(r2_score(y_test, pred_3), 3),
      " RMSE:", round(mean_squared_error(y_test, pred_3) ** 0.5))

# Model 4: Polynomial Regression (scale the data, add polynomial terms, then fit)
model_4 = Pipeline([
    ("scale", StandardScaler()),
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression()),
])
model_4.fit(train_data[features], y_train)
pred_4 = model_4.predict(test_data[features])
print("4. Polynomial            R2:", round(r2_score(y_test, pred_4), 3),
      " RMSE:", round(mean_squared_error(y_test, pred_4) ** 0.5))

# Model 5: Ridge Regression
model_5 = Ridge(alpha=0.1)
model_5.fit(train_data[features], y_train)
pred_5 = model_5.predict(test_data[features])
print("5. Ridge                 R2:", round(r2_score(y_test, pred_5), 3),
      " RMSE:", round(mean_squared_error(y_test, pred_5) ** 0.5))

# Model 6: Polynomial + Ridge Regression
model_6 = Pipeline([
    ("scale", StandardScaler()),
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", Ridge(alpha=0.1)),
])
model_6.fit(train_data[features], y_train)
pred_6 = model_6.predict(test_data[features])
print("6. Polynomial + Ridge    R2:", round(r2_score(y_test, pred_6), 3),
      " RMSE:", round(mean_squared_error(y_test, pred_6) ** 0.5))

# ---------------------------------------------------------
# Step 7: Chart comparing the models
# ---------------------------------------------------------
model_names = ["Long only", "Sqft living", "All features", "Polynomial", "Ridge", "Poly + Ridge"]
r2_scores = [
    r2_score(y_test, pred_1), r2_score(y_test, pred_2), r2_score(y_test, pred_3),
    r2_score(y_test, pred_4), r2_score(y_test, pred_5), r2_score(y_test, pred_6),
]

plt.figure(figsize=(9, 4))
plt.barh(model_names, r2_scores, color="blue")
plt.xlabel("Test R2 (higher is better)")
plt.title("House Price - Model Comparison")
plt.gca().invert_yaxis()
plt.savefig("images/house_price_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
