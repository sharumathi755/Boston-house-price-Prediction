
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (replace with your dataset file)
data = pd.read_csv('Housing.csv')  # Updated to your downloaded file name

# Display basic information about the dataset
print(data.head())
print(data.info())
print(data.describe())

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Handle missing values (if any)
data = data.dropna()  # Simple approach: Drop rows with missing values

# Exploratory data analysis (EDA)
sns.pairplot(data)
plt.show()

# Correlation heatmap
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Define features (X) and target (y)
# Replace these columns with actual columns in your dataset
X = data[['area', 'bedrooms', 'age']]  # Example columns, replace if necessary
y = data['price']  # Replace with your actual target column name if different

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

