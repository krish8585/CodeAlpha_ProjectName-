import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('Advertising.csv')

# 2. Data Preparation and Exploration
# Remove the unnamed first column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print("First 5 rows of the dataset:")
print(df.head())

print("\nInformation about the dataset:")
df.info()

# Let's check for any missing values
print("\nMissing values check:")
print(df.isnull().sum())

# Visualize the relationship between advertising spend and sales
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='reg')
plt.suptitle('Advertising Spend vs. Sales', y=1.02)
plt.savefig('advertising_vs_sales_pairplot.png')

# 3. Use a regression model to forecast sales
# Define features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Analyze how changes in advertising impact sales outcomes
print("\nModel Coefficients:")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance on Test Data:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Visualize predicted vs actual sales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.grid(True)
plt.savefig('actual_vs_predicted_sales.png')

# 5. Deliver actionable insights
# This will be done in the final response by interpreting the coefficients and R2 score.
# The code execution provides the necessary data points for this interpretation.
# For example, the coefficients will show which advertising medium has the most impact.
# The R2 score indicates how well the model explains the variance in the data.
# The MAE gives a sense of the average error in predictions.