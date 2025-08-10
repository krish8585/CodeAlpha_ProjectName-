# Load the dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Unemployment in India.csv')

# Display initial information about the dataset
print("Initial DataFrame Info:")
df.info()

# Rename the columns for easier use
df.columns = [
    'Region',
    'Date',
    'Frequency',
    'Estimated Unemployment Rate (%)',
    'Estimated Employed',
    'Estimated Labour Participation Rate (%)',
    'Area',
]

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Display the first few rows and updated info
print("\nDataFrame after cleaning:")
print(df.head())
print("\nDataFrame Info after cleaning:")
df.info()

# ----------------------------------------------------
# Analyze unemployment trends by area (Urban vs. Rural)
# ----------------------------------------------------
# Group by 'Date' and 'Area' and calculate the mean unemployment rate
unemployment_by_area = (
    df.groupby(['Date', 'Area'])['Estimated Unemployment Rate (%)']
    .mean()
    .reset_index()
)

# Plotting the unemployment rate by area
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=unemployment_by_area,
    x='Date',
    y='Estimated Unemployment Rate (%)',
    hue='Area',
)
plt.title('Unemployment Rate by Area (Rural vs. Urban)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.savefig('unemployment_by_area.png')

# -----------------------------------------------------------------
# Investigate the impact of Covid-19 (around March-April-May 2020)
# -----------------------------------------------------------------
# Filter data for the Covid-19 lockdown period
covid_start = '2020-03-01'
covid_end = '2020-06-30'
covid_period_df = df[
    (df['Date'] >= covid_start) & (df['Date'] <= covid_end)
]

# Calculate and print the mean unemployment rate before and during the period
pre_covid_df = df[df['Date'] < '2020-03-01']
pre_covid_avg = pre_covid_df['Estimated Unemployment Rate (%)'].mean()
covid_avg = covid_period_df['Estimated Unemployment Rate (%)'].mean()

print(f'\nAverage Unemployment Rate (Pre-Covid): {pre_covid_avg:.2f}%')
print(f'Average Unemployment Rate (During Covid): {covid_avg:.2f}%')

# ---------------------------------------------------------------------------------------
# Visualize the impact of Covid-19 on unemployment rates across the country
# ---------------------------------------------------------------------------------------
# Plot unemployment rate over time with Covid-19 period highlighted
plt.figure(figsize=(15, 7))
sns.lineplot(
    data=df, x='Date', y='Estimated Unemployment Rate (%)', color='blue'
)
plt.axvspan(
    pd.to_datetime('2020-03-25'),
    pd.to_datetime('2020-05-31'),
    color='red',
    alpha=0.3,
    label='Covid-19 Lockdown',
)
plt.title('Estimated Unemployment Rate in India Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.savefig('unemployment_rate_over_time.png')

# ---------------------------------------------------------------------------------------
# Identify key patterns or seasonal trends in the data
# ---------------------------------------------------------------------------------------
# Visualize monthly average unemployment rate
df['Month'] = df['Date'].dt.month
monthly_avg_unemployment = (
    df.groupby('Month')['Estimated Unemployment Rate (%)'].mean()
)

plt.figure(figsize=(10, 6))
monthly_avg_unemployment.plot(kind='bar', color='skyblue')
plt.title('Average Monthly Unemployment Rate')
plt.xlabel('Month')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(ticks=range(12), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
plt.grid(axis='y')
plt.savefig('monthly_unemployment_trend.png')

# ---------------------------------------------------------------------------------------
# Analyze Unemployment Rate by Region (for a more detailed view)
# ---------------------------------------------------------------------------------------
# Calculate the average unemployment rate for each region
avg_unemployment_by_region = (
    df.groupby('Region')['Estimated Unemployment Rate (%)']
    .mean()
    .sort_values(ascending=False)
)

# Plotting the average unemployment rate by region
plt.figure(figsize=(15, 8))
sns.barplot(
    x=avg_unemployment_by_region.values,
    y=avg_unemployment_by_region.index,
    palette='viridis',
)
plt.title('Average Unemployment Rate by Region')
plt.xlabel('Average Unemployment Rate (%)')
plt.ylabel('Region')
plt.grid(axis='x')
plt.tight_layout()
plt.savefig('unemployment_by_region.png')

# ---------------------------------------------------------------------------------------
# Present insights that could inform economic or social policies
# ---------------------------------------------------------------------------------------
# For this, I will summarize the findings from the above analysis.
# The code execution will provide the data for this summary.
# I will use the print statements to display key metrics that will be used in the final response.
print('\nTop 5 Regions with Highest Average Unemployment Rate:')
print(avg_unemployment_by_region.head())
print('\nTop 5 Regions with Lowest Average Unemployment Rate:')
print(avg_unemployment_by_region.tail())