
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
data_path = "../data/processed/cleaned_AQI_dataset.csv"
df = pd.read_csv(data_path, parse_dates=['Date'])

# Quick check
print(df.head())
print(df.info())
print(df.isnull().sum())

# -----------------------------
# Plot 1: AQI trend for a city
city = 'Bangalore'
city_df = df[df['City'] == city]

plt.figure(figsize=(12,5))
plt.plot(city_df['Date'], city_df['AQI'], label='AQI')
plt.title(f'{city} AQI Over Time')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.show()

# -----------------------------
# Plot 2: Pollutant distributions
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

plt.figure(figsize=(12,6))
sns.boxplot(data=df[pollutants])
plt.title('Pollutant Distribution Across All Cities')
plt.show()

# -----------------------------
# Plot 3: Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[pollutants + ['AQI']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between AQI and Pollutants')
plt.show()

# -----------------------------
# Plot 4: Monthly AQI trend for a city
df['Month'] = df['Date'].dt.month
monthly_avg = df[df['City']=='Bangalore'].groupby('Month')['AQI'].mean()

plt.figure(figsize=(10,5))
monthly_avg.plot(kind='bar')
plt.title('Average Monthly AQI - Bangalore')
plt.ylabel('AQI')
plt.show()
