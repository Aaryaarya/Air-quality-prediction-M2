import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Paths
data_path = "../data/processed/cleaned_AQI_dataset.csv"
save_path = "../Milestone_1/Visualizations"

# Create folder if not exists
os.makedirs(save_path, exist_ok=True)

# -------------------------------
# Load dataset
df = pd.read_csv(data_path, parse_dates=['Date'])

# List of pollutants
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

# -------------------------------
# 1. AQI trend over time for each city
cities = df['City'].unique()
for city in cities:
    city_df = df[df['City'] == city]
    plt.figure(figsize=(12,5))
    plt.plot(city_df['Date'], city_df['AQI'], label='AQI')
    plt.title(f'{city} AQI Over Time')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.legend()
    plt.savefig(f"{save_path}/AQI_trend_{city}.png")
    plt.close()

# -------------------------------
# 2. Pollutant distributions across all cities (boxplot)
plt.figure(figsize=(12,6))
sns.boxplot(data=df[pollutants])
plt.title('Pollutant Distribution Across All Cities')
plt.savefig(f"{save_path}/Pollutant_Distribution.png")
plt.close()

# -------------------------------
# 3. Correlation heatmap of AQI and pollutants
plt.figure(figsize=(10,6))
sns.heatmap(df[pollutants + ['AQI']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between AQI and Pollutants')
plt.savefig(f"{save_path}/Correlation_Heatmap.png")
plt.close()

# -------------------------------
# 4. Monthly average AQI trend for each city
df['Month'] = df['Date'].dt.month
for city in cities:
    monthly_avg = df[df['City']==city].groupby('Month')['AQI'].mean()
    plt.figure(figsize=(10,5))
    monthly_avg.plot(kind='bar')
    plt.title(f'Average Monthly AQI - {city}')
    plt.ylabel('AQI')
    plt.xlabel('Month')
    plt.savefig(f"{save_path}/Monthly_AQI_{city}.png")
    plt.close()

print("All visualizations saved in Milestone_1/Visualizations folder!")
