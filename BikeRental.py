import pandas as pd
import numpy as np

df = pd.read_csv("03_bike_rental.csv")

print(df.shape)
print(df.dtypes)
print(df.head())
print(df.describe())

print(df.isnull().sum())
median_humidity = df["Humidity_Percent"].median()
df["Humidity_Percent"] = df["Humidity_Percent"].fillna(median_humidity)
print("Missing after fill:", df["Humidity_Percent"].isnull().sum())

df["DateTime"] = pd.to_datetime(df["DateTime"])
df["Date"] = df["DateTime"].dt.date
df["Month"] = df["DateTime"].dt.month
df["Hour_Extracted"] = df["DateTime"].dt.hour

total_rentals = df["Rental_Count"].sum()
average_rentals = df["Rental_Count"].mean()
print("Total rentals:", total_rentals)
print("Average rentals:", round(average_rentals, 2))

top10 = df.sort_values("Rental_Count", ascending=False).head(10)
print(top10[["Record_ID", "DateTime", "Season", "Weather", "Rental_Count"]])

season_avg = df.groupby("Season")["Rental_Count"].mean().sort_values(ascending=False)
print(season_avg)

weekend_avg = df.groupby("Is_Weekend")["Rental_Count"].mean()
print(weekend_avg)

weather_avg = df.groupby("Weather")["Rental_Count"].mean().sort_values(ascending=False)
print(weather_avg)

hourly = df.groupby("Hour")["Rental_Count"].agg(["mean", "sum", "count"]).round(2)
print(hourly)
print("Peak hour:", hourly["mean"].idxmax())

temp_array = df["Temperature_C"].to_numpy()
rental_array = df["Rental_Count"].to_numpy()
print("Average temperature (NumPy):", round(np.mean(temp_array), 2))
print("Average rental count (NumPy):", round(np.mean(rental_array), 2))

q1 = np.percentile(rental_array, 25)
q3 = np.percentile(rental_array, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df[(df["Rental_Count"] < lower_bound) | (df["Rental_Count"] > upper_bound)]
print("Q1:", q1, "Q3:", q3, "IQR:", iqr)
print("Bounds:", lower_bound, upper_bound)
print("High/unusual records:", len(outliers))
print(outliers[["Record_ID", "DateTime", "Rental_Count"]])

season_weather = df.groupby(["Season", "Weather"])["Rental_Count"].agg(["mean", "sum", "count"]).round(2)
print(season_weather)

daily = df.groupby("Date")["Rental_Count"].sum().sort_values(ascending=False)
print(daily.head(10))
sorted_records = df.sort_values("Rental_Count", ascending=False)
print(sorted_records.head(10))

correlation = df["Temperature_C"].corr(df["Rental_Count"])
print("Correlation (temperature vs rentals):", round(correlation, 4))

df.to_csv("bike_rental_cleaned.csv", index=False)

best_season = season_avg.idxmax()
best_weather = weather_avg.idxmax()
worst_weather = weather_avg.idxmin()
peak_hour = hourly["mean"].idxmax()
low_hour = hourly["mean"].idxmin()
print("Best season:", best_season)
print("Best weather:", best_weather)
print("Worst weather:", worst_weather)
print("Peak hour:", peak_hour, "| Lowest hour:", low_hour)ṇṇ


