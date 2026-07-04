# Creating Historical data set to train ML model
# One year of Weather Data from Delhi (2025)

import requests
import pandas as pd

latitude = 28.6139
longitude = 77.2090

url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}"
    f"&longitude={longitude}"
    "&start_date=2025-01-01"
    "&end_date=2025-12-31"
    "&hourly="
    "temperature_2m,"
    "relative_humidity_2m,"
    "surface_pressure,"
    "precipitation,"
    "wind_speed_10m,"
    "wind_direction_10m"
    "&timezone=Asia/Kolkata"
)

response = requests.get(url)

weather = response.json()

weather.keys()



weather_df = pd.DataFrame(weather["hourly"])
weather_df.head()
weather_df.shape

weather_df.rename(columns={
    "time": "DateTime",
    "temperature_2m": "Temperature (°C)",
    "relative_humidity_2m": "Humidity (%)",
    "surface_pressure": "Pressure (hPa)",
    "precipitation": "Rainfall (mm)",
    "wind_speed_10m": "Wind Speed (km/h)",
    "wind_direction_10m": "Wind Direction (°)"
}, inplace=True)

weather_df.to_csv("Delhi_Weather_2025.csv", index=False)


aq_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={latitude}"
    f"&longitude={longitude}"
    "&start_date=2025-01-01"
    "&end_date=2025-12-31"
    "&hourly="
    "pm2_5,"
    "pm10,"
    "carbon_monoxide,"
    "nitrogen_dioxide,"
    "sulphur_dioxide,"
    "ozone,"
    "aerosol_optical_depth,"
    "dust,"
    "us_aqi,"
    "european_aqi"
    "&timezone=Asia/Kolkata"
)

response = requests.get(aq_url)

air_quality = response.json()
air_quality.keys()
aq_df = pd.DataFrame(air_quality["hourly"])
aq_df.head()
aq_df.shape

aq_df.rename(columns={
    "time": "DateTime",
    "pm2_5": "PM2.5 (µg/m³)",
    "pm10": "PM10 (µg/m³)",
    "carbon_monoxide": "CO (µg/m³)",
    "nitrogen_dioxide": "NO₂ (µg/m³)",
    "sulphur_dioxide": "SO₂ (µg/m³)",
    "ozone": "Ozone (µg/m³)",
    "dust": "Dust (µg/m³)",
    "aerosol_optical_depth": "AOD",
    "us_aqi": "US AQI",
    "european_aqi": "EU AQI"
}, inplace=True)

merged_df = pd.merge(
    weather_df,
    aq_df,
    on="DateTime",
    how="inner"
)
merged_df.head()
merged_df.head()

import cdsapi
import calendar

client = cdsapi.Client()

year = 2025

latitude = 28.6139
longitude = 77.2090

area = [
    latitude + 0.5,      # North
    longitude - 0.5,      # West
    latitude - 0.5,      # South
    longitude + 0.5       # East
]

for month in range(1,13):
    month_str = f"{month:02d}"
    days = calendar.monthrange(year, month)[1]
    day_list = [f"{d:02d}" for d in range(1, days+1)]
    time_list = [f"{h:02d}:00" for h in range(24)]
    filename = f"PBL_{year}_{month_str}.nc"

    print("Downloading", filename)

    client.retrieve(

        "reanalysis-era5-single-levels",
        {
            "product_type":"reanalysis",
            "variable":"boundary_layer_height",
            "year":str(year),
            "month":month_str,
            "day":day_list,
            "time":time_list,
            "data_format":"netcdf",
            "download_format":"unarchived",
            "area":area
        },

        filename

    )

import xarray as xr
import pandas as pd
import glob

files = sorted(glob.glob("PBL_2025_*.nc"))
print(files)
pbl_list = []

for file in files:

    ds = xr.open_dataset(file)

    pbl = ds["blh"].sel(
        latitude=28.6139,
        longitude=77.2090,
        method="nearest"
    )

    df = pbl.to_dataframe().reset_index()
    pbl_list.append(df)

pbl_df = pd.concat(pbl_list, ignore_index=True)

pbl_df = pbl_df.rename(columns={
    "valid_time": "DateTime",
    "blh": "PBL Height (m)"
})

print(pbl_df.columns)

pbl_df = pbl_df[["DateTime", "PBL Height (m)"]]


# merging

weather_df["DateTime"] = pd.to_datetime(weather_df["DateTime"])
aq_df["DateTime"] = pd.to_datetime(aq_df["DateTime"])
pbl_df["DateTime"] = pd.to_datetime(pbl_df["DateTime"])

merged_df = weather_df.merge(aq_df, on="DateTime", how="inner")
merged_df = merged_df.merge(pbl_df, on="DateTime", how="left")



merged_df = merged_df.drop(
    columns=["number", "latitude", "longitude", "expver"],
    errors="ignore"
)

merged_df.head()

# final dataset
merged_df["DateTime"] = pd.to_datetime(merged_df["DateTime"])

merged_df["Year"] = merged_df["DateTime"].dt.year
merged_df["Month"] = merged_df["DateTime"].dt.month
merged_df["Day"] = merged_df["DateTime"].dt.day
merged_df["Hour"] = merged_df["DateTime"].dt.hour
merged_df["DayOfYear"] = merged_df["DateTime"].dt.dayofyear
merged_df["Weekday"] = merged_df["DateTime"].dt.dayofweek

def season(month):
    if month in [11, 12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5, 6]:
        return "Summer"
    elif month in [7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

merged_df["Season"] = merged_df["Month"].apply(season)

merged_df.to_csv(
    "Delhi_Environmental_Dataset_2025.csv",
    index=False
)

print("Dataset Saved Successfully!")
 # checking for any missing value
merged_df.isnull().sum()

# Dataset is ready, Lets train Random Forest
# import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Define Inputs and output
features = [
    "Temperature (°C)",
    "Humidity (%)",
    "Pressure (hPa)",
    "Rainfall (mm)",
    "Wind Speed (km/h)",
    "PBL Height (m)"
]

X = merged_df[features]

y = merged_df["PM2.5 (µg/m³)"]

# Train-Test Split
split = int(len(merged_df) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:] # training set is the first 80% of the year, and your test set is the last 20%.

datetime_test = merged_df["DateTime"].iloc[split:].reset_index(drop=True)

datetime_test = merged_df.loc[X_test.index, "DateTime"].reset_index(drop=True)

#standardize
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

#Prediction
pred = model.predict(X_test_scaled)

#Evaluation
mae = mean_absolute_error(y_test, pred)

rmse = np.sqrt(mean_squared_error(y_test, pred))

r2 = r2_score(y_test, pred)

print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("R²  :", round(r2,4))

#Feature Importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

#Plotting Feature importance


plt.figure(figsize=(8,5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")

plt.title(" Atmospheric Features Importance (Random Forest)")

plt.gca().invert_yaxis()
plt.savefig("Atmospheric Features Importance (Random Forest)")
plt.show()

# SHAP

!pip install shap
import shap
explainer = shap.TreeExplainer(model)

# To keep it fast, let's explain only the first 100 test samples.
X_sample = X_test_scaled[:100]

shap_values = explainer.shap_values(X_test_scaled)

#Summary Plot
shap.summary_plot(
    shap_values,
     X_test_scaled,
    feature_names=features
)
plt.savefig("SHAP_Summary.png", dpi=300, bbox_inches="tight")

#Bar Plot
shap.summary_plot(
    shap_values,
     X_test_scaled,
    feature_names=features,
    plot_type="bar"
)
plt.savefig("SHAP_Summary_Bar.png", dpi=300, bbox_inches="tight")

#Dependence plot for PBL
shap.dependence_plot(
    "PBL Height (m)",
    shap_values,
     X_test_scaled,
    feature_names=features
)
plt.savefig("SHAP_PBL_Dependence.png", dpi=300, bbox_inches="tight")
#Dependence plot for Temp
shap.dependence_plot(
    "Temperature (°C)",
    shap_values,
     X_test_scaled,
    feature_names=features
)
plt.savefig("SHAP_Temperature_Dependence.png", dpi=300, bbox_inches="tight")

# XAI Report
# AQI Category Function
def pm25_category(pm25):

    if pm25 <= 12:
        return "Good"
    elif pm25 <= 35.4:
        return "Moderate"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25 <= 150.4:
        return "Unhealthy"
    elif pm25 <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# Explanation Dictionary

feature_text = {
"Temperature (°C)":{
"high":"The observed temperature contributed to a higher PM2.5 prediction.",
"low":"The observed temperature contributed to a lower PM2.5 prediction."
},
"Humidity (%)":{
"high":"High humidity favored particle growth and haze formation.",
"low":"Low humidity reduced particle growth."
},
"Pressure (hPa)":{
"high":"Higher pressure created more stable atmospheric conditions that can trap pollutants.",
"low":"Lower pressure generally supports better atmospheric mixing."
},
"Rainfall (mm)":{
"high":"Rainfall removed pollutants from the atmosphere through wet deposition.",
"low":"Little or no rainfall allowed pollutants to remain suspended."
},
"Wind Speed (km/h)":{
"high":"Higher wind speed dispersed pollutants efficiently.",
"low":"Weak winds limited pollutant dispersion."
},
"PBL Height (m)":{
"high":"A higher Planetary Boundary Layer promoted vertical mixing and dilution of pollutants.",
"low":"A shallow Planetary Boundary Layer trapped pollutants close to the surface."
}
}

# XAI Function

def explain_prediction(sample):

    prediction = model.predict(
        X_test_scaled[sample].reshape(1,-1)
    )[0]
    category = pm25_category(prediction)
    explanation = pd.DataFrame({
        "Feature":features,
        "Value":X_test.iloc[sample].values,
        "SHAP":shap_values[sample]
    })

    explanation["ABS"] = explanation["SHAP"].abs()
    explanation = explanation.sort_values(
        "ABS",
        ascending=False
    )

    print("="*65)
    print(f"Date & Time     : {datetime_test.iloc[sample]}")
    print(f"Predicted PM2.5 : {prediction:.2f} µg/m³")
    print(f"Air Quality     : {category}")
    print("="*65)

    print("\nTop contributing factors:\n")

    for _,row in explanation.head(4).iterrows():
        feature = row["Feature"]
        value = row["Value"]
        shap = row["SHAP"]
        direction = "high" if shap>0 else "low"
        print(f"{feature}: {value:.2f}")
        print(" ",feature_text[feature][direction])

        print()
    print("="*65)
    print("Overall Explanation:\n")

    positive = explanation[explanation.SHAP>0]
    negative = explanation[explanation.SHAP<0]

    if prediction>100:

        print("Atmospheric conditions favored pollutant accumulation,")
        print("leading to elevated PM2.5 concentrations.")

    else:

        print("Meteorological conditions promoted pollutant")
        print("dispersion and dilution, resulting in relatively")
        print("lower PM2.5 concentrations.")

    print("="*65)

# Report of data

explain_prediction(2) # choose any sample number as you like
