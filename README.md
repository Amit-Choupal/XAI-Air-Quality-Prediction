# XAI-Air-Quality-Prediction
Explainable AI framework for predicting PM2.5 concentrations using meteorological variables and Planetary Boundary Layer (PBL) height. Built with Python, Random Forest, SHAP, ERA5, Open-Meteo, and OpenAQ datasets.

# Project Overview

Air pollution is one of the most serious environmental challenges affecting cities around the world. Among all air pollutants, PM2.5 (Particulate Matter smaller than 2.5 micrometers) is considered one of the most dangerous because these tiny particles are nearly thirty times smaller than the width of a human hair. Due to their extremely small size, they can bypass the body's natural defense mechanisms, travel deep into the lungs, enter the bloodstream, and contribute to respiratory diseases, cardiovascular problems, and reduced life expectancy.
Predicting PM2.5 concentrations is therefore important not only for environmental monitoring but also for public health and sustainable urban planning. However, air pollution is influenced by many interacting atmospheric processes. Pollutants emitted from vehicles and industries do not simply remain where they are released. Their concentration depends heavily on weather conditions, atmospheric stability, rainfall, wind, and the vertical structure of the atmosphere.

The objective of this project is to develop an Explainable Artificial Intelligence (XAI) model capable of predicting hourly PM2.5 concentrations using real atmospheric observations while also explaining the scientific reasons behind every prediction. Instead of producing only a numerical value, the model identifies which atmospheric variables were responsible for increasing or decreasing PM2.5 concentrations.

The study uses Delhi, India as a case study because it frequently experiences severe air pollution episodes and exhibits strong seasonal variations that make it suitable for studying atmospheric controls on pollutant accumulation.

# Dataset Description and Data Acquisition

This project combines multiple real-world atmospheric datasets obtained from internationally recognized scientific sources, every observation represents actual environmental conditions measured or reconstructed for the year 2025.

Meteorological variables were collected using the Open-Meteo Historical Weather API. These observations include hourly temperature, relative humidity, atmospheric pressure, rainfall, and wind speed. Each of these variables influences the behaviour of pollutants in different ways. Wind speed controls how efficiently pollutants disperse away from their source, rainfall removes airborne particles through a process known as wet deposition, humidity affects the growth of aerosol particles, while temperature and pressure influence atmospheric stability and mixing.

Air quality observations were obtained using the Open-Meteo Air Quality API, which provides hourly concentrations of PM2.5, PM10, Carbon Monoxide (CO), Nitrogen Dioxide (NO₂), Sulfur Dioxide (SO₂), Ozone (O₃), Aerosol Optical Depth (AOD), Dust concentration, and Air Quality Index values. Although several pollutants were collected, PM2.5 was selected as the target variable because it represents one of the most harmful pollutants affecting human health.

One of the key scientific contributions of this project is the inclusion of Planetary Boundary Layer (PBL) Height, downloaded from the Copernicus Climate Data Store (ERA5 Reanalysis).

The Planetary Boundary Layer is the lowest part of Earth's atmosphere that is directly influenced by the surface. During daytime, sunlight heats the ground, causing warm air to rise and creating turbulence. This increases the height of the boundary layer, allowing pollutants to mix within a much larger volume of air. As a result, pollutant concentrations near the ground generally decrease.
At night, the ground cools rapidly, atmospheric mixing becomes much weaker, and the boundary layer becomes shallow. Since pollutants can no longer mix vertically, they remain trapped close to the surface, often causing large increases in PM2.5 concentrations. This daily cycle makes PBL Height one of the most important atmospheric variables controlling urban air quality.
After downloading all datasets, they were merged using their common DateTime information to create a single hourly dataset representing the atmospheric conditions over Delhi throughout the year.

# Methodology

The workflow of this project follows a complete environmental data science pipeline, beginning with real atmospheric observations and ending with an Explainable Artificial Intelligence model.
The first stage involved collecting meteorological observations, air quality measurements, and Planetary Boundary Layer Height from different scientific data sources. Since each dataset used the same hourly timestamps, they were merged into a unified dataset where every row represents one hour of atmospheric conditions.

Before training the machine learning model, the dataset was examined for missing values and inconsistencies. No missing observations were found, allowing the complete dataset to be used without interpolation.

Although many atmospheric variables were available, the final machine learning model was intentionally designed using only six physically meaningful predictors: temperature, humidity, atmospheric pressure, rainfall, wind speed, and Planetary Boundary Layer Height. These variables describe the meteorological processes responsible for transporting, dispersing, and removing pollutants from the atmosphere.

PM2.5 concentration was selected as the prediction target.

Because atmospheric observations form a time series, the dataset was divided chronologically into training and testing periods. This allows the model to learn from earlier observations and evaluate its performance on future data, making the prediction process more realistic.

Random Forest Regression produced the best overall performance and was therefore selected as the final prediction model. 

To improve transparency, the final Random Forest model was combined with SHAP (SHapley Additive exPlanations). SHAP calculates how much each atmospheric variable contributed to an individual prediction. This allows the model to explain not only what PM2.5 concentration is expected, but also why it is expected.

For example, instead of simply predicting a high PM2.5 concentration, the model can explain that the prediction is mainly due to a shallow Planetary Boundary Layer, weak winds, high humidity, and the absence of rainfall.

# Results and Discussion

The Random Forest model successfully learned the relationship between meteorological conditions and PM2.5 concentration, achieving an R² score of approximately 0.68, meaning that nearly seventy percent of the variability in hourly PM2.5 concentrations could be explained using only six atmospheric variables.

Feature importance analysis revealed that temperature was the strongest predictor, followed by wind speed, humidity, Planetary Boundary Layer Height, pressure, and rainfall. These results are consistent with atmospheric science because pollutant concentrations depend strongly on weather conditions and the ability of the atmosphere to disperse emissions.

One particularly important observation was the contribution of Planetary Boundary Layer Height. Even without using pollutant concentrations such as NO₂ or CO as input variables, the model identified PBL Height as one of the major factors influencing PM2.5. This demonstrates that atmospheric mixing processes play a significant role in determining pollution levels.

SHAP analysis further strengthened the interpretation of the model. Instead of behaving like a traditional "black box," the Explainable AI model identified the atmospheric conditions responsible for each prediction. For example, when the model predicted high PM2.5 concentrations, SHAP frequently highlighted low PBL Height, weak wind speed, and dry conditions as major contributors. Similarly, low PM2.5 predictions were generally associated with stronger winds, deeper boundary layers, and rainfall, all of which promote pollutant dispersion and removal.

This combination of machine learning and atmospheric science makes the model scientifically interpretable rather than purely statistical.

# Sustainability Perspective

One of the most important aspects of air pollution management is understanding when emissions become most harmful. The same amount of pollution released into the atmosphere can produce very different air quality depending on atmospheric conditions.

During winter nights or early mornings, the Planetary Boundary Layer often becomes very shallow. Pollutants emitted from industries, vehicles, and other human activities remain trapped close to the ground because there is very little vertical mixing. As a result, PM2.5 concentrations can rise rapidly even if emission rates remain unchanged.

On the other hand, during sunny afternoons, the atmosphere becomes much more turbulent, the boundary layer grows deeper, and pollutants disperse into a much larger volume of air. This naturally reduces their concentration near the surface.

Understanding these atmospheric processes provides opportunities for more sustainable environmental management. Industrial operations, traffic restrictions, or emission-control measures could be timed to coincide with periods when atmospheric conditions are least capable of dispersing pollutants. Instead of applying identical emission strategies throughout the day, environmental agencies could use atmospheric forecasts to identify high-risk periods and implement targeted interventions.

By combining real atmospheric observations, machine learning, and Explainable AI, this project demonstrates how environmental data science can support smarter air-quality management, sustainable urban planning, and evidence-based climate adaptation strategies.

------------------------------------------------------------------*--------------------------------------------------------------------------------------------
