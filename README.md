# Flight departure delay prediction model

For the final project of the course Scalable Machine Learning and Deep Learning (HT25), we (Álvaro Mazcuñán Herreros and Jonas Lorenz) implemented a daily pipeline for the prediction of flight delays based on weather data at big European airports.

## Problem description

We tackled the hard problem of predicting flight delays. What makes the problem a tough one to tackle is the variety of reasons that can cause flight delays. There is plenty of stages in which delays can be introduced, like labor shortages, previous flights being delayed, weather conditions and rare events like medical emergencies. We focused on the data everyone has access to, namely weather (via the Open-Meteo API). To get a history of flights, we used the Aviation Edge API, which has the history of flights going back one year with departure delays and flight schedules. The list of airports we focused on are London Heathrow (LHR), Frankfurt (FRA), Amsterdam Shiphol (AMS), Copenhagen Airport (CPH), Charles de Gaulle Airport (CDG), Istanbul Airport (IST), Madrid Adolfo Suárez-Barajas (MAD), Barcelona-El Prat (BCN), Rome Fiuicino (FCO) and Munich Airport (MUC). After training plenty of models, we decided that our best shot at predicting meaningfully was to focus on departure delays, as arrival delays are usually lower since pilots can make up for delays by increasing speed and because planes, once in the air, cannot be kept in the air for ever just because of poor weather conditions at the arrival airport.

## Data and Features

### Data Sources
* **Flight Data:** Sourced via the [Aviation Edge API](https://aviation-edge.com/), including scheduled/actual times, airline IATA codes, and delay durations (in minutes).
* **Weather Data:** High-resolution hourly archives from [Open-Meteo](https://open-meteo.com/). Features are matched to flights by "flooring" the scheduled departure time to the nearest hour.

### Primary Features

#### Meteorological Features (Origin & Destination)
Weather is the primary driver of variance in this model. For every flight, we capture:

| Feature | Description |
| :--- | :--- |
| `temperature_2m` | Air temperature at 2 meters above ground. |
| `precipitation` | Combined rain, snow, and sleet (mm). |
| `wind_speed_10m` | Sustained wind speeds (km/h) at 10 meters. |
| `wind_gusts_10m` | Maximum instantaneous wind speed (km/h). |
| `pressure_msl` | Sea-level pressure (hPa), used to detect storm systems. |
| `cloudcover` | Total cloud cover percentage. |
| `weather_code` | WMO code representing specific conditions (e.g., fog, thunderstorm). |

#### Cyclical Time Encoding
We transformed timestamps into circular coordinates using **Sine and Cosine** transformations. We used the times for both departure and arrival.

#### Spatial & Wind Encoding
* **Origin One-Hot Encoding:** Categorical variables for each hub airport to account for airport-specific ways of dealing with given weather conditions
* **Wind Direction Vectors:** Wind direction (0-360°) is decomposed into `wind_dir_sin` and `wind_dir_cos`. This prevents the model from seeing 359° and 1° as opposites, treating them instead as nearly identical northern winds.

### Data Pre-processing & Target
* **The Target ($y$):** We predict the **Natural Log** of the departure delay: $\log(1 + \text{delay})$. This "Log1p" transformation compresses extreme outliers and helps the regressor focus on the more common lower delay range. Moreover, the log-transform serves as a variance reducing transformation.
* **Delay Clipping:** Delays are clipped at **180 minutes** during training to prevent extreme technical delays (plane breakdowns) from skewing the model's understanding of weather-related delays.
* **Duplicate Handling:** The pipeline strictly enforces uniqueness of the flights to prevent data leakage.

### Feature Summary Table

| Group | Final Feature Count |
| :--- | :--- | :--- |
| **Airports** | 10 (One-Hot) |
| **Weather** | 10 (including 2 for wind direction)|
| **Time** | 4 (Cyclical) |
