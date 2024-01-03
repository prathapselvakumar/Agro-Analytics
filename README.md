# Agro Analytics

Agro Analytics is a Streamlit web application that brings three decision-support tools for crop planning into one interface. It recommends a crop from four environmental readings, detects leaf disease in uploaded photographs with a YOLO object detector, and fits seasonal ARIMA models to historical rainfall and temperature records. Team Byte Bay Bugs built it during an internship at the National University of Singapore (NUS) on Big Data Analysis using Deep Learning, and it is intended as a working prototype. It has not been validated in the field.

## Overview

Smallholder farmers often choose crops and respond to plant disease with little access to agronomic advice or local climate data. Agro Analytics tests whether a small set of standard machine learning methods, packaged behind a login page, can give a non-specialist a first answer to three practical questions:

1. Which crop suits the temperature, humidity, soil pH, and rainfall of my plot?
2. Does this leaf show signs of disease?
3. What rainfall and temperature should I expect in the coming period?

Each question maps to one module, described in the methodology section below.

## Methodology

### Crop recommendation

The recommender is trained on `dataset/crop_recommendation.csv`, which contains 2,200 labelled samples covering 22 crops (100 samples per crop, including rice, maize, chickpea, cotton, coffee, and several fruits). The file also records soil nitrogen, phosphorus, and potassium, but the application drops these three columns, so the model uses only temperature (°C), relative humidity (%), soil pH, and rainfall (mm).

Labels are integer-encoded and the data is split 85/15 into training and test sets (`random_state=0`). Three classifiers are trained, each inside a scikit-learn pipeline with a `StandardScaler`:

| Model | Configuration |
| --- | --- |
| Random forest | 9 trees, `random_state=128` |
| XGBoost | library defaults, `random_state=128` |
| Gaussian naive Bayes | library defaults |

The page shows the prediction from all three models side by side, so the user can see whether they agree. A k-nearest-neighbours classifier was also tried and remains in the code, commented out.

### Crop health monitoring

The user uploads a JPEG photograph of a leaf. The image is written to the `temp/` folder and passed to a YOLO model loaded through the Ultralytics library from fine-tuned weights in `model/best.pt`. The page displays the image with the predicted bounding boxes and class labels drawn on it. The training data, class list, and training procedure for these weights are not included in this repository.

### Weather forecasting

Two forecasting routines are in `pages/Forcasting/`:

- `TemperatureForecasting.temperature()` reads daily temperature readings for Mumbai from 1999 to 2019 (`dataset/TemperatureMum_1999-2019.csv`), holds out the last 20% of the series, and fits a SARIMA(1,1,1)(1,1,1,12) model to the first 80%. It forecasts over the held-out period, plots the forecast with a 95% prediction interval, and returns the mean absolute error.
- `RainFallForecasting.rainfall()` fits a SARIMAX(5,1,3) model to the `ANNUAL` column of `dataset/rainfall.csv.xls`, forecasts 50 steps ahead, and plots the result with a 95% prediction interval.

### Authentication

Users register and log in on the landing page. Passwords are hashed with PBKDF2-SHA256 (via `passlib`) and stored in a MySQL table named `users`. A flag in the Streamlit session state controls access, and the analysis pages show only a log-in prompt until it is set.

## Results

The figure `assets/accuracycomp.jpg`, shown on the recommendation page, reports the following classification accuracies for seven algorithms on the crop dataset:

| Algorithm | Accuracy |
| --- | --- |
| Random forest | 0.99 |
| XGBoost | 0.99 |
| Gaussian naive Bayes | 0.99 |
| k-nearest neighbours | 0.97 |
| Logistic regression | 0.95 |
| Decision tree | 0.90 |
| Support vector machine | 0.11 |

The three best-performing models were kept in the application. The script that produced this comparison is not part of the repository, so the exact split and feature set behind these numbers cannot be confirmed from the code. The SVM result is far below the others and most likely reflects an unscaled or untuned baseline rather than the method's real performance on this data. No accuracy figures are reported for the YOLO model, and the temperature MAE is printed to the console without being recorded here.

## Limitations

Readers should keep the following in mind before relying on any output.

- The crop dataset is balanced and small, and models trained on it tend to score very highly. Accuracy on real farm measurements is likely to be lower.
- The recommendation page labels the rainfall input as "Rain (in %)", while the model was trained on rainfall in millimetres. Users need to enter millimetres for the prediction to be meaningful.
- The three classifiers are retrained every time the page reloads. This is fast on 2,200 rows but would not scale to a larger dataset.
- The health module runs prediction on every image in `temp/`, not only the newest upload. Files left in that folder from earlier sessions (one sample image is already there) will appear in the results.
- The rainfall file holds one row per district, not one row per year. The rainfall model therefore treats the sequence of districts as if it were a time series, and its forecast should not be read as a forecast over time.
- The temperature model uses a seasonal period of 12 on daily data, which corresponds to a 12-day cycle rather than an annual one.
- Neither forecasting routine is linked to the Streamlit interface yet. Streamlit only registers scripts placed directly in `pages/`, and nothing imports these modules, so they must be called from Python.
- The `users` table does not enforce unique usernames, and the database credentials are hard-coded in `AgroAnalytics.py`.

## Installation and use

### Requirements

- Python 3.8 or later
- MySQL Server

### Setup

1. Clone the repository or navigate to the project directory
   ```bash
   cd Agro-Analytics
   ```

2. Install dependencies. A virtual environment is recommended:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   Then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the database. Start the MySQL server and create a database named `crops`:
   ```sql
   CREATE DATABASE crops;
   ```
   The application creates the `users` table on first run. It connects as `user="root"` with an empty password; if your MySQL configuration differs, edit `create_connection()` in `AgroAnalytics.py`.

4. Run the application.
   ```bash
   streamlit run AgroAnalytics.py
   ```
   Register an account, log in, and open the recommendation or health page from the sidebar.

To run a forecast, call the functions directly from the project root, for example:

```bash
python -c "from pages.Forcasting.TemperatureForecasting import temperature; print(temperature())"
```

## Repository structure

| Path | Contents |
| --- | --- |
| `AgroAnalytics.py` | Entry point: page setup, registration, and login |
| `pages/1_Crop_Recommendation.py` | Crop recommendation page |
| `pages/2_Crop_Health.py` | Leaf disease detection page |
| `pages/Forcasting/` | Rainfall and temperature forecasting functions |
| `dataset/` | Crop recommendation, district rainfall, and Mumbai temperature data |
| `model/best.pt` | Fine-tuned YOLO weights for the health module |
| `assets/` | Logo and the model accuracy comparison figure |
| `temp/` | Working folder for uploaded leaf images |
| `requirements.txt` | Python dependencies |

## Credits

Developed by Team Byte Bay Bugs as part of the NUS internship programme on Big Data Analysis using Deep Learning.

The project was assessed by Dr Tan Wee Kek and Dr Manoranjan Dash of the NUS School of Computing and received a score of 44 out of 50.
