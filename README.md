---

# GDP Forecasting & Visualization

## Project Overview

This project combines **Python development** and **Machine Learning** to analyze and forecast GDP trends for multiple countries.
It uses World Bank open data, applies a **Log-Linear Regression model**, and provides an interactive visualization with both historical and predicted GDP values.

---

## Features

* Fetches real GDP data from the World Bank API
* Forecasts future GDP using **Machine Learning (Log-Linear Regression with log transform)**
* Accuracy for India and other contries: **R² = 0.97** (MAE: 1.14e+11, RMSE: 1.81e+11)
* Tkinter GUI with ttkbootstrap theme for user-friendly interaction
* Interactive Matplotlib charts with tooltips and shaded forecast zone

---

## Tech Stack

* Python (3.x)
* Tkinter + ttkbootstrap (GUI)
* Matplotlib + mplcursors (interactive visualization)
* Scikit-learn (ML regression model)
* Pandas & NumPy (data handling)
* World Bank Open Data API

---

## Project Structure

```
GDP-Forecasting-Visualization/
│── data/
│   └── gdp_data.csv         # Fetched GDP dataset
│── models/                  # Saved ML models
│── DVP_Project.py           # Main GUI + visualization
│── ML.py                    # Machine Learning module
│── README.md
```

---

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/channveerswamiHiremath/GDP-Forecasting-Visualization.git
   cd GDP-Forecasting-Visualization
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the GUI:

   ```bash
   python DVP_Project.py
   ```

---

## Example Output

* Historical GDP plotted alongside forecasted GDP (next 5 years).
* Forecast zone shaded in gray.
* Tooltips display GDP in trillions (`$23.9T` format).

*(Insert screenshot or GIF here)*

---

## Future Improvements

* Add more indicators like population, inflation, and trade balance.
* Try advanced forecasting models like Prophet or LSTMs.
* Export forecasts as CSV/Excel for deeper analysis.

---

## About

This project was developed out of a keen interest in exploring how Machine Learning and Data Visualization can be combined to solve real-world economic challenges.

---
