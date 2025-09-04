import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mplcursors
import mplcursors
import matplotlib.ticker as mtick
import ML  

# Country codes 
countries = {
    "India": "IN", "United States": "US", "China": "CN", "Japan": "JP",
    "Germany": "DE", "United Kingdom": "GB", "France": "FR",
    "Brazil": "BR", "South Africa": "ZA", "Australia": "AU"
}

selected_countries = []   
gdp_history = {}          # historical + predictions

CHART_COLORS = ['#ff6384', '#36a2eb', '#cc65fe', '#ffce56', '#4bc0c0', '#9966ff']
BACKGROUND_COLOR = "#121212"
TEXT_COLOR = "#ffffff"

def fetch_gdp_data():
    global gdp_history
    gdp_history.clear()

    for country, _ in selected_countries:
        
        hist = ML.get_historical_data(country)
        years = hist["Year"].tolist()
        values = hist["GDP"].tolist()

        preds = ML.predict_future(country, years=5)  # next 5 years
        pred_years = [y for y, _ in preds]
        pred_values = [v for _, v in preds]

        gdp_history[country] = {
            "years": years + pred_years,
            "values": values + pred_values,
            "split_index": len(years)  
        }

# 🔹 Update chart
def update_chart(frame):
    plt.cla()
    for i, (country, data) in enumerate(gdp_history.items()):
        years = data['years']
        values = data['values']
        split = data['split_index']

        # Historical GDP (solid line)
        plt.plot(
            years[:split], values[:split],
            marker='o', label=f"{country} (Actual)",
            color=CHART_COLORS[i % len(CHART_COLORS)], linewidth=2
        )

        # Predicted GDP (dashed line)
        plt.plot(
            years[split:], values[split:],
            marker='x', linestyle='--',
            label=f"{country} (Predicted)",
            color=CHART_COLORS[i % len(CHART_COLORS)], linewidth=2
        )
        

        last_actual_year = max(years[:split])
        plt.axvspan(last_actual_year, max(years),
                    color="gray", alpha=0.1, label="Forecast Zone")

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("GDP Visualization with ML Forecasting",
              fontsize=18, fontweight='bold', color='#16a085')
    plt.xlabel("Year", fontsize=14, fontweight='bold', color=TEXT_COLOR)
    plt.ylabel("GDP (in USD)", fontsize=14, fontweight='bold', color=TEXT_COLOR)
    plt.xticks(rotation=45, color=TEXT_COLOR)
    plt.yticks(color=TEXT_COLOR)
    plt.legend(loc='upper left', fontsize=10)
    
    plt.gca().yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f'${x/1e12:.1f}T')
    )

    plt.tight_layout()

    cursor = mplcursors.cursor(hover=True)
    @cursor.connect("add")
    def on_add(sel):
        year = int(sel.target[0])
        gdp = sel.target[1]
        gdp_trillions = gdp / 1e12
        sel.annotation.set_text(f"Year: {year}\nGDP: ${gdp_trillions:.2f}T")


def select_countries():
    global selected_countries
    selected_countries.clear()
    gdp_history.clear()

    names = simpledialog.askstring(
        "Select Countries",
        "Enter country names separated by commas (e.g. India, United States):"
    )
    if names:
        for name in names.split(','):
            name = name.strip().title()
            if name in countries:
                selected_countries.append((name, countries[name]))
            else:
                messagebox.showwarning("Invalid Country", f"Country '{name}' is not valid.")

        if selected_countries:
            countries_display_label['text'] = "Selected Countries: " + ", ".join([c[0] for c in selected_countries])
            messagebox.showinfo("Countries Selected", f"Selected: {', '.join([c[0] for c in selected_countries])}")
        else:
            countries_display_label['text'] = ""
            messagebox.showwarning("No Countries", "Please select at least one valid country.")

# Start chart
def start_live_chart():
    if not selected_countries:
        messagebox.showwarning("No Countries Selected", "Please select countries first.")
        return

    fetch_gdp_data()  # load's the  ML data

    fig = plt.figure(figsize=(12, 6))
    ani = FuncAnimation(fig, update_chart, interval=2000) 
    plt.gcf().patch.set_facecolor(BACKGROUND_COLOR)
    plt.show()

root = tb.Window(themename="darkly")
root.title("GDP Visualization with ML Forecasting")
root.geometry("800x500")

frame = ttk.Frame(root, padding=20)
frame.pack(pady=20, padx=20, expand=True)

title_label = ttk.Label(
    frame, text="GDP Forecasting (ML Powered)",
    font=("Arial", 20, 'bold'), bootstyle="success")
title_label.grid(row=0, column=0, columnspan=2, pady=10)

countries_display_label = ttk.Label(
    frame, text="", font=("Arial", 12), bootstyle="info")
countries_display_label.grid(row=1, column=0, columnspan=2, pady=10)

select_button = ttk.Button(
    frame, text="Select Countries", command=select_countries,
    bootstyle="primary-outline", padding=(10, 5))
select_button.grid(row=2, column=0, padx=20, pady=10)

start_button = ttk.Button(
    frame, text="Start Chart", command=start_live_chart,
    bootstyle="success-outline", padding=(10, 5))
start_button.grid(row=2, column=1, padx=20, pady=10)

footer_label = ttk.Label(
    root, text="Created with Python | ML + Data Visualization",
    font=("Arial", 12), bootstyle="info")
footer_label.pack(side="bottom", pady=5)

root.mainloop()
