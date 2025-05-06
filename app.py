import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox

# === Step 1: Load & Process Data ===
dataset = pd.read_csv("climate-change.csv")
dataset.replace("..", np.nan, inplace=True)

yearly_columns = [str(year) for year in range(1990, 2012)]
dataset[yearly_columns] = dataset[yearly_columns].apply(pd.to_numeric, errors='coerce')
dataset.drop(columns=['Country code', 'Country name', 'Series code', 'SCALE', 'Decimals'], inplace=True)

target_data = dataset[dataset['Series name'].str.contains("CO2 emissions", case=False, na=False)].copy()
target_data.drop(columns=['Series name'], inplace=True)

# Updated fillna
target_data = target_data.ffill(axis=1)

target_data = target_data.apply(pd.to_numeric, errors='coerce')
target_data.dropna(inplace=True)

features = target_data[yearly_columns[:-1]]  # 1990-2010
labels = target_data['2011']

# === Step 2: Train the Model ===
model = LinearRegression()
model.fit(features, labels)

# === Step 3: Build UI with Tkinter ===
root = tk.Tk()
root.title("CO₂ Emission Predictor (2011)")
entries = {}

tk.Label(root, text="Enter CO₂ emissions from 1990 to 2010:", font=("Helvetica", 14)).pack(pady=10)

input_frame = tk.Frame(root)
input_frame.pack()

for year in range(1990, 2011):
    row = tk.Frame(input_frame)
    row.pack(padx=5, pady=2, anchor="w")
    tk.Label(row, text=str(year), width=8).pack(side=tk.LEFT)
    ent = tk.Entry(row, width=10)
    ent.insert(0, "0")
    ent.pack(side=tk.LEFT)
    entries[str(year)] = ent

def predict():
    try:
        values = [float(entries[str(y)].get()) for y in range(1990, 2011)]
        prediction = model.predict([values])[0]
        messagebox.showinfo("Prediction", f"🌍 Predicted CO₂ Emission in 2011: {prediction:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{e}")

tk.Button(root, text="Predict", command=predict, bg="green", fg="white", font=("Helvetica", 12)).pack(pady=10)

root.mainloop()
