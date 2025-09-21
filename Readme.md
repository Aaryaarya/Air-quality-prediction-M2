
# Air Quality Project – Milestone 2

This repository contains code and resources for the **Air Quality Forecasting & Dashboard project**, focusing on **Milestone 2**, but also includes **Milestone 1** resources.

---

## 🚀 Project Overview

- Train and evaluate models (**ARIMA, Prophet, LSTM, XGBoost**) for air quality prediction.
- Visualize historical pollutant trends.
- Compare model performance using metrics like **RMSE, MAE, MAPE, sMAPE**.
- Streamlit dashboards for model training, predictions, and results visualization.

---

## 📁 Repository Structure

```

AIR-QUALITY-PROJECT/
├── data/
│   ├── processed/
│   │   └── cleaned\_AQI\_dataset.csv
│   └── raw/
├── Milestone\_1/
│   └── Visualizations/
├── Milestone\_2/
│   ├── all\_metrics\_summary.csv
│   ├── bestmodel.zip
│   ├── model\_metrics\_20250921\_1712.txt
│   └── model\_results.txt
├── scripts/
│   ├── **pycache**/
│   ├── eda\_save.py
│   ├── eda.py
│   ├── milestone\_1.ipynb
│   ├── train\_all\_pollutants.py
│   ├── train\_simple.py
│   └── utils.py
├── milestone1\_dashboard.py
├── milestone2\_dashboard.py
├── requirements.txt
├── .gitignore
└── README.md

````

---

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd AIR-QUALITY-PROJECT
````

2. **Create a virtual environment**

```bash
python -m venv venv
```

3. **Activate the virtual environment**

* **Windows (PowerShell):**

```bash
.\venv\Scripts\Activate.ps1
```

* **Windows (CMD):**

```bash
.\venv\Scripts\activate.bat
```

* **Linux / macOS:**

```bash
source venv/bin/activate
```

4. **Install required packages**

```bash
pip install -r requirements.txt
```

5. **Unzip pre-trained models (optional)**

> The `Milestone_1/bestmodel.zip` file contains all pre-trained model files.
> These files are large, so they are provided as a zip to reduce repository size.
> To use them for faster testing, unzip it into the `bestmodel/` folder:

```bash
unzip Milestone_1/bestmodel.zip -d bestmodel/
```

6. **Run the Streamlit dashboards**

* **Milestone 1 dashboard:**

```bash
streamlit run milestone1_dashboard.py
```

* **Milestone 2 dashboard:**

```bash
streamlit run milestone2_dashboard.py
```

---

## ⚠️ Notes

* `venv/` and other large files like `.pkl` models are ignored in `.gitignore`.
* Milestone 2 dashboard uses the CSV data and trained models to display predictions and metrics.
* Large models are zipped to reduce repository size; unzip them before running the dashboards.

---

## 📊 Features

* Compare air quality predictions for **multiple pollutants**.
* Interactive visualization of trends and model results.
* Evaluate models using multiple error metrics.
* Easy-to-use Streamlit dashboards for visual insights.

---


```










