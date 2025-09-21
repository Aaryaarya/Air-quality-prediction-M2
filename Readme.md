
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
│   │   └── cleaned_AQI_dataset.csv
│   └── raw/
├── Milestone_1/
│   └── Visualizations/
├── Milestone_2/
│   ├── all_metrics_summary.csv
│   ├── bestmodel.zip
│   ├── model_metrics_20250921_1712.txt
│   └── model_results.txt
├── scripts/
│   ├── **pycache**/
│   ├── eda_save.py
│   ├── eda.py
│   ├── milestone_1.ipynb
│   ├── train\_all_pollutants.py
│   ├── train_simple.py
│   └── utils.py
├── milestone1_dashboard.py
├── milestone2_dashboard.py
├── requirements.txt
├── .gitignore
└── README.md

````

---

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Aaryaarya/Air-quality-prediction-M2.git>
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

<img width="1918" height="1027" alt="Screenshot 2025-09-21 231725" src="https://github.com/user-attachments/assets/732b351c-82ec-4528-912b-04eb99d4bd8a" />

<img width="1919" height="1016" alt="Screenshot 2025-09-21 231657" src="https://github.com/user-attachments/assets/7c93c9be-56c9-4371-afd8-bde120711167" />
<img width="1919" height="1030" alt="Screenshot 2025-09-21 232125" src="https://github.com/user-attachments/assets/30d0a371-0b42-4561-8926-50146627bcc9" />

<img width="1911" height="970" alt="Screenshot 2025-09-21 232248" src="https://github.com/user-attachments/assets/fb884000-6993-4f6a-a2e9-06ad1659934c" />




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










