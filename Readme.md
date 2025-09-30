
# Air Quality Project – Milestone 2

This repository contains code and resources for the **Air Quality Forecasting & Dashboard project**, focusing on **Milestone 3,2**, but also includes **Milestone 1** resources.

---

## 🚀 Project Overview

- Train and evaluate models (**ARIMA, Prophet, LSTM, XGBoost**) for air quality prediction.
- Visualize historical pollutant trends.
- Compare model performance using metrics like **RMSE, MAE, MAPE, sMAPE**.
- Streamlit dashboards for model training, predictions, and results visualization.
-uploaded the csv file and Convert pollutant data to AQI and generate alerts with insights based on air quality levels
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

   Run the Final Dashboard:
To see the outputs of the final dashboard , run the Streamlit app:

```bash
streamlit run app.py
```



The project also includes a dashboard for each milestone to visualize outputs:
* **Milestone 1 dashboard:**

```bash
streamlit run milestone1_dashboard.py
```

* **Milestone 2 dashboard:**

```bash
streamlit run milestone2_dashboard.py
```


* **Milestone 3-Alert System dashboard:**

```bash
streamlit run milestone2_dashboard.py
```

---
Final output:

<img width="1919" height="994" alt="image" src="https://github.com/user-attachments/assets/b114850b-6118-45c1-82b0-e97b2497d638" />
<img width="1919" height="996" alt="image" src="https://github.com/user-attachments/assets/8bcb9c6f-1943-428e-9c60-d4251dda5615" />
<img width="1911" height="996" alt="image" src="https://github.com/user-attachments/assets/593147fe-9582-4cab-b214-e36be3c15513" />
<img width="1919" height="1002" alt="image" src="https://github.com/user-attachments/assets/bbba4569-a4b6-4982-8bed-b79188822f8c" />




MILESTONE1
<img width="1918" height="1027" alt="Screenshot 2025-09-21 231725" src="https://github.com/user-attachments/assets/732b351c-82ec-4528-912b-04eb99d4bd8a" />

<img width="1919" height="1016" alt="Screenshot 2025-09-21 231657" src="https://github.com/user-attachments/assets/7c93c9be-56c9-4371-afd8-bde120711167" />
MILESTONE-2
<img width="1919" height="1030" alt="Screenshot 2025-09-21 232125" src="https://github.com/user-attachments/assets/30d0a371-0b42-4561-8926-50146627bcc9" />

<img width="1911" height="970" alt="Screenshot 2025-09-21 232248" src="https://github.com/user-attachments/assets/fb884000-6993-4f6a-a2e9-06ad1659934c" />

MILESTONE-3
<img width="1843" height="852" alt="Screenshot 2025-09-23 215749" src="https://github.com/user-attachments/assets/6f733f65-fb6d-4ff5-810d-893d5a099e0f" />
<img width="1909" height="1013" alt="Screenshot 2025-09-23 215923" src="https://github.com/user-attachments/assets/d18d416f-3a1d-4e0a-b5ae-11ddc989c40b" />




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










