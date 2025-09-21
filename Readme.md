# Air Quality Project – Milestone 2

This repository contains the code and resources for the **Air Quality Forecasting & Dashboard** project ( focusing Milestone 2,also contains milestone 1).  

---

## 🚀 Project Overview

- Train and evaluate models (ARIMA, Prophet, LSTM, XGBoost) for air quality prediction.
- Visualize historical pollutant trends.
- Compare model performance using metrics like RMSE, MAE, MAPE, and sMAPE.
- Streamlit dashboard for model training, predictions, and results visualization.

---

## 📁 Repository Structure
AIR-QUALITY-PROJECT/
├── data/
│   ├── processed/
│   │   └── cleaned_AQI_dataset.csv
│   ├── raw/
├── Milestone_1/
│   └── Visualizations/
├── Milestone_2/
│   ├── all_metrics_summary.csv
│   ├── bestmodel.zip
│   ├── model_metrics_20250921_1712.txt
│   └── model_results.txt
├── scripts/
│   ├── __pycache__/
│   ├── eda_save.py
│   ├── eda.py
│   ├── milestone_1.ipynb
│   ├── train_all_pollutants.py
│   ├── train_simple.py
│   └── utils.py
├── venv/
├── .gitignore
├── milestone1_dashboard.py
├── milestone2_dashboard.py
├── README.md
└── requirements.txt




---

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd AIR-QUALITY-PROJECT

2. **Create a virtual environment**
```bash
python -m venv venv
 
 3.Activate the virtual environment
Windows (PowerShell)

.\venv\Scripts\Activate.ps1
Windows (CMD)
.\venv\Scripts\activate.bat

4.Install required packages

pip install -r requirements.txt

5. **Unzip pre-trained models (optional)**  
> The `Milestone1/bestmodels.zip` file contains all the pre-trained model files. These files are large, so for now they are provided as a zip to reduce repo size.  
> To use them for faster testing, unzip it into the `bestmodel/` folder:

```bash
unzip Milestone1/bestmodels.zip -d bestmodel/

6. **Run the Streamlit dashboard**
```bash
streamlit run milestone1_dashboard.py
 for Milestone 1 dashboard or
```bash
streamlit run milestone2_dashboard.py
 for Milestone 2 dashboard

 