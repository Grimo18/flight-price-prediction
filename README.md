# ✈️ Flight Price Prediction System

A complete **end-to-end Machine Learning pipeline** for predicting flight prices with 99.87% R² accuracy and interactive analysis tools.

## 🎯 Key Features

- **🤖 ML Ensemble Model**: Random Forest Regressor trained on 300K+ flight records
- **📊 Ground Truth Validation**: MAE ₹161.27* on test set (20K flights)
- **🌍 Multilingual Support**: Italian & English interfaces
- **💱 Smart Currency Detection**: Auto-detects symbols (€, $, ₹, £) or asks user when ambiguous
- **📈 Data Visualization**: Price distribution, seasonality trends, feature importance
- **🎯 Smart Recommendations**: Auto-selects cheapest airline per route
- **🔄 Robust Data Cleaning**: Handles corrupted airline names, duplicates, outliers
- **⚙️ Feature Engineering**: 10 optimized features with StandardScaler normalization

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9987 (99.87%) |
| **MAE** | ₹161.27* |
| **RMSE** | ₹871.33* |
| **Test Set Size** | 60,000 flights |
| **Training Set Size** | 240,000 flights |

*Currency-dependent values (example shown in ₹). Actual values scale with dataset's native currency or user selection.

---

## 🚀 Quick Start

### 1. **Clone Repository**

```bash
git clone https://github.com/yourusername/flight-price-prediction.git
cd flight-price-prediction
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Download Dataset**

⭐ **Download from Kaggle** (Required - NOT included in repo):
- Go to: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction
- Download: `economy.csv`, `business.csv`, `Clean_Dataset.csv`
- Place in project root folder

Folder structure after setup:
```
flight-price-prediction/
├── main_it.py
├── main_en.py
├── requirements.txt
├── README.md
├── economy.csv          ← Add here (from Kaggle)
├── business.csv         ← Add here (from Kaggle)
├── Clean_Dataset.csv    ← Add here (from Kaggle)
└── flight_price_predictor/
```

### 4. **Run the Application**

```bash
# Italian Version
python main_it.py

# English Version
python main_en.py
```

The system will:
1. Load & unify datasets
2. Run exploratory analysis
3. Clean data
4. Generate visualizations
5. Train ML model
6. Show interactive prediction interface

⏱️ **Total runtime**: ~50-80 seconds

---

## 📁 Project Structure

```
flight_price_prediction/
├── main_it.py                      # Italian entry point
├── main_en.py                      # English entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── flight_price_predictor/
│   ├── __init__.py
│   ├── core.py                     # Main pipeline (920 lines)
│   │   ├── load_data()            # Dataset loading + currency detection
│   │   ├── explore_data()         # EDA with multilingual output
│   │   ├── clean_data()           # Robust airline cleaning
│   │   ├── visualize_price_distribution()  # 4-subplot analysis
│   │   ├── analyze_seasonality()  # Monthly trend analysis
│   │   ├── prepare_features()     # Feature engineering + encoding
│   │   ├── train_model()          # RF training + metrics
│   │   └── interactive_prediction() # User interface
│   │
│   ├── models.py                   # ML model wrapper
│   │   ├── prepare_features()     # OneHotEncoder, LabelEncoder, StandardScaler
│   │   ├── train_model()          # 80-20 train-test split, cross-validation
│   │   ├── predict()              # Single prediction
│   │   └── get_feature_importance() # Top 10 features
│   │
│   ├── utils.py                    # Helper functions
│   │   └── format_currency()      # Currency formatting
│   │
│   ├── data_loader.py              # Dataset loading utilities
│   │   ├── carica_o_crea_unificato() # Unifies 3 CSV files
│   │   ├── load_generic_dataset() # Generic CSV loader
│   │   └── file_hash()            # MD5 hash for cache
│   │
│   └── i18n/
│       ├── it.py                   # Italian translations
│       └── en.py                   # English translations
│
├── economy.csv                     # Dataset (economy class flights)  
├── business.csv                    # Dataset (business class flights)
└── Clean_Dataset.csv               # Dataset (all classes - primary)

Note: CSV files are NOT tracked in Git (.gitignore)
      Download from Kaggle after cloning the repo
```

---

## 🔬 Technical Architecture

### **Data Pipeline**
```
Raw CSV (300K rows)
    ↓
Currency Detection (auto-detect symbols or ask user if ambiguous)
    ↓
Airline Cleaning (regex deduplication + whitelist)
    ↓
Feature Engineering (OneHot + LabelEncoder + StandardScaler)
    ↓
Train-Test Split (80-20)
    ↓
Random Forest Training (100 estimators)
    ↓
Cross-Validation (5-fold)
    ↓
Prediction + Visualization
```

### **ML Model Details**
- **Algorithm**: RandomForestRegressor (scikit-learn)
- **Hyperparameters**: 
  - n_estimators=100
  - max_depth=20
  - min_samples_split=10
  - min_samples_leaf=4
- **Features**: 10 engineered features (categorical + numeric)
- **Evaluation**: MAE, RMSE, R², 5-fold cross-validation

### **Output Visualizations**
1. **price_distribution_analysis.png**: 4-subplot analysis
   - Histogram (50 bins)
   - Box plot (outlier detection)
   - Log-scale histogram
   - Statistical summary

2. **seasonality_analysis.png**: Monthly trends
   - Average price per month
   - Flight count per month

3. **feature_importance.png**: Top 10 features
   - Horizontal bar chart
   - Importance scores

---

## 🌍 Multilingual Support

### **Console Output** (Multilingue)
```bash
python main_it.py   # All messages in Italian
python main_en.py   # All messages in English
```

### **Graph Labels** (Always English)
- Standard international format
- Currency symbols detected or selected by user (€, $, ₹, £)
- Month names translated in data display

---

## 📊 Example Usage

### **Italian Version**
```bash
$ python main_it.py

🎯 MENU PRINCIPALE
1. Carica dataset unificato (3 file)
2. Carica file CSV generico
3. Esci

Scelta (1/2/3): 1

📊 Caricamento dataset...

[?] Valuta non rilevata automaticamente.
Scegli la valuta:
  1. € (Euro)
  2. $ (Dollaro)
  3. ₹ (Rupia Indiana)
  4. £ (Sterlina)
Scelta (1/2/3/4): 2

💰 Valuta selezionata: $
✅ Dataset unificato caricato: 300153 righe, 11 colonne

============================================================
📈 ANALISI ESPLORATIVA DEI DATI
============================================================
[*] Prime righe del dataset:
...
[*] Statistiche numeriche:
...

🤖 TRAINING MODELLO ML
============================================================
[*] Training Random Forest Regressor...
    MAE: $161.27
    RMSE: $871.33
    R²: 0.9987

✅ ANALISI COMPLETATA CON SUCCESSO!
```

---

## 🔧 Configuration

### **requirements.txt**
```
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
```

### **.gitignore**

This project is configured to exclude:
- `__pycache__/`, `.venv/` - Cache and virtual environment
- `*.csv` - Raw datasets (600+ MB, user downloads from Kaggle)
- `*.png`, `*.pkl` - Generated files (recreated on each run)

**What's tracked on GitHub**: Only source code (*.py), requirements.txt, and documentation

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"ModuleNotFoundError: No module named 'pandas'"** | Run `pip install -r requirements.txt` |
| **"FileNotFoundError: economy.csv"** | Download from Kaggle or provide your CSV |
| **"Currency not detected"** | Ensure 'price' column has numeric values |
| **Graphs not opening** | Install default image viewer or check file permissions |
| **Slow training** | Reduce dataset size or adjust RandomForest hyperparameters |

---

## 📈 Expected Output Timeline

```
1. Load Data              ~2 seconds
2. Explore Data           ~3 seconds
3. Clean Data             ~5 seconds
4. Visualize Distribution ~3 seconds (+ opens PNG)
5. Analyze Seasonality    ~2 seconds (+ opens PNG)
6. Prepare Features       ~2 seconds
7. Train Model            ~30-60 seconds (depends on CPU)
8. Feature Importance     ~2 seconds (+ opens PNG)
9. Interactive Prediction ~0.5 seconds per prediction

Total Runtime: ~50-80 seconds
```

---

## 🎓 Key Insights from Analysis

### **Price Drivers (Top 5 Features)**
1. Duration of flight
2. Journey month (seasonality)
3. Airline company
4. Number of stops
5. Days left until departure

### **Seasonality Patterns**
- **Cheapest**: February, September
- **Most Expensive**: December, June
- **Price Range**: ₹2,000 - ₹100,000+* (currency-dependent)

### **Best Practices**
- ✅ Book flights 30+ days in advance
- ✅ Travel in off-season months
- ✅ Prefer airlines with lower average prices
- ✅ Morning departures tend to be cheaper

---

## 📝 Author Notes

- **Development Environment**: Python 3.11+
- **Dataset Size**: 300,153 flights across 11 features
- **Training Time**: ~1 minute on modern CPU
- **Memory Usage**: ~500MB during training
- **Model Size**: ~5-10MB when serialized (.pkl)

---

## 🤝 Contributing

Feel free to:
- Submit bug reports
- Suggest new features
- Improve documentation
- Propose new ML algorithms

---

## 📄 License

This project is licensed under the **MIT License** - see `LICENSE` file for details.

You're free to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use in private projects

Just include a copy of the license and give credit.

---

## 🔗 References

- **Dataset**: Kaggle Flight Price Prediction
- **ML Framework**: scikit-learn 1.3+
- **Data Processing**: pandas 2.0+
- **Visualization**: matplotlib + seaborn

---

**Last Updated**: October 2025  
**Status**: Production Ready ✅ 
