"""
Core Module
===========
Classe principale FlightPricePredictor che coordina
caricamento dati, analisi, training e predizioni.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
import os
import subprocess
import platform
import warnings
import re
warnings.filterwarnings('ignore')

from .models import MLModels
from .utils import format_currency

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def open_file(filepath):
    """Apre un file con l'applicazione predefinita del sistema operativo."""
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', filepath])
        else:  # Linux
            subprocess.run(['xdg-open', filepath])
    except Exception as e:
        print(f"[!] Impossibile aprire {filepath}: {e}")


def _compress_repeated_substring(text: str) -> str:
    """Comprimi pattern di sottostringhe ripetute (es: VistaraVistara -> Vistara).

    Strategia:
    - riduce anche caratteri singoli ripetuti molte volte (aaaaa -> a)
    - cerca la sottostringa più rappresentativa (3..20) che copre >=60% del testo e si ripete >=2 volte
    - restituisce la sottostringa se trovata, altrimenti il testo originale
    """
    if not isinstance(text, str):
        text = str(text)
    s = text.strip()
    if not s:
        return s

    # comprimi caratteri ripetuti (più di 3 consecutivi)
    s = re.sub(r"(.)\1{3,}", r"\1", s)

    n = len(s)
    best_sub = None
    best_score = 0.0

    max_L = min(20, max(3, n // 2))
    for L in range(3, max_L + 1):
        counts = {}
        for i in range(0, n - L + 1):
            sub = s[i:i+L]
            counts[sub] = counts.get(sub, 0) + 1
        if not counts:
            continue
        # valuta la sottostringa più frequente per questa L
        sub, cnt = max(counts.items(), key=lambda kv: kv[1])
        coverage = (cnt * L) / float(n)
        # privilegia copertura e ripetizioni
        score = coverage
        if cnt >= 2 and coverage >= 0.6 and score > best_score:
            best_sub = sub
            best_score = score

    return best_sub if best_sub else s



def _deduplicate_airline(s):
    # Deduplica sequenze ripetute di nomi noti
    known = ['Indigo', 'Vistara', 'Spicejet', 'Airasia', 'Goair', 'AirIndia', 'Trujet', 'StarAir']
    for k in known:
        s = re.sub(f'({k})+', r'\1', s)
    return s

def _clean_airline_value(val):
    # Cleaning robusto: rimuove sequenze ripetute, normalizza, limita lunghezza, fallback
    if not isinstance(val, str) or not val.strip():
        return 'Unknown'
    val = val.strip()
    val = re.sub(r'[^A-Za-z ]', '', val)  # Solo lettere e spazi
    val = _deduplicate_airline(val)
    val = re.sub(r'\s+', ' ', val)
    val = val.strip()
    if not val or len(val) < 3 or any(c.isdigit() for c in val):
        return 'Unknown'
    return val[:30]


def _safe_label(text: str, max_len: int = 40) -> str:
    """Tronca in modo sicuro una label per stampa UI."""
    s = str(text).strip()
    return (s[:max_len-1] + '…') if len(s) > max_len else s


class FlightPricePredictor:
    """
    Sistema completo di predizione prezzi voli con analisi stagionalità.
    
    Caratteristiche:
    - Caricamento e pulizia dati automatica
    - Analisi esplorativa (EDA)
    - Feature engineering
    - Machine learning (Random Forest + Gradient Boosting)
    - Analisi stagionalità dettagliata
    - Predizioni interattive
    - Export risultati
    """
    
    def __init__(self, data_path, lang='en'):
        """
        Inizializza il predittore.
        
        Args:
            data_path (str): Percorso del file CSV del dataset
            lang (str): Lingua dell'interfaccia ('en' o 'it')
        """
        self.data_path = data_path
        self.df = None
        self.df_cleaned = None
        self.ml_models = MLModels()
        self.seasonal_analysis = {}
        self.currency_symbol = '€'  # Default currency
        self.lang = lang  # Language setting
    
    
    def detect_currency(self):
        """
        Rileva automaticamente la valuta dal dataset analizzando i valori di prezzo.
        
        Returns:
            str: Simbolo della valuta rilevata (€, $, ₹, £, etc.)
        """
        if self.df is None or 'price' not in self.df.columns:
            return '€'
        
        # Prendi un campione di prezzi
        sample = self.df['price'].dropna().head(100)
        
        # Controlla se ci sono simboli di valuta nelle stringhe
        sample_str = sample.astype(str)
        
        if sample_str.str.contains('₹', regex=False).any():
            return '₹'
        elif sample_str.str.contains('$', regex=False).any():
            return '$'
        elif sample_str.str.contains('£', regex=False).any():
            return '£'
        elif sample_str.str.contains('€', regex=False).any():
            return '€'
        
        # Se non ci sono simboli, cerca di dedurre dalla grandezza media
        try:
            mean_price = pd.to_numeric(sample, errors='coerce').mean()
            
            # Range tipici per valuta (euristica)
            if mean_price > 50000:  # Probabilmente Rupie indiane
                return '₹'
            elif mean_price > 10000:  # Probabilmente Dollari o Euro
                return '$'  # Default a dollari per valori alti
            else:
                return '€'  # Default a euro per valori bassi
        except:
            return '€'
    
    
    def load_data(self):
        """Carica il dataset da CSV"""
        if self.lang == 'it':
            loading_msg = "📊 Caricamento dataset..."
            currency_detected = "💰 Valuta rilevata: {}"
            loaded_msg = "✅ Dataset caricato: {} righe, {} colonne"
            columns_label = "Colonne disponibili: {}\n"
        else:
            loading_msg = "📊 Loading dataset..."
            currency_detected = "💰 Currency detected: {}"
            loaded_msg = "✅ Dataset loaded: {} rows, {} columns"
            columns_label = "Available columns: {}\n"
        
        print(loading_msg)
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^unnamed', case=False)]
        
        # Rileva la valuta automaticamente
        self.currency_symbol = self.detect_currency()
        print(currency_detected.format(self.currency_symbol))
        
        print(loaded_msg.format(self.df.shape[0], self.df.shape[1]))
        print(columns_label.format(list(self.df.columns)))
        return self.df
    
    
    def explore_data(self):
        """Esplorazione iniziale dei dati"""
        if self.lang == 'it':
            title = "📈 ANALISI ESPLORATIVA DEI DATI"
            first_rows = "[*] Prime righe del dataset:"
            info_msg = "[*] Informazioni sul dataset:"
            rows_msg = "    Righe: {}"
            cols_msg = "    Colonne: {}"
            memory_msg = "    Memoria: {:.2f} MB"
            types_msg = "[*] Tipi di dato:"
            stats_msg = "[*] Statistiche numeriche:"
            nulls_msg = "[*] Valori nulli:"
        else:
            title = "📈 EXPLORATORY DATA ANALYSIS"
            first_rows = "[*] First rows of dataset:"
            info_msg = "[*] Dataset information:"
            rows_msg = "    Rows: {}"
            cols_msg = "    Columns: {}"
            memory_msg = "    Memory: {:.2f} MB"
            types_msg = "[*] Data types:"
            stats_msg = "[*] Numerical statistics:"
            nulls_msg = "[*] Null values:"
        
        print("\n" + "="*60)
        print(title)
        print("="*60)
        
        print(f"\n{first_rows}")
        print(self.df.head())
        
        print(f"\n{info_msg}")
        print(rows_msg.format(len(self.df)))
        print(cols_msg.format(self.df.shape[1]))
        print(memory_msg.format(self.df.memory_usage(deep=True).sum() / 1024**2))
        
        print(f"\n{types_msg}")
        print(self.df.dtypes)
        
        print(f"\n{stats_msg}")
        print(self.df.describe())
        
        print(f"\n{nulls_msg}")
        null_counts = self.df.isnull().sum()
        print(null_counts[null_counts > 0])
    
    
    def clean_data(self):
        """Pulizia e preprocessing dei dati"""
        if self.lang == 'it':
            title = "🧹 PULIZIA E PREPROCESSING DEI DATI"
            removed_msg = "[*] Rimossi {} righe con prezzo mancante"
            temporal_msg = "[*] Estrazione features temporali..."
            airline_cleaning = "[*] Pulizia robusta colonna airline..."
            cleaned_airlines = "   ✅ Airline pulite: {} compagnie uniche"
            final_msg = "✅ Dataset pulito: {} righe, {} colonne"
        else:
            title = "🧹 DATA CLEANING AND PREPROCESSING"
            removed_msg = "[*] Removed {} rows with missing price"
            temporal_msg = "[*] Extracting temporal features..."
            airline_cleaning = "[*] Robust cleaning of airline column..."
            cleaned_airlines = "   ✅ Cleaned airlines: {} unique companies"
            final_msg = "✅ Cleaned dataset: {} rows, {} columns"
        
        print("\n" + "="*60)
        print(title)
        print("="*60)
        
        self.df_cleaned = self.df.copy()
        
        # Rimuovi righe con prezzo mancante
        if 'price' in self.df_cleaned.columns:
            initial_rows = len(self.df_cleaned)
            self.df_cleaned = self.df_cleaned.dropna(subset=['price'])
            removed = initial_rows - len(self.df_cleaned)
            if removed > 0:
                print(f"\n{removed_msg.format(removed)}")
        
        # Estrai features temporali se presente 'date'
        if 'date' in self.df_cleaned.columns:
            print(f"\n{temporal_msg}")
            self.df_cleaned['date'] = pd.to_datetime(self.df_cleaned['date'], errors='coerce')
            self.df_cleaned['journey_month'] = self.df_cleaned['date'].dt.month
            self.df_cleaned['journey_dayofweek'] = self.df_cleaned['date'].dt.dayofweek
            self.df_cleaned['journey_day'] = self.df_cleaned['date'].dt.day
        
        # Converti price a float se necessario
        if 'price' in self.df_cleaned.columns:
            self.df_cleaned['price'] = pd.to_numeric(self.df_cleaned['price'], errors='coerce')
            self.df_cleaned = self.df_cleaned.dropna(subset=['price'])
        

        # Normalizza stringhe (airline, from, to, class) con cleaning robusto
        for col in ['airline', 'from', 'to', 'class']:
            if col in self.df_cleaned.columns:
                if col == 'airline':
                    print(f"\n{airline_cleaning}")
                    self.df_cleaned[col] = self.df_cleaned[col].apply(_clean_airline_value)
                    # Filtro valori non validi
                    self.df_cleaned = self.df_cleaned[self.df_cleaned['airline'].str.lower() != 'unknown']
                    print(cleaned_airlines.format(self.df_cleaned['airline'].nunique()))
                else:
                    self.df_cleaned[col] = self.df_cleaned[col].astype(str).str.strip().str.title()
        
        print(final_msg.format(len(self.df_cleaned), self.df_cleaned.shape[1]))
        
        return self.df_cleaned
    
    
    def visualize_price_distribution(self):
        """Visualizza distribuzione prezzi"""
        if self.lang == 'it':
            creating_msg = "[*] Creating price distribution graphs..."
            no_price_col = "   [!] Column 'price' not found"
        else:
            creating_msg = "[*] Creating price distribution graphs..."
            no_price_col = "   [!] Column 'price' not found"
        
        print(f"\n{creating_msg}")
        
        if 'price' not in self.df_cleaned.columns:
            print(no_price_col)
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Istogramma
        axes[0, 0].hist(self.df_cleaned['price'], bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Price Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel(f'Price ({self.currency_symbol})')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(self.df_cleaned['price'])
        axes[0, 1].set_title('Price Box Plot', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel(f'Price ({self.currency_symbol})')
        
        # Log scale
        axes[1, 0].hist(np.log10(self.df_cleaned['price'] + 1), bins=50, color='coral', edgecolor='black')
        axes[1, 0].set_title('Price Distribution (Log Scale)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Log10(Price)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Statistiche testuali - SOLO IN INGLESE
        axes[1, 1].axis('off')
        stats_text = f"""
PRICE STATISTICS:

Mean: {format_currency(self.df_cleaned['price'].mean(), self.currency_symbol)}
Median: {format_currency(self.df_cleaned['price'].median(), self.currency_symbol)}
Min: {format_currency(self.df_cleaned['price'].min(), self.currency_symbol)}
Max: {format_currency(self.df_cleaned['price'].max(), self.currency_symbol)}
Std Dev: {format_currency(self.df_cleaned['price'].std(), self.currency_symbol)}
Skewness: {self.df_cleaned['price'].skew():.2f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('price_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print(f"    [OK] Saved: price_distribution_analysis.png")
        plt.close()
        
        # Apri il grafico automaticamente
        print(f"    [*] Opening graph...")
        open_file('price_distribution_analysis.png')
    
    
    def analyze_seasonality(self):
        """Analizza stagionalità prezzi per mese e giorno settimana"""
        if self.lang == 'it':
            title = "📅 ANALISI STAGIONALITA'"
            no_month_col = "[!] Colonna 'journey_month' non trovata. Analisi stagionalità non disponibile."
            no_price_col = "[!] Colonna 'price' non trovata."
            month_analysis = "[*] Analisi per mese..."
            best_month_msg = "[OK] Mese più economico: {} ({})"
            worst_month_msg = "[OK] Mese più costoso: {} ({})"
            diff_msg = "[OK] Differenza: {}"
            creating_graphs = "[*] Creazione grafici stagionalità..."
            saved_msg = "    [OK] Salvato: seasonality_analysis.png"
            opening_msg = "    [*] Apertura grafico..."
            months_names = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
        else:
            title = "📅 SEASONALITY ANALYSIS"
            no_month_col = "[!] Column 'journey_month' not found. Seasonality analysis not available."
            no_price_col = "[!] Column 'price' not found."
            month_analysis = "[*] Analysis by month..."
            best_month_msg = "[OK] Cheapest month: {} ({})"
            worst_month_msg = "[OK] Most expensive month: {} ({})"
            diff_msg = "[OK] Difference: {}"
            creating_graphs = "[*] Creating seasonality graphs..."
            saved_msg = "    [OK] Saved: seasonality_analysis.png"
            opening_msg = "    [*] Opening graph..."
            months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        print("\n" + "="*60)
        print(title)
        print("="*60)
        
        if 'journey_month' not in self.df_cleaned.columns:
            print(no_month_col)
            return None
        
        if 'price' not in self.df_cleaned.columns:
            print(no_price_col)
            return None
        
        # Analisi per mese
        print(f"\n{month_analysis}")
        monthly_stats = self.df_cleaned.groupby('journey_month')['price'].agg([
            ('prezzo_medio', 'mean'),
            ('prezzo_min', 'min'),
            ('prezzo_max', 'max'),
            ('num_voli', 'count'),
            ('std_dev', 'std')
        ]).round(2)
        
        monthly_stats.index = [months_names[i-1] for i in monthly_stats.index]
        print(monthly_stats)
        
        # Trova best/worst
        best_month = monthly_stats['prezzo_medio'].idxmin()
        worst_month = monthly_stats['prezzo_medio'].idxmax()
        best_price = monthly_stats['prezzo_medio'].min()
        worst_price = monthly_stats['prezzo_medio'].max()
        price_variation = worst_price - best_price
        
        print(f"\n{best_month_msg.format(best_month, format_currency(best_price, self.currency_symbol))}")
        print(f"{worst_month_msg.format(worst_month, format_currency(worst_price, self.currency_symbol))}")
        print(f"{diff_msg.format(format_currency(price_variation, self.currency_symbol))}")
        
        # Crea visualizzazione
        print(f"\n{creating_graphs}")
        
        # TESTO GRAFICI SEMPRE IN INGLESE
        chart1_title = f'Average Price by Month'
        chart1_xlabel = 'Month'
        chart1_ylabel = f'Average Price ({self.currency_symbol})'
        chart2_title = 'Number of Flights by Month'
        chart2_xlabel = 'Month'
        chart2_ylabel = 'Number of Flights'
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Prezzo medio per mese
        monthly_stats['prezzo_medio'].plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title(chart1_title, fontsize=12, fontweight='bold')
        axes[0].set_xlabel(chart1_xlabel)
        axes[0].set_ylabel(chart1_ylabel)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Numero voli per mese
        monthly_stats['num_voli'].plot(kind='bar', ax=axes[1], color='coral')
        axes[1].set_title(chart2_title, fontsize=12, fontweight='bold')
        axes[1].set_xlabel(chart2_xlabel)
        axes[1].set_ylabel(chart2_ylabel)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('seasonality_analysis.png', dpi=300, bbox_inches='tight')
        print(f"{saved_msg}")
        plt.close()
        
        # Apri il grafico automaticamente
        print(f"{opening_msg}")
        open_file('seasonality_analysis.png')
        
        self.seasonal_analysis = {
            'monthly_stats': monthly_stats,
            'best_month': (best_month, best_price),
            'worst_month': (worst_month, worst_price),
            'price_variation': price_variation
        }
        
        return self.seasonal_analysis
    
    
    def prepare_features(self):
        """Prepara features per il modello ML"""
        if self.lang == 'it':
            title = "⚙️ PREPARAZIONE FEATURES"
            features_ok = "[OK] Features preparate:"
            features_label = "    Features: {}"
            sample_size_label = "    Sample size: {}"
            features_cols_label = "    Features columns: {}\n"
        else:
            title = "⚙️ FEATURE PREPARATION"
            features_ok = "[OK] Features prepared:"
            features_label = "    Features: {}"
            sample_size_label = "    Sample size: {}"
            features_cols_label = "    Features columns: {}\n"
        
        print("\n" + "="*60)
        print(title)
        print("="*60)
        
        X, y = self.ml_models.prepare_features(self.df_cleaned)
        
        print(f"\n{features_ok}")
        print(features_label.format(len(self.ml_models.feature_columns)))
        print(sample_size_label.format(len(X)))
        print(features_cols_label.format(self.ml_models.feature_columns))
        
        return X, y
    
    
    def train_model(self, X, y):
        """Addestra il modello ML"""
        if self.lang == 'it':
            title = "🤖 TRAINING MODELLO ML"
            top_features = "[*] Top 10 Features Più Importanti:"
            saved_msg = "[OK] Salvato: feature_importance.png"
            opening_msg = "[*] Apertura grafico..."
            chart_xlabel = 'Importanza'
            chart_title = 'Feature Importance - Top 10'
        else:
            title = "🤖 ML MODEL TRAINING"
            top_features = "[*] Top 10 Most Important Features:"
            saved_msg = "[OK] Saved: feature_importance.png"
            opening_msg = "[*] Opening graph..."
            chart_xlabel = 'Importance'
            chart_title = 'Feature Importance - Top 10'
        
        print("\n" + "="*60)
        print(title)
        print("="*60)
        
        model, X_test, y_test, metrics = self.ml_models.train_model(X, y)
        
        # Feature importance
        print(f"\n{top_features}")
        importance = self.ml_models.get_feature_importance(top_n=10)
        print(importance.to_string(index=False))
        
        # Visualizza feature importance - SEMPRE IN INGLESE
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'], color='seagreen')
        plt.xlabel('Importance')
        plt.title('Feature Importance - Top 10', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\n{saved_msg}")
        plt.close()
        
        # Apri il grafico automaticamente
        print(f"{opening_msg}")
        open_file('feature_importance.png')
        
        return model, X_test, y_test
    
    
    def predict_price(self, flight_data):
        """
        Predice il prezzo per un volo.
        
        Args:
            flight_data (dict): Dati del volo (mese, giorno, durata, ecc.)
        
        Returns:
            float: Prezzo predetto
        """
        # Crea DataFrame dai dati
        pred_df = pd.DataFrame([flight_data])
        
        # Riempi le colonne mancanti con il dataset di training
        for col in self.ml_models.feature_columns:
            if col not in pred_df.columns:
                # Se la colonna è numerica, usa la media
                if pd.api.types.is_numeric_dtype(self.df_cleaned[col]):
                    pred_df[col] = self.df_cleaned[col].mean()
                else:
                    # Se è categorica/testuale, usa il valore più frequente (moda)
                    pred_df[col] = self.df_cleaned[col].mode()[0] if not self.df_cleaned[col].mode().empty else 'Unknown'
        
        # Seleziona solo le feature colonne
        pred_df = pred_df[self.ml_models.feature_columns]
        
        # Predici
        price = self.ml_models.predict(pred_df)
        
        return price
    
    
    def interactive_prediction(self):
        """Modalità interattiva per predizioni personalizzate"""
        
        # Se df_cleaned non è stato inizializzato, usare df
        if self.df_cleaned is None or self.df_cleaned.empty:
            if self.df is not None and not self.df.empty:
                self.df_cleaned = self.df.copy()
                # Normalizza stringhe
                for col in ['airline', 'from', 'to', 'class']:
                    if col in self.df_cleaned.columns:
                        self.df_cleaned[col] = self.df_cleaned[col].astype(str).str.strip().str.title()
            else:
                error_msg = "❌ Errore: Dataset non disponibile per la predizione" if self.lang == 'it' else "❌ Error: Dataset not available for prediction"
                print(error_msg)
                return None
        
        # Rileva la valuta se non è stata ancora rilevata
        if not hasattr(self, 'currency_symbol') or self.currency_symbol == '€':
            self.currency_symbol = self.detect_currency()
        
        # Messaggi multilingua
        if self.lang == 'it':
            title = "🎯 PREDIZIONE INTERATTIVA PREZZO VOLO - MODALITA' INTELLIGENTE"
            details_msg = "Inserisci i dettagli del volo per ottenere una predizione del prezzo:"
            cities_msg = "[*] Città disponibili: {}"
            departure_msg = "✈️ Città di partenza: "
            city_not_found = "❌ Città non trovata. Scegli da: {}"
            destinations_msg = "[*] Destinazioni disponibili da {}: {}"
            destination_msg = "✈️ Città di destinazione: "
            dest_not_found = "❌ Destinazione non trovata. Scegli da: {}"
            month_msg = "📅 Mese del viaggio (1-12): "
            invalid_month = "❌ Inserisci un numero tra 1 e 12"
            day_msg = "📅 Giorno del mese (1-{}): "
            invalid_day = "❌ Inserisci un numero tra 1 e {}"
            day_of_week_msg = "✅ Giorno della settimana: {}"
            invalid_date = "❌ Data non valida"
            analyzing_airlines = "[*] Analisi compagnie su questa rotta..."
            best_airline_selected = "✅ Migliore compagnia selezionata: {}"
            default_airline = "[*] Uso compagnia predefinita: {}"
            calculating = "🔮 CALCOLO PREDIZIONE IN CORSO..."
            predicted_price_msg = "💰 PREZZO PREDETTO: {}"
            flight_details = "📋 Dettagli del volo:"
            route_label = "  ✈️ Rotta: {} -> {}"
            date_label = "  📅 Data: {} {}"
            day_label = "  📅 Giorno: {}"
            duration_label = "  ⏱️ Durata: {}h {}m"
            airline_label = "  ✈️ Compagnia: {}"
            tip_title = "💡 CONSIGLIO: COMPAGNIA AEREA PIU' ECONOMICA"
            best_tip = "✅ Migliore: {} - Prezzo medio: {} ({} voli)"
            other_airlines = "Altre compagnie su questa rotta:"
            airline_item = "  • {}: {} ({} voli)"
            no_recommendations = "[!] Nessuna raccomandazione compagnia valida trovata."
            unable_to_generate = "[!] Impossibile generare raccomandazione compagnie: {}"
            another_prediction = "🔄 Vuoi fare un'altra predizione? (s/n): "
            cancelled_msg = "❌ Predizione annullata dall'utente"
            error_msg = "❌ Errore durante la predizione: {}"
            months = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
            days = ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì', 'Sabato', 'Domenica']
            duration_calc = "✅ Durata media calcolata: {}h {}m"
            note_leap = "   [*] Nota: 2022 non è un anno bisestile"
            month_days_info = "   [*] {} 2022 ha {} giorni"
        else:
            title = "🎯 INTERACTIVE FLIGHT PRICE PREDICTION - SMART MODE"
            details_msg = "Enter flight details to get a price prediction:"
            cities_msg = "[*] Available cities: {}"
            departure_msg = "✈️ Departure city: "
            city_not_found = "❌ City not found. Choose from: {}"
            destinations_msg = "[*] Available destinations from {}: {}"
            destination_msg = "✈️ Destination city: "
            dest_not_found = "❌ Destination not found. Choose from: {}"
            month_msg = "📅 Journey month (1-12): "
            invalid_month = "❌ Enter a number between 1 and 12"
            day_msg = "📅 Day of month (1-{}): "
            invalid_day = "❌ Enter a number between 1 and {}"
            day_of_week_msg = "✅ Day of week: {}"
            invalid_date = "❌ Invalid date"
            analyzing_airlines = "[*] Analyzing airlines on this route..."
            best_airline_selected = "✅ Best airline selected: {}"
            default_airline = "[*] Using default airline: {}"
            calculating = "🔮 CALCULATING PREDICTION..."
            predicted_price_msg = "💰 PREDICTED PRICE: {}"
            flight_details = "📋 Flight details:"
            route_label = "  ✈️ Route: {} -> {}"
            date_label = "  📅 Date: {} {}"
            day_label = "  📅 Day: {}"
            duration_label = "  ⏱️ Duration: {}h {}m"
            airline_label = "  ✈️ Airline: {}"
            tip_title = "💡 TIP: CHEAPEST AIRLINE ON THIS ROUTE"
            best_tip = "✅ Best: {} - Avg price: {} ({} flights)"
            other_airlines = "Other airlines on this route:"
            airline_item = "  • {}: {} ({} flights)"
            no_recommendations = "[!] No valid airline recommendations found."
            unable_to_generate = "[!] Unable to generate airline recommendation: {}"
            another_prediction = "🔄 Do you want another prediction? (y/n): "
            cancelled_msg = "❌ Prediction cancelled by user"
            error_msg = "❌ Error during prediction: {}"
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            duration_calc = "✅ Calculated average duration: {}h {}m"
            note_leap = "   [*] Note: 2022 is not a leap year"
            month_days_info = "   [*] {} 2022 has {} days"
        
        print("\n" + "="*70)
        print(title)
        print("="*70)
        print(f"\n{details_msg}\n")
        

        flight_data = {}

        try:
            from datetime import datetime
            # Get unique cities
            cities = sorted(self.df_cleaned['from'].unique()) if 'from' in self.df_cleaned.columns else []
            print(cities_msg.format(', '.join(cities)) + "\n")

            # Departure city
            if cities:
                while True:
                    from_city = input(departure_msg).strip().title()
                    if from_city in cities:
                        break
                    print(city_not_found.format(', '.join(cities)))

                # Destination city
                destination_cities = sorted(self.df_cleaned[self.df_cleaned['from'] == from_city]['to'].unique()) if 'to' in self.df_cleaned.columns else []
                print(f"\n{destinations_msg.format(from_city, ', '.join(destination_cities))}\n")
                while True:
                    to_city = input(destination_msg).strip().title()
                    if to_city in destination_cities:
                        break
                    print(dest_not_found.format(', '.join(destination_cities)))

            # Month
            while True:
                month = input(f"\n{month_msg}").strip()
                if month.isdigit() and 1 <= int(month) <= 12:
                    flight_data['journey_month'] = int(month)
                    break
                print(invalid_month)

            # Max days in month
            import calendar
            max_days_in_month = calendar.monthrange(2022, flight_data['journey_month'])[1]
            print(month_days_info.format(months[flight_data['journey_month']-1], max_days_in_month))
            if flight_data['journey_month'] == 2:
                print(note_leap)

            # Day
            while True:
                day = input(day_msg.format(max_days_in_month)).strip()
                if day.isdigit() and 1 <= int(day) <= max_days_in_month:
                    flight_data['journey_day'] = int(day)
                    break
                print(invalid_day.format(max_days_in_month))

            # Day of week
            try:
                date_obj = datetime(2022, flight_data['journey_month'], flight_data['journey_day'])
                flight_data['journey_dayofweek'] = date_obj.weekday()
                print(day_of_week_msg.format(days[flight_data['journey_dayofweek']]))
            except ValueError:
                print(invalid_date)
                return None

            # Calculate average duration for route AND find best airline automatically
            route_flights = None
            best_airline = None
            if 'from' in self.df_cleaned.columns and 'to' in self.df_cleaned.columns:
                route_flights = self.df_cleaned[(self.df_cleaned['from'] == from_city) & (self.df_cleaned['to'] == to_city)]
                if len(route_flights) > 0:
                    # Calculate duration
                    if 'duration' in route_flights.columns:
                        avg_duration = route_flights['duration'].mean()
                        flight_data['duration'] = avg_duration
                        hours = int(avg_duration)
                        minutes = int((avg_duration - hours) * 60)
                        print(f"\n{duration_calc.format(hours, minutes)}")
                    
                    # Find best airline automatically
                    if 'airline' in route_flights.columns and 'price' in route_flights.columns:
                        print(f"\n{analyzing_airlines}")
                        route_flights_clean = route_flights.copy()
                        route_flights_clean['airline'] = route_flights_clean['airline'].apply(_clean_airline_value)
                        route_flights_clean = route_flights_clean[route_flights_clean['airline'].str.lower() != 'unknown']
                        route_flights_clean = route_flights_clean[pd.to_numeric(route_flights_clean['price'], errors='coerce').notnull()]
                        route_flights_clean['price'] = pd.to_numeric(route_flights_clean['price'], errors='coerce')
                        
                        if len(route_flights_clean) > 0:
                            route_airlines = route_flights_clean.groupby('airline')['price'].agg(['mean', 'count']).sort_values('mean')
                            if len(route_airlines) > 0:
                                best_airline = route_airlines.index[0]
                                flight_data['airline'] = best_airline
                                print(best_airline_selected.format(best_airline))
            
            # Fallback if no airline found
            if 'airline' not in flight_data or not flight_data.get('airline'):
                airlines_available = sorted(self.df_cleaned['airline'].unique()) if 'airline' in self.df_cleaned.columns else []
                if airlines_available:
                    flight_data['airline'] = airlines_available[0]
                    print(default_airline.format(flight_data['airline']))

            # Prediction
            print("\n" + "="*70)
            print(calculating)
            print("="*70)

            predicted_price = self.predict_price(flight_data)

            print("\n" + "🎉 "*20)
            print(f"\n{predicted_price_msg.format(format_currency(predicted_price, self.currency_symbol))}")
            print("\n" + "🎉 "*20)

            # Show flight details
            print(f"\n{flight_details}")
            if 'from' in flight_data and 'to' in flight_data:
                print(route_label.format(flight_data.get('from', 'N/A'), flight_data.get('to', 'N/A')))
            print(date_label.format(flight_data['journey_day'], months[flight_data['journey_month']-1]))
            print(day_label.format(days[flight_data['journey_dayofweek']]))
            if 'duration' in flight_data:
                hours = int(flight_data['duration'])
                minutes = int((flight_data['duration'] - hours) * 60)
                print(duration_label.format(hours, minutes))
            print(airline_label.format(flight_data['airline']))

            # Robust airline recommendation logic
            if route_flights is not None and len(route_flights) > 0 and 'airline' in route_flights.columns and 'price' in route_flights.columns:
                print("\n" + "="*70)
                print(tip_title)
                print("="*70)

                try:
                    # Clean and deduplicate airline names (AGAIN, for safety)
                    route_flights = route_flights.copy()
                    route_flights['airline'] = route_flights['airline'].apply(_clean_airline_value)
                    route_flights = route_flights[route_flights['airline'].str.lower() != 'unknown']
                    # Ensure price is numeric and filter out non-numeric
                    route_flights = route_flights[pd.to_numeric(route_flights['price'], errors='coerce').notnull()]
                    route_flights['price'] = pd.to_numeric(route_flights['price'], errors='coerce')
                    route_airlines = route_flights.groupby('airline')['price'].agg(['mean', 'count']).sort_values('mean')

                    shown = 0
                    for idx, (airline, row) in enumerate(route_airlines.iterrows()):
                        airline_name = _safe_label(airline, 40)
                        price = row['mean']
                        flights = int(row['count'])
                        if len(airline_name) > 0 and airline_name.lower() != 'unknown' and flights > 0:
                            if shown == 0:
                                print(f"\n{best_tip.format(airline_name, format_currency(price, self.currency_symbol), flights)}")
                            else:
                                if shown == 1:
                                    print(f"\n{other_airlines}")
                                print(airline_item.format(airline_name, format_currency(price, self.currency_symbol), flights))
                            shown += 1
                        if shown > 5:
                            break
                    if shown == 0:
                        print(no_recommendations)
                except Exception as e:
                    print(unable_to_generate.format(e))

            # Ask for another prediction
            print("\n" + "="*70)
            another = input(f"\n{another_prediction}").strip().lower()
            if another == 'y' or another == 's' or another == 'si' or another == 'sì':
                self.interactive_prediction()

            return predicted_price

        except KeyboardInterrupt:
            print(f"\n\n{cancelled_msg}")
            return None
        except Exception as e:
            print(f"\n{error_msg.format(e)}")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"\n❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def run_complete_analysis(self):
        """Esegue l'intera pipeline di analisi"""
        if self.lang == 'it':
            title = "✈️ SISTEMA DI PREDIZIONE PREZZI VOLI - ANALISI COMPLETA"
            completed = "✅ ANALISI COMPLETATA CON SUCCESSO!"
            files_label = "\nFile generati:"
            file1_desc = "  📊 price_distribution_analysis.png - Analisi distribuzione prezzi"
            file2_desc = "  📈 seasonality_analysis.png - Analisi stagionalità"
            file3_desc = "  🎯 feature_importance.png - Importanza delle features"
        else:
            title = "✈️ FLIGHT PRICE PREDICTION SYSTEM - COMPLETE ANALYSIS"
            completed = "✅ ANALYSIS COMPLETED SUCCESSFULLY!"
            files_label = "\nGenerated files:"
            file1_desc = "  📊 price_distribution_analysis.png - Price distribution analysis"
            file2_desc = "  📈 seasonality_analysis.png - Seasonality analysis"
            file3_desc = "  🎯 feature_importance.png - Feature importance"
        
        print("\n" + "="*80)
        print(title)
        print("="*80)
        
        # 1. Carica dati
        self.load_data()
        
        # 2. Esplora dati
        self.explore_data()
        
        # 3. Pulisci dati
        self.clean_data()
        
        # 4. Visualizza distribuzione prezzi
        self.visualize_price_distribution()
        
        # 5. Analisi stagionalità
        self.analyze_seasonality()
        
        # 6. Prepara features
        X, y = self.prepare_features()
        
        # 7. Addestra modello
        self.train_model(X, y)
        
        print("\n" + "="*80)
        print(completed)
        print("="*80)
        print(files_label)
        print(file1_desc)
        print(file2_desc)
        print(file3_desc)
        print("\n🎉 Il sistema è pronto per fare predizioni!")
