"""
Texts and messages in ENGLISH
==============================
Centralizes all user interface messages in English.
"""

MENU_PRINCIPALE = """
================================================================================
✈️ FLIGHT PRICE PREDICTOR - FLIGHT PRICE PREDICTION SYSTEM
================================================================================

[*] What type of dataset do you have?

  1. Three separate files (economy.csv + business.csv + Clean_Dataset.csv)
     → If you have flights divided by travel class

  2. One single generic file
     → If you have a single CSV with all flights (any destination/country)

  3. Cancel

"""

CARICAMENTO_DATASET = "[*] Loading dataset..."
DATASET_CARICATO = "✅ Dataset loaded: {rows} rows, {cols} columns"
COLONNE_DISPONIBILI = "Available columns: {cols}"

MENU_SCELTA_INVALIDA = "❌ Invalid choice. Enter 1, 2 or 3"

CARICAMENTO_3_FILE = "[*] Loading dataset from 3 files (economy + business + Clean_Dataset)..."
ERRORE_CARICAMENTO_3_FILE = "❌ Error loading the 3 files: {error}"
ASSICURATI_3_FILE = "   Make sure these files exist: economy.csv, business.csv, Clean_Dataset.csv"

CARICAMENTO_FILE_GENERICO = "[*] Which CSV file do you want to load?"
INSERISCI_PERCORSO = "Enter the file path (e.g., C:\\flights_tokyo.csv): "
FILE_NON_TROVATO = "❌ File not found: {path}"
CARICAMENTO_GENERICO = "[*] Loading generic dataset..."
ERRORE_CARICAMENTO_FILE = "❌ Error loading the file: {error}"

OPERAZIONE_ANNULLATA = "❌ Operation cancelled"
DATASET_PRONTO = "[OK] Dataset ready!\n"

PROGRAMMA_COMPLETATO = "✅ Program completed!"

# Analysis messages
ANALISI_ESPLORATIVA = "📈 EXPLORATORY DATA ANALYSIS"
ANALISI_STAGIONALITA = "📅 SEASONALITY ANALYSIS"
PREPARAZIONE_FEATURES = "⚙️ FEATURE PREPARATION"
TRAINING_MODELLO = "🤖 MODEL TRAINING"
ANALISI_COMPLETATA = "✅ ANALYSIS COMPLETED SUCCESSFULLY!"

FILE_GENERATI = "Generated files:"
PRICE_DISTRIBUTION = "  📊 price_distribution_analysis.png - Price distribution analysis"
SEASONALITY_ANALYSIS = "  📈 seasonality_analysis.png - Seasonality analysis"
FEATURE_IMPORTANCE = "  🎯 feature_importance.png - Feature importance"
SISTEMA_PRONTO = "🎉 The system is ready to make predictions!"

# Interactive predictions
PREDIZIONE_INTERATTIVA = "🎯 INTERACTIVE FLIGHT PRICE PREDICTION - SMART MODE"
INSERISCI_DETTAGLI = "Enter flight details to get a price prediction:"

CITTA_DISPONIBILI = "[*] Available cities: {cities}"
CITTA_PARTENZA = "✈️ Departure city: "
CITTA_NON_TROVATA = "❌ City not found. Choose from: {cities}"

DESTINAZIONI_DISPONIBILI = "[*] Available destinations from {city}: {destinations}"
CITTA_DESTINAZIONE = "✈️ Destination city: "
DESTINAZIONE_NON_TROVATA = "❌ Destination not found. Choose from: {destinations}"

MESE_VIAGGIO = "📅 Journey month (1-12): "
NUMERO_INVALIDO_MESE = "❌ Enter a number between 1 and 12"

GIORNO_MESE = "📅 Day of month (1-{max}): "
NUMERO_INVALIDO_GIORNO = "❌ Enter a number between 1 and {max}"

GIORNI_MESE_INFO = "   [*] {month} 2022 has {days} days"
ANNO_NON_BISESTILE = "   [*] Note: 2022 is not a leap year"

GIORNO_SETTIMANA = "✅ Day of week: {day}"
DATA_NON_VALIDA = "❌ Invalid date"

DURATA_MEDIA = "✅ Average duration calculated: {hours}h {minutes}m"

CALCOLO_PREDIZIONE = "🔮 CALCULATING PREDICTION..."

PREZZO_PREDETTO = "💰 PREDICTED PRICE: {price}"

DETTAGLI_VOLO = "📋 Flight details:"
ROTTA = "  ✈️ Route: {from} -> {to}"
DATA = "  📅 Date: {day} {month}"
GIORNO = "  📅 Day: {day}"
DURATA = "  ⏱️ Duration: {hours}h {minutes}m"

CONSIGLIO_COMPAGNIA = "💡 TIP: CHEAPEST AIRLINE"
COMPAGNIA_ECONOMICA = "✅ {airline} - Average price: {price} ({count} flights)"
ALTRE_COMPAGNIE = "Other airlines on this route:"
COMPAGNIA_ITEM = "  • {airline}: {price} ({count} flights)"

VUOI_ALTRA_PREDIZIONE = "🔄 Do you want another prediction? (y/n): "
PREDIZIONE_ANNULLATA = "❌ Prediction cancelled by user"
ERRORE_PREDIZIONE = "❌ Error during prediction: {error}"

# Weekdays and months names
GIORNI_SETTIMANA = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MESI = ['January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December']
MESI_BREVI = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
