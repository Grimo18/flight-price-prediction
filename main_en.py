
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAIN ENTRY POINT - ENGLISH VERSION
===================================

Flight Price Prediction System - Main application in English language.

Usage:
    python main_en.py

Features:
    • Load 3-file dataset (economy.csv + business.csv + Clean_Dataset.csv)
    • OR load any generic flight CSV file
    • Full exploratory data analysis
    • Seasonality analysis
    • Interactive flight price prediction with smart features:
      - Automatic day-of-week calculation
      - Duration lookup from route history
      - Airline recommendations
"""

import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from flight_price_predictor import FlightPricePredictor, load_generic_dataset
from flight_price_predictor.data_loader import carica_o_crea_unificato
from i18n.en import (
    MENU_PRINCIPALE, CARICAMENTO_DATASET, DATASET_CARICATO,
    MENU_SCELTA_INVALIDA, CARICAMENTO_3_FILE, ERRORE_CARICAMENTO_3_FILE,
    ASSICURATI_3_FILE, CARICAMENTO_FILE_GENERICO, INSERISCI_PERCORSO,
    FILE_NON_TROVATO, CARICAMENTO_GENERICO, ERRORE_CARICAMENTO_FILE,
    OPERAZIONE_ANNULLATA, DATASET_PRONTO, PROGRAMMA_COMPLETATO,
    ANALISI_ESPLORATIVA, ANALISI_STAGIONALITA, PREPARAZIONE_FEATURES,
    TRAINING_MODELLO, ANALISI_COMPLETATA, FILE_GENERATI,
    PRICE_DISTRIBUTION, SEASONALITY_ANALYSIS, FEATURE_IMPORTANCE,
    SISTEMA_PRONTO
)


def main():
    """Main application entry point - English interface"""
    
    # Show menu
    print(MENU_PRINCIPALE)
    
    try:
        choice = input("Choice (1/2/3): ").strip()
        
        if choice == "1":
            # Load 3-file format (economy + business + Clean_Dataset)
            print(CARICAMENTO_3_FILE)
            
            # Check if files exist
            if not os.path.exists("economy.csv"):
                print(ERRORE_CARICAMENTO_3_FILE.format(error="economy.csv not found"))
                print(ASSICURATI_3_FILE)
                return
            
            if not os.path.exists("business.csv"):
                print(ERRORE_CARICAMENTO_3_FILE.format(error="business.csv not found"))
                print(ASSICURATI_3_FILE)
                return
            
            if not os.path.exists("Clean_Dataset.csv"):
                print(ERRORE_CARICAMENTO_3_FILE.format(error="Clean_Dataset.csv not found"))
                print(ASSICURATI_3_FILE)
                return
            
            try:
                # Create unified dataset from 3 files (or load if already exists)
                df_unified = carica_o_crea_unificato()
                
                # Create predictor with unified data (English language)
                predictor = FlightPricePredictor(None, lang='en')
                predictor.df = df_unified
                
                # Detect and ask for currency if necessary
                detected, has_symbol = predictor.detect_currency()
                if detected is None or not has_symbol:
                    print("\n[?] Currency not auto-detected.")
                    print("Choose currency:\n  1. € (Euro)\n  2. $ (Dollar)\n  3. ₹ (Indian Rupee)\n  4. £ (Pound)")
                    choice_input = input("Choice (1/2/3/4): ").strip()
                    currency_map = {'1': '€', '2': '$', '3': '₹', '4': '£'}
                    predictor.currency_symbol = currency_map.get(choice_input, '€')
                else:
                    predictor.currency_symbol = detected
                print(f"💰 Currency selected: {predictor.currency_symbol}")
                
                rows, cols = df_unified.shape
                print(f"✅ Unified dataset loaded: {rows} rows, {cols} columns")
                print(DATASET_PRONTO)
            except Exception as e:
                print(ERRORE_CARICAMENTO_3_FILE.format(error=str(e)))
                return
        
        elif choice == "2":
            # Load generic CSV file
            print(CARICAMENTO_FILE_GENERICO)
            file_path = input(INSERISCI_PERCORSO).strip()
            
            if not os.path.exists(file_path):
                print(FILE_NON_TROVATO.format(path=file_path))
                return
            
            print(CARICAMENTO_GENERICO)
            try:
                # Load the generic dataset
                df = load_generic_dataset(file_path)
                
                # Create predictor with the loaded dataset (English language)
                predictor = FlightPricePredictor(None, lang='en')
                predictor.df = df
                
                # Detect and ask for currency if necessary
                detected, has_symbol = predictor.detect_currency()
                if detected is None or not has_symbol:
                    print("\n[?] Currency not auto-detected.")
                    print("Choose currency:\n  1. € (Euro)\n  2. $ (Dollar)\n  3. ₹ (Indian Rupee)\n  4. £ (Pound)")
                    choice_input = input("Choice (1/2/3/4): ").strip()
                    currency_map = {'1': '€', '2': '$', '3': '₹', '4': '£'}
                    predictor.currency_symbol = currency_map.get(choice_input, '€')
                else:
                    predictor.currency_symbol = detected
                print(f"💰 Currency selected: {predictor.currency_symbol}")
                
                rows, cols = df.shape
                print(DATASET_CARICATO.format(rows=rows, cols=cols))
                print(DATASET_PRONTO)
            except Exception as e:
                print(ERRORE_CARICAMENTO_FILE.format(error=str(e)))
                return
        
        elif choice == "3":
            print(OPERAZIONE_ANNULLATA)
            return
        
        else:
            print(MENU_SCELTA_INVALIDA)
            return
        
        # Run complete analysis
        if predictor.df is not None:
            predictor.explore_data()
        else:
            print("❌ Error: Dataset not loaded")
            return
        
        predictor.clean_data()
        predictor.analyze_seasonality()
        
        X, y = predictor.prepare_features()
        
        predictor.train_model(X, y)
        
        print("\n" + ANALISI_COMPLETATA)
        print(FILE_GENERATI)
        print(PRICE_DISTRIBUTION)
        print(SEASONALITY_ANALYSIS)
        print(FEATURE_IMPORTANCE)
        print(SISTEMA_PRONTO)
        
        # Interactive prediction
        print("\n" + "="*80)
        predictor.interactive_prediction()
        
        print("\n" + PROGRAMMA_COMPLETATO)
    
    except KeyboardInterrupt:
        print("\n" + OPERAZIONE_ANNULLATA)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
