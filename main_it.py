#!/usr/bin/env python3
"""
MAIN ENTRY POINT- VERSIONE ITALIANA
=================================================

Sistema di Predizione Prezzi Voli - Applicazione principale in lingua italiana.

Utilizzo:
    python main_it.py

Funzionalità:
    • Caricamento dataset da 3 file (economy.csv + business.csv + Clean_Dataset.csv)
    • OPPURE caricamento di un file CSV generico
    • Analisi esplorativa completa dei dati
    • Analisi della stagionalità
    • Predizione interattiva del prezzo volo con funzionalità intelligenti:
      - Calcolo automatico del giorno della settimana
      - Ricerca durata dalla cronologia della rotta
      - Raccomandazioni compagnie aeree
"""

import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from flight_price_predictor import FlightPricePredictor, load_generic_dataset
from flight_price_predictor.data_loader import carica_o_crea_unificato
from i18n.it import (
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
    """Punto di ingresso principale dell'applicazione - Interfaccia italiana"""
    
    # Mostra menu
    print(MENU_PRINCIPALE)
    
    try:
        scelta = input("Scelta (1/2/3): ").strip()
        
        if scelta == "1":
            # Carica dataset da 3 file (economy + business + Clean_Dataset)
            print(CARICAMENTO_3_FILE)
            
            # Verifica esistenza file
            if not os.path.exists("economy.csv"):
                print(ERRORE_CARICAMENTO_3_FILE.format(error="economy.csv non trovato"))
                print(ASSICURATI_3_FILE)
                return
            
            if not os.path.exists("business.csv"):
                print(ERRORE_CARICAMENTO_3_FILE.format(error="business.csv non trovato"))
                print(ASSICURATI_3_FILE)
                return
            
            if not os.path.exists("Clean_Dataset.csv"):
                print(ERRORE_CARICAMENTO_3_FILE.format(error="Clean_Dataset.csv non trovato"))
                print(ASSICURATI_3_FILE)
                return
            
            try:
                # Crea dataset unificato da 3 file (o carica se esiste già)
                df_unified = carica_o_crea_unificato()
                
                # Crea predittore con dati unificati (lingua italiana)
                predictor = FlightPricePredictor(None, lang='it')
                predictor.df = df_unified
                
                rows, cols = df_unified.shape
                print(f"✅ Dataset unificato caricato: {rows} righe, {cols} colonne")
                print(DATASET_PRONTO)
            except Exception as e:
                print(ERRORE_CARICAMENTO_3_FILE.format(error=str(e)))
                return
        
        elif scelta == "2":
            # Carica file CSV generico
            print(CARICAMENTO_FILE_GENERICO)
            file_path = input(INSERISCI_PERCORSO).strip()
            
            if not os.path.exists(file_path):
                print(FILE_NON_TROVATO.format(path=file_path))
                return
            
            print(CARICAMENTO_GENERICO)
            try:
                # Carica il dataset generico
                df = load_generic_dataset(file_path)
                
                # Crea predittore con il dataset caricato (lingua italiana)
                predictor = FlightPricePredictor(None, lang='it')
                predictor.df = df
                
                rows, cols = df.shape
                print(DATASET_CARICATO.format(rows=rows, cols=cols))
                print(DATASET_PRONTO)
            except Exception as e:
                print(ERRORE_CARICAMENTO_FILE.format(error=str(e)))
                return
        
        elif scelta == "3":
            print(OPERAZIONE_ANNULLATA)
            return
        
        else:
            print(MENU_SCELTA_INVALIDA)
            return
        
        # Esegui analisi completa
        if predictor.df is not None:
            predictor.explore_data()
        else:
            print("❌ Errore: Dataset non caricato")
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
        
        # Predizione interattiva
        print("\n" + "="*80)
        predictor.interactive_prediction()
        
        print("\n" + PROGRAMMA_COMPLETATO)
    
    except KeyboardInterrupt:
        print("\n" + OPERAZIONE_ANNULLATA)
    except Exception as e:
        print(f"\n❌ Errore imprevisto: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
