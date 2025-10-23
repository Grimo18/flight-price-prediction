"""
Flight Price Prediction System
================================
Un sistema completo di machine learning per predire prezzi di voli,
analizzare stagionalità e fornire insights intelligenti.

Moduli disponibili:
- core: Classe principale FlightPricePredictor
- models: Training e valutazione modelli ML
- data_loader: Caricamento e preparazione dati
- utils: Funzioni ausiliarie generiche

Utilizzo rapido:
    from flight_price_predictor import FlightPricePredictor, load_generic_dataset
    
    # Carica dataset generico (qualsiasi meta)
    df = load_generic_dataset('voli_tokyo.csv')
    
    # Crea predittore e analizza
    predictor = FlightPricePredictor('voli_tokyo.csv')
    predictor.run_complete_analysis()
    
    # Predizioni interattive
    predictor.interactive_prediction()

Utilizzo avanzato - 3 file separati (economy + business):
    from flight_price_predictor import carica_o_crea_unificato, FlightPricePredictor
    
    # Unifica automaticamente e crea Unified_Flights.csv
    df = carica_o_crea_unificato()
    
    # Analizza
    predictor = FlightPricePredictor('Unified_Flights.csv')
    predictor.run_complete_analysis()

Version: 2.0.0 - Struttura modulare e multilingua
"""

from .core import FlightPricePredictor
from .models import MLModels
from .data_loader import load_generic_dataset, genera_file_unificato, carica_o_crea_unificato
from .utils import parse_generic_duration, file_hash, format_currency

__version__ = "2.0.0"
__author__ = "Andrea Grimaldi"
__all__ = [
    'FlightPricePredictor',
    'MLModels',
    'load_generic_dataset',
    'genera_file_unificato',
    'carica_o_crea_unificato',
    'parse_generic_duration',
    'file_hash',
    'format_currency',
]
