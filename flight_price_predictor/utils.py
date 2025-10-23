"""
Funzioni ausiliarie per il Flight Price Prediction System
========================================================
Contiene funzioni di utilità generiche usate dai moduli principali.
"""

import hashlib
import pandas as pd
import numpy as np
import os


def parse_generic_duration(s):
    """
    Parse intelligente della durata del volo da vari formati.
    
    Supporta:
    - Numeri interi/float: 2.17, 120 (minuti)
    - String numeriche: "1.75", "120"
    - Formato ore/minuti: "02h 10m", "2 hours 10 minutes"
    
    Args:
        s: Valore di durata in vario formato
    
    Returns:
        float: Durata in ore decimali (es: 2.17)
    """
    if pd.isna(s):
        return np.nan
    
    # Se è già un numero, ritorna direttamente
    if isinstance(s, (int, float)):
        try:
            return float(s)
        except:
            return np.nan
    
    s_str = str(s).strip()
    
    # Prova a convertire come stringa numerica
    try:
        return float(s_str)
    except ValueError:
        pass
    
    # Parse formato "02h 10m" o "2h 10m" o "2 hours 10 minutes"
    h = 0.0
    m = 0.0
    
    for sep in ['h', 'hour', 'ore']:
        if sep in s_str.lower():
            h_part = s_str.lower().split(sep)[0].strip()
            try:
                h = float(h_part)
            except:
                h = 0.0
            break
    
    for sep in ['m', 'min', 'minute', 'minuti']:
        if sep in s_str.lower():
            after_h = s_str.lower().split('h' if 'h' in s_str.lower() else 'hour' if 'hour' in s_str.lower() else 'ore')[1] if any(x in s_str.lower() for x in ['h', 'hour', 'ore']) else s_str.lower()
            m_part = after_h.replace(sep, '').strip()
            try:
                m = float(m_part)
            except:
                m = 0.0
            break
    
    return h + m/60.0 if h + m/60.0 > 0 else np.nan


def file_hash(paths):
    """
    Calcola hash MD5 di uno o più file per verificare modifiche.
    
    Args:
        paths (list): Lista di percorsi file
    
    Returns:
        str: Hash MD5 concatenato
    """
    m = hashlib.md5()
    for p in sorted(paths):
        with open(p, 'rb') as f:
            m.update(f.read())
    return m.hexdigest()


def find_column(df, candidates):
    """
    Trova una colonna in un DataFrame con mapping case-insensitive.
    
    Args:
        df (pd.DataFrame): DataFrame da cercare
        candidates (list): Lista di nomi candidati da cercare
    
    Returns:
        str or None: Nome della colonna trovata, None se non trovata
    """
    cols_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def normalize_string_columns(df, columns):
    """
    Normalizza colonne stringa a minuscolo e rimuove spazi.
    
    Args:
        df (pd.DataFrame): DataFrame da modificare
        columns (list): Nomi colonne da normalizzare
    
    Returns:
        pd.DataFrame: DataFrame con colonne normalizzate
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[f'{col}_norm'] = df_copy[col].astype(str).str.lower().str.strip()
    return df_copy


def clean_price_value(price_str):
    """
    Pulisce un valore di prezzo, rimuovendo simboli valuta.
    
    Args:
        price_str (str): Prezzo come stringa (es: "$1,234.56")
    
    Returns:
        float: Prezzo come float
    """
    if pd.isna(price_str):
        return np.nan
    
    # Converti a stringa
    s = str(price_str)
    
    # Sostituisci virgola con punto (per locali che usano virgola come decimale)
    s = s.replace(',', '.')
    
    # Rimuovi tutti i caratteri non numerici tranne il punto
    s = ''.join(c for c in s if c.isdigit() or c == '.')
    
    try:
        return float(s)
    except:
        return np.nan


def ensure_output_directory(path='output'):
    """
    Crea directory di output se non esiste.
    
    Args:
        path (str): Percorso della directory
    
    Returns:
        str: Percorso della directory (creata se necessaria)
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def format_currency(value, symbol='€', decimals=2):
    """
    Formatta un numero come valuta.
    
    Args:
        value (float): Valore da formattare
        symbol (str): Simbolo valuta
        decimals (int): Decimali da mostrare
    
    Returns:
        str: Valore formattato (es: "€1,234.56")
    """
    if pd.isna(value):
        return "N/A"
    return f"{symbol}{value:,.{decimals}f}"
