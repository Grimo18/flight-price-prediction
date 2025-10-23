"""
Data Loader Module
==================
Gestisce il caricamento e la preparazione dei dati da vari formati.
Supporta sia il formato unificato (3 file) che dataset generici.
"""

import pandas as pd
import numpy as np
import os
import hashlib
from .utils import parse_generic_duration, file_hash, find_column, normalize_string_columns, clean_price_value


def load_generic_dataset(dataset_path):
    """
    Carica un dataset generico di voli (funziona con qualsiasi meta/dataset).
    Rileva automaticamente le colonne e le adatta intelligentemente.
    Supporta dataset unici (non richiede economy.csv + business.csv separati).
    
    Args:
        dataset_path (str): Percorso del file CSV generico
    
    Returns:
        pd.DataFrame: Dataset pulito e normalizzato
    """
    print(f"\n[*] Caricamento dataset generico: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    # Rimuove colonne 'unnamed' create dal salvataggio con index
    df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]
    
    print(f"    Colonne rilevate: {list(df.columns)}")
    print(f"    Righe: {len(df)}")
    
    # Mapping intelligente delle colonne (case-insensitive)
    col_mapping = {
        'price': ['price', 'fare', 'cost', 'prezzo', 'tariff', 'ticket_price'],
        'airline': ['airline', 'compagnia', 'carrier', 'airline_name', 'company'],
        'from': ['from', 'origin', 'source', 'departure_city', 'departure', 'from_city', 'source_city'],
        'to': ['to', 'destination', 'dest', 'arrival_city', 'to_city', 'destination_city'],
        'duration': ['duration', 'flight_duration', 'durata', 'tempo_volo', 'time_taken', 'flight_time'],
        'date': ['date', 'departure_date', 'data', 'data_partenza', 'journey_date'],
        'class': ['class', 'cabin', 'classe', 'cabin_class', 'ticket_class'],
        'stops': ['stops', 'stop', 'num_stops', 'total_stops', 'scali', 'layovers']
    }
    
    # Rileva colonne disponibili
    detected_cols = {}
    for standard_name, candidates in col_mapping.items():
        col = find_column(df, candidates)
        if col:
            detected_cols[standard_name] = col
            print(f"    [OK] '{standard_name}' rilevato come '{col}'")
        else:
            print(f"    [!] '{standard_name}' NON trovato")
    
    # Copia il dataframe e rinomina le colonne rilevate
    df_clean = df.copy()
    rename_dict = {v: k for k, v in detected_cols.items() if v != k}
    df_clean = df_clean.rename(columns=rename_dict)
    
    # Applica parsing della durata se presente
    if 'duration' in df_clean.columns:
        print("    [*] Parsing duration...")
        df_clean['duration'] = df_clean['duration'].apply(parse_generic_duration)
    
    # Normalizza valori stringa per il matching
    for col in ['airline', 'from', 'to', 'class']:
        if col in df_clean.columns:
            df_clean[f'{col}_norm'] = df_clean[col].astype(str).str.lower().str.strip()
    
    # Converti price a float
    if 'price' in df_clean.columns:
        df_clean['price'] = df_clean['price'].apply(clean_price_value)
    
    # Converti date se presente
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    # Seleziona solo colonne utili
    essential_cols = ['date', 'airline', 'from', 'to', 'duration', 'class', 'price', 'stops']
    available_cols = [col for col in essential_cols if col in df_clean.columns]
    df_clean = df_clean[available_cols]
    
    print(f"[OK] Dataset generico caricato: {len(df_clean)} righe, {df_clean.shape[1]} colonne\n")
    
    return df_clean


def genera_file_unificato(economy_path, business_path, clean_path, output_path):
    """
    Unisce economy.csv e business.csv, matcha con Clean_Dataset.csv sui campi principali.
    Salva solo le righe con prezzo trovato nel dataset unificato.
    
    Args:
        economy_path (str): Percorso economy.csv
        business_path (str): Percorso business.csv
        clean_path (str): Percorso Clean_Dataset.csv
        output_path (str): Percorso file output unificato
    """
    
    # Carica i dati
    df_e = pd.read_csv(economy_path)
    df_b = pd.read_csv(business_path)
    df_e['class'] = 'Economy'
    df_b['class'] = 'Business'
    df = pd.concat([df_e, df_b], ignore_index=True)
    
    # Normalizza colonne
    if 'time_taken' in df.columns:
        df['duration'] = df['time_taken'].apply(parse_generic_duration)
    
    # Carica prezzi
    df_price = pd.read_csv(clean_path)
    if 'duration' in df_price.columns:
        df_price['duration'] = pd.to_numeric(df_price['duration'], errors='coerce')
    
    # Normalizza colonne per il matching
    df['airline_norm'] = df['airline'].astype(str).str.lower()
    df['from_norm'] = df['from'].astype(str).str.lower()
    df['to_norm'] = df['to'].astype(str).str.lower()
    df['class_norm'] = df['class'].astype(str).str.lower()
    
    df_price['airline_norm'] = df_price['airline'].astype(str).str.lower()
    df_price['source_city_norm'] = df_price['source_city'].astype(str).str.lower()
    df_price['destination_city_norm'] = df_price['destination_city'].astype(str).str.lower()
    df_price['class_norm'] = df_price['class'].astype(str).str.lower()
    
    # Groupby i prezzi per chiave per evitare il cartesian product
    print("[*] Grouping prices by key...")
    df_price_grouped = df_price.groupby(['airline_norm', 'source_city_norm', 'destination_city_norm', 'class_norm', 'duration']).agg({
        'price': 'mean'
    }).reset_index()
    df_price_grouped = df_price_grouped.rename(columns={'price': 'price_unified'})
    
    print(f"    Grouped prices: {len(df_price_grouped)} unique keys")
    
    # Matching merge su chiavi principali
    print("[*] Matching righe economy+business con prezzi...")
    df_merged = df.merge(
        df_price_grouped,
        left_on=['airline_norm', 'from_norm', 'to_norm', 'class_norm', 'duration'],
        right_on=['airline_norm', 'source_city_norm', 'destination_city_norm', 'class_norm', 'duration'],
        how='inner'
    )
    
    print(f"    After exact match: {len(df_merged)} rows")
    
    # Se l'exact match non dà risultati, prova con tolleranza di durata
    if len(df_merged) == 0:
        print("[*] No exact match found, trying with duration tolerance...")
        df_merged = df.merge(
            df_price_grouped[['airline_norm', 'source_city_norm', 'destination_city_norm', 'class_norm', 'duration', 'price_unified']],
            left_on=['airline_norm', 'from_norm', 'to_norm', 'class_norm'],
            right_on=['airline_norm', 'source_city_norm', 'destination_city_norm', 'class_norm'],
            how='left',
            suffixes=('', '_price')
        )
        df_merged['duration_diff'] = np.abs(df_merged['duration'] - df_merged['duration_price'])
        df_merged = df_merged[df_merged['duration_diff'] < 0.2].copy()
    
    # Dropna sul prezzo se presente
    if 'price_unified' in df_merged.columns:
        df_merged = df_merged.dropna(subset=['price_unified']).copy()
    
    # Mantieni solo le colonne utili
    cols_to_keep = [c for c in df.columns if c not in ['airline_norm', 'from_norm', 'to_norm', 'class_norm', 'price']]
    cols_available = [c for c in cols_to_keep if c in df_merged.columns]
    
    if 'price_unified' in df_merged.columns:
        cols_available.append('price_unified')
    
    if cols_available:
        df_merged = df_merged[cols_available]
    
    if 'price_unified' in df_merged.columns:
        df_merged = df_merged.rename(columns={'price_unified': 'price'})
    
    if 'price' in df_merged.columns:
        df_merged['price'] = pd.to_numeric(df_merged['price'], errors='coerce')
    
    if 'date' in df_merged.columns:
        df_merged['date'] = pd.to_datetime(df_merged['date'], format='%d-%m-%Y', errors='coerce')
    
    # Salva file
    df_merged.to_csv(output_path, index=False)
    print(f"[OK] File unificato creato: {output_path} ({len(df_merged)} righe, {df_merged.shape[1]} colonne)\n")


def carica_o_crea_unificato():
    """
    Carica il file unificato se esiste e aggiornato, altrimenti lo crea.
    Usa MD5 hashing per verificare se i file sorgente sono cambiati.
    
    Returns:
        pd.DataFrame: Dataset unificato
    """
    base_paths = ['.', '..']
    economy_path = None
    business_path = None
    clean_path = None
    
    for base in base_paths:
        e = os.path.join(base, 'economy.csv')
        b = os.path.join(base, 'business.csv')
        c = os.path.join(base, 'Clean_Dataset.csv')
        if os.path.exists(e) and os.path.exists(b) and os.path.exists(c):
            economy_path = e
            business_path = b
            clean_path = c
            break
    
    if economy_path is None:
        raise FileNotFoundError("Uno dei file sorgente non è presente (economy.csv, business.csv, Clean_Dataset.csv).")
    
    output_path = os.path.join(os.path.dirname(economy_path), 'Unified_Flights.csv')
    hash_path = output_path + '.md5'
    sorgenti = [economy_path, business_path, clean_path]
    
    new_hash = file_hash(sorgenti)
    if os.path.exists(output_path) and os.path.exists(hash_path):
        with open(hash_path) as f:
            old_hash = f.read().strip()
        if old_hash == new_hash:
            print(f"[OK] File unificato già aggiornato: {output_path}\n")
            return pd.read_csv(output_path)
    
    genera_file_unificato(economy_path, business_path, clean_path, output_path)
    with open(hash_path, 'w') as f:
        f.write(new_hash)
    
    return pd.read_csv(output_path)
