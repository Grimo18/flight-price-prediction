"""
Testi e messaggi in ITALIANO
=============================
Centralizza tutti i messaggi dell'interfaccia utente in italiano.
"""

MENU_PRINCIPALE = """
================================================================================
✈️ FLIGHT PRICE PREDICTOR - SISTEMA PREDIZIONE PREZZI VOLI
================================================================================

[*] Quale tipo di dataset hai?

  1. Tre file separati (economy.csv + business.csv + Clean_Dataset.csv)
     → Se hai i voli divisi per classe di viaggio

  2. Un file unico generico
     → Se hai un CSV unico con tutti i voli (qualsiasi meta/paese)

  3. Annulla

"""

CARICAMENTO_DATASET = "[*] Caricamento dataset..."
DATASET_CARICATO = "✅ Dataset caricato: {rows} righe, {cols} colonne"
COLONNE_DISPONIBILI = "Colonne disponibili: {cols}"

MENU_SCELTA_INVALIDA = "❌ Scelta non valida. Inserisci 1, 2 o 3"

CARICAMENTO_3_FILE = "[*] Caricamento dataset da 3 file (economy + business + Clean_Dataset)..."
ERRORE_CARICAMENTO_3_FILE = "❌ Errore nel caricamento dei 3 file: {error}"
ASSICURATI_3_FILE = "   Assicurati che esistano: economy.csv, business.csv, Clean_Dataset.csv"

CARICAMENTO_FILE_GENERICO = "[*] Quale file CSV vuoi caricare?"
INSERISCI_PERCORSO = "Inserisci il percorso del file (es: C:\\voli_tokyo.csv): "
FILE_NON_TROVATO = "❌ File non trovato: {path}"
CARICAMENTO_GENERICO = "[*] Caricamento dataset generico..."
ERRORE_CARICAMENTO_FILE = "❌ Errore nel caricamento del file: {error}"

OPERAZIONE_ANNULLATA = "❌ Operazione annullata"
DATASET_PRONTO = "[OK] Dataset pronto!\n"

PROGRAMMA_COMPLETATO = "✅ Programma completato!"

# Messaggi analisi
ANALISI_ESPLORATIVA = "📈 ANALISI ESPLORATIVA DEI DATI"
ANALISI_STAGIONALITA = "📅 ANALISI STAGIONALITA'"
PREPARAZIONE_FEATURES = "⚙️ PREPARAZIONE FEATURES"
TRAINING_MODELLO = "🤖 TRAINING MODELLO ML"
ANALISI_COMPLETATA = "✅ ANALISI COMPLETATA CON SUCCESSO!"

FILE_GENERATI = "File generati:"
PRICE_DISTRIBUTION = "  📊 price_distribution_analysis.png - Analisi distribuzione prezzi"
SEASONALITY_ANALYSIS = "  📈 seasonality_analysis.png - Analisi stagionalità"
FEATURE_IMPORTANCE = "  🎯 feature_importance.png - Importanza delle features"
SISTEMA_PRONTO = "🎉 Il sistema è pronto per fare predizioni!"

# Predizioni interattive
PREDIZIONE_INTERATTIVA = "🎯 PREDIZIONE INTERATTIVA PREZZO VOLO - MODALITA' INTELLIGENTE"
INSERISCI_DETTAGLI = "Inserisci i dettagli del volo per ottenere una predizione del prezzo:"

CITTA_DISPONIBILI = "[*] Città disponibili: {cities}"
CITTA_PARTENZA = "✈️ Città di partenza: "
CITTA_NON_TROVATA = "❌ Città non trovata. Scegli da: {cities}"

DESTINAZIONI_DISPONIBILI = "[*] Destinazioni disponibili da {city}: {destinations}"
CITTA_DESTINAZIONE = "✈️ Città di destinazione: "
DESTINAZIONE_NON_TROVATA = "❌ Destinazione non trovata. Scegli da: {destinations}"

MESE_VIAGGIO = "📅 Mese del viaggio (1-12): "
NUMERO_INVALIDO_MESE = "❌ Inserisci un numero tra 1 e 12"

GIORNO_MESE = "📅 Giorno del mese (1-{max}): "
NUMERO_INVALIDO_GIORNO = "❌ Inserisci un numero tra 1 e {max}"

GIORNI_MESE_INFO = "   [*] {month} 2022 ha {days} giorni"
ANNO_NON_BISESTILE = "   [*] Nota: 2022 non è un anno bisestile"

GIORNO_SETTIMANA = "✅ Giorno della settimana: {day}"
DATA_NON_VALIDA = "❌ Data non valida"

DURATA_MEDIA = "✅ Durata media calcolata: {hours}h {minutes}m"

CALCOLO_PREDIZIONE = "🔮 CALCOLO PREDIZIONE IN CORSO..."

PREZZO_PREDETTO = "💰 PREZZO PREDETTO: {price}"

DETTAGLI_VOLO = "📋 Dettagli del volo:"
ROTTA = "  ✈️ Rotta: {from} -> {to}"
DATA = "  📅 Data: {day} {month}"
GIORNO = "  📅 Giorno: {day}"
DURATA = "  ⏱️ Durata: {hours}h {minutes}m"

CONSIGLIO_COMPAGNIA = "💡 CONSIGLIO: COMPAGNIA AEREA PIU' ECONOMICA"
COMPAGNIA_ECONOMICA = "✅ {airline} - Prezzo medio: {price} ({count} voli)"
ALTRE_COMPAGNIE = "Altre compagnie su questa rotta:"
COMPAGNIA_ITEM = "  • {airline}: {price} ({count} voli)"

VUOI_ALTRA_PREDIZIONE = "🔄 Vuoi fare un'altra predizione? (s/n): "
PREDIZIONE_ANNULLATA = "❌ Predizione annullata dall'utente"
ERRORE_PREDIZIONE = "❌ Errore durante la predizione: {error}"

# Nomi giorni settimana e mesi
GIORNI_SETTIMANA = ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì', 'Sabato', 'Domenica']
MESI = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno',
        'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']
MESI_BREVI = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
