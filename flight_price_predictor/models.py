"""
Models Module
=============
Gestisce training, valutazione e predizioni dei modelli ML.
Utilizza RandomForest e GradientBoosting per regressione.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MLModels:
    """Gestisce training e validazione dei modelli di machine learning"""
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.categorical_features = []
        self.numeric_features = []
        self.categorical_modes = {}
        self.numeric_means = {}
    
    
    def prepare_features(self, df):
        """
        Prepara le features per il training del modello.
        
        Args:
            df (pd.DataFrame): Dataset pulito
        
        Returns:
            tuple: (X, y) dove X sono le features e y è il target (price)
        """
        df_model = df.copy()
        
        # Identifica colonne categoriche e numeriche
        categorical_features = df_model.select_dtypes(include=['object']).columns.tolist()
        numeric_features = df_model.select_dtypes(include=[np.number]).columns.tolist()

        # Salva liste per uso futuro
        self.categorical_features = categorical_features.copy()
        self.numeric_features = numeric_features.copy()
        
        # Rimuovi 'price' dal target
        if 'price' in numeric_features:
            numeric_features.remove('price')
        
        # Calcola defaults per riempimento
        for col in categorical_features:
            col_series = df_model[col].astype(str)
            if not col_series.empty:
                try:
                    self.categorical_modes[col] = col_series.mode().iloc[0]
                except Exception:
                    self.categorical_modes[col] = col_series.dropna().iloc[0] if col_series.dropna().shape[0] else ''
            else:
                self.categorical_modes[col] = ''

        for col in numeric_features:
            try:
                self.numeric_means[col] = float(pd.to_numeric(df_model[col], errors='coerce').mean())
            except Exception:
                self.numeric_means[col] = 0.0

        # Encode colonne categoriche
        for col in categorical_features:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df_model[col] = self.encoders[col].fit_transform(df_model[col].astype(str))
            else:
                df_model[col] = self.encoders[col].transform(df_model[col].astype(str))
        
        # Seleziona features finali
        self.feature_columns = categorical_features + numeric_features
        X = df_model[self.feature_columns].fillna(0)
        y = df_model['price'].fillna(df_model['price'].mean())
        
        return X, y
    
    
    def train_model(self, X, y, test_size=0.2):
        """
        Addestra il modello di Random Forest.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target price
            test_size (float): Proporzione test set
        
        Returns:
            tuple: (model, X_test, y_test, metrics)
        """
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train RandomForest
        print("\n[*] Training Random Forest Regressor...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Valutazione
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mse': mse
        }
        
        print(f"    MAE: €{mae:.2f}")
        print(f"    RMSE: €{rmse:.2f}")
        print(f"    R² (Test Set): {r2:.4f}")
        
        # Cross-validation
        print("\n[*] Cross-validation (5-fold)...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"    CV R² scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"    Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.models['random_forest'] = model
        
        return model, X_test_scaled, y_test, metrics
    
    
    def predict(self, X):
        """
        Fa una predizione con il modello addestrato.
        
        Args:
            X (pd.DataFrame or dict): Features per predizione
        
        Returns:
            float: Prezzo predetto
        """
        if 'random_forest' not in self.models:
            raise ValueError("Modello non ancora addestrato. Esegui train_model() prima.")
        
        # Se X è un dict, convertilo a DataFrame
        if isinstance(X, dict):
            X = pd.DataFrame([X])

        # Assicurati che tutte le feature esistano, riempi mancanti con defaults
        for col in self.feature_columns:
            if col not in X.columns:
                if col in self.encoders:  # categorica
                    default_val = self.categorical_modes.get(col, '')
                    X[col] = default_val
                else:  # numerica
                    default_val = self.numeric_means.get(col, 0.0)
                    X[col] = default_val

        # Encode colonne categoriche (usando i label encoder addestrati)
        for col in self.encoders:
            if col in X.columns:
                try:
                    X[col] = self.encoders[col].transform(X[col].astype(str))
                except Exception:
                    # se valore non visto, mappa a 0
                    X[col] = 0

        # Converti numeriche e riempi NaN
        for col in self.feature_columns:
            if col not in self.encoders:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                if X[col].isna().any():
                    X[col] = X[col].fillna(self.numeric_means.get(col, 0.0))

        # Seleziona feature columns
        X_subset = X[self.feature_columns].copy()

        # Scale
        X_scaled = self.scaler.transform(X_subset)
        
        # Predici
        prediction = self.models['random_forest'].predict(X_scaled)
        
        return float(prediction[0])
    
    
    def get_feature_importance(self, top_n=10):
        """
        Ritorna l'importanza delle features dal modello.
        
        Args:
            top_n (int): Numero di top features da ritornare
        
        Returns:
            pd.DataFrame: Feature importance
        """
        if 'random_forest' not in self.models:
            raise ValueError("Modello non ancora addestrato.")
        
        model = self.models['random_forest']
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance.head(top_n)
    
    
    def save_model(self, path='flight_price_model.pkl'):
        """
        Salva il modello addestrato.
        
        Args:
            path (str): Percorso file
        """
        if 'random_forest' not in self.models:
            raise ValueError("Nessun modello da salvare.")
        
        model_data = {
            'model': self.models['random_forest'],
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, path)
        print(f"[OK] Modello salvato: {path}")
    
    
    def load_model(self, path='flight_price_model.pkl'):
        """
        Carica un modello precedentemente salvato.
        
        Args:
            path (str): Percorso file
        """
        model_data = joblib.load(path)
        
        self.models['random_forest'] = model_data['model']
        self.encoders = model_data['encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        
        print(f"[OK] Modello caricato: {path}")
