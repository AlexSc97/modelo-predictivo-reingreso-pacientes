import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración para guardar gráficos y modelos
MODEL_DIR = '../models'
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("Cargando datos...")
    # Cargar datos
    try:
        data = pd.read_csv('data/processed/data.csv')
    except FileNotFoundError:
        print("Error: No se encontró el archivo '../data/processed/data.csv'. Asegúrate de ejecutar los pasos previos.")
        return

    # Definir las 10 mejores características
    TOP_10_FEATURES = [
        'number_inpatient',      # Rank 1
        'discharge_segment',     # Rank 2
        'diabetesMed',           # Rank 3
        'number_emergency',      # Rank 4
        'metformin',             # Rank 5 (Replaces insulin)
        'diag_1_group',          # Rank 6
        'diag_2_group',          # Rank 10
        'number_diagnoses',      # Rank 9
        'age',                   # Rank 14
        'time_in_hospital'       # Rank 12
    ]

    print(f"Usando las siguientes características: {TOP_10_FEATURES}")

    # Preparar X e y
    X = data[TOP_10_FEATURES]
    y = data['target']

    # Calcular scale_pos_weight para manejar el desbalance de clases
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"Peso calculado para la clase positiva (scale_pos_weight): {scale_pos_weight:.2f}")

    # División estratificada para mantener la proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Identificar columnas numéricas y categóricas
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    # Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Pipeline base
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss',
            random_state=42
        ))
    ])

    # Función objetivo para Optuna
    def objective(trial):
        param_grid = {
            'xgb__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'xgb__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'xgb__max_depth': trial.suggest_int('max_depth', 3, 10),
            'xgb__min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'xgb__gamma': trial.suggest_float('gamma', 0, 5),
            'xgb__subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'xgb__colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'xgb__reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
            'xgb__reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            # Ampliar el rango de scale_pos_weight
            'xgb__scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_pos_weight, scale_pos_weight * 10)
        }

        pipeline.set_params(**param_grid)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Usamos roc_auc como métrica principal
        cv_scores = cross_val_score(
            pipeline, 
            X_train, 
            y_train, 
            cv=cv, 
            scoring='roc_auc', 
            n_jobs=-1
        )

        return cv_scores.mean()

    print("Iniciando optimización de hiperparámetros con Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Mejores hiperparámetros encontrados:")
    print(study.best_params)
    print(f"Mejor AUC ROC (CV): {study.best_value:.4f}")

    # Entrenar modelo final con los mejores parámetros
    best_params = {f'xgb__{k}': v for k, v in study.best_params.items()}
    pipeline.set_params(**best_params)
    
    print("Entrenando modelo final...")
    pipeline.fit(X_train, y_train)

    # Evaluación
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nReporte de Clasificación Final:")
    print(classification_report(y_test, y_pred))
    print(f"AUC ROC Final (Test): {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Guardar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.savefig(f'{MODEL_DIR}/confusion_matrix.png')
    print(f"Matriz de confusión guardada en {MODEL_DIR}/confusion_matrix.png")

    # Guardar modelo
    model_path = f'{MODEL_DIR}/xgboost_model.joblib'
    joblib.dump(pipeline, model_path)
    print(f"Modelo guardado exitosamente en {model_path}")
    print("Este archivo puede ser cargado en Flask usando joblib.load()")

if __name__ == "__main__":
    main()
