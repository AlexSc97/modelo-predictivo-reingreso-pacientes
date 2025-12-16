"""
Script para entrenar el modelo de producción siguiendo la lógica de modelo_produccion_flask.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import pickle
import os

# Configuración de rutas
DATA_PATH = 'data/raw/diabetic_data.csv'
MODEL_PATH = 'models/modelo_produccion.pkl'

print("="*60)
print("ENTRENAMIENTO DEL MODELO DE PRODUCCIÓN")
print("="*60)

# 1. CARGA DE DATOS
print("\n1. Cargando datos...")
df = pd.read_csv(DATA_PATH, na_values="?")
print(f"   Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")

# 2. DEFINIR TARGET
print("\n2. Definiendo variable objetivo...")
df['target'] = (df['readmitted'] == '<30').astype(int)
df = df.drop(columns=['readmitted'])
print(f"   Distribución target: {df['target'].value_counts().to_dict()}")

# 3. CARACTERÍSTICAS MÁS IMPORTANTES (según SHAP del notebook)
print("\n3. Seleccionando características importantes...")
features_importantes = {
    'discharge_disposition_id',  # Destino del alta
    'number_inpatient',          # Numero de hospitalizaciones previas
    'diabetesMed',               # Medicación para diabetes
    'number_emergency',          # Numero de visitas a emergencias
    'diag_1',                   # Diagnóstico primario
    'metformin',                # Uso de metformina
    'diag_2',                   # Diagnóstico secundario
    'time_in_hospital',         # Días en hospital
    'age',                      # Edad
    'number_diagnoses',         # Número de diagnósticos
    'num_procedures',           # Número de procedimientos
    'insulin',                  # Resistencia a la insulina
    'A1Cresult',               # Resultado del A1C
    'weight'                    # Peso del paciente
}

print(f"   Características seleccionadas: {len(features_importantes)}")

# 4. MAPEO DE DIAGNÓSTICOS A GRUPOS (según notebook)
print("\n4. Mapeando diagnósticos ICD-9 a grupos...")

def map_icd9_to_category(code):
    """Mapea códigos ICD-9 a categorías según el notebook"""
    if pd.isna(code):
        return 'Other'
    
    try:
        code_num = float(str(code).replace('V', '').replace('E', ''))
        
        # Diabetes
        if 250 <= code_num < 251:
            return 'Diabetes'
        # Circulatory
        elif (390 <= code_num < 460) or (785 <= code_num < 786):
            return 'Circulatory'
        # Respiratory
        elif (460 <= code_num < 520) or (786 <= code_num < 787):
            return 'Respiratory'
        # Digestive
        elif (520 <= code_num < 580) or (787 <= code_num < 788):
            return 'Digestive'
        # Injury
        elif (800 <= code_num < 1000) or str(code).startswith('E'):
            return 'Injury'
        # Musculoskeletal
        elif 710 <= code_num < 740:
            return 'Musculoskeletal'
        # Genitourinary
        elif (580 <= code_num < 630) or (788 <= code_num < 789):
            return 'Genitourinary'
        # Neoplasms
        elif 140 <= code_num < 240:
            return 'Neoplasms'
        else:
            return 'Other'
    except:
        return 'Other'

# Aplicar mapeo
df['diag_1_group'] = df['diag_1'].apply(map_icd9_to_category)
df['diag_2_group'] = df['diag_2'].apply(map_icd9_to_category)

# 5. SIMPLIFICAR DISCHARGE_DISPOSITION_ID (según notebook)
print("\n5. Simplificando discharge_disposition_id...")

def simplify_discharge(discharge_id):
    """Simplifica discharge_disposition_id en dos categorías"""
    home_discharge = [1, 6, 8]  # Discharged to home
    if discharge_id in home_discharge:
        return 'Discharged to home'
    else:
        return 'Otherwise'

df['discharge_segment'] = df['discharge_disposition_id'].apply(simplify_discharge)

# 6. PREPARAR CARACTERÍSTICAS
print("\n6. Preparando características...")

# Actualizar lista de features con las nuevas columnas
numerical_features = [
    'time_in_hospital',
    'num_procedures',
    'number_diagnoses',
    'number_inpatient',
    'number_emergency'
]

categorical_features = [
    'discharge_segment',
    'diabetesMed',
    'age',
    'metformin',
    'diag_1_group',
    'diag_2_group',
    'A1Cresult',
    'insulin',
    'weight'
]

# Seleccionar solo las columnas que necesitamos
features_to_use = numerical_features + categorical_features
X = df[features_to_use].copy()
y = df['target']

# --- AUMENTO DE IMPORTANCIA DE WEIGHT ---
# Como weight tiene muchos faltantes (?), el modelo lo ignora.
# Vamos a duplicar los casos que SÍ tienen peso para forzar al modelo a aprenderlos.
print("\n6.1. Aumentando importancia de variable 'weight' (Oversampling)...")
indices_con_peso = X[X['weight'] != '?'].index
n_duplicados = 25  # Repetir 25 veces estos registros

# Crear DataFrames de los registros a duplicar
X_weight = X.loc[indices_con_peso]
y_weight = y.loc[indices_con_peso]

# Concatenar al dataset original
X = pd.concat([X] + [X_weight] * n_duplicados, ignore_index=True)
y = pd.concat([y] + [y_weight] * n_duplicados, ignore_index=True)

print(f"   Registros con peso original: {len(indices_con_peso)}")
print(f"   Nuevos registros agregados: {len(indices_con_peso) * n_duplicados}")
print(f"   Total registros entrenamiento: {X.shape[0]}")
# ----------------------------------------

print(f"   Features numéricas: {len(numerical_features)}")
print(f"   Features categóricas: {len(categorical_features)}")
print(f"   Total features: {X.shape[1]}")

# 7. SPLIT DE DATOS
print("\n7. Dividiendo datos en train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
print(f"   Train: {X_train.shape[0]} registros")
print(f"   Test: {X_test.shape[0]} registros")

# 8. CREAR PIPELINE
print("\n8. Creando pipeline de preprocesamiento...")

# Preprocesador numérico
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocesador categórico
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combinador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Calcular scale_pos_weight para balancear clases
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
scale_pos_weight = num_neg / num_pos
print(f"\n   Ajustando desbalance de clases:")
print(f"   Negativos: {num_neg}, Positivos: {num_pos}")
print(f"   Scale Pos Weight: {scale_pos_weight:.2f}")

# Pipeline completo (según los mejores hiperparámetros del notebook de Optuna)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb', XGBClassifier(
        scale_pos_weight=scale_pos_weight,  # IMPORTANTE: Corrige el desbalance
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1,
        tree_method='hist'
    ))
])

print("   Pipeline creado exitosamente (con balanceo de clases)")

# 9. ENTRENAR MODELO
print("\n9. Entrenando modelo XGBoost...")
print("   Esto puede tomar varios minutos...")

model_pipeline.fit(X_train, y_train)

print("   ✓ Modelo entrenado")

# 10. EVALUAR MODELO
print("\n10. Evaluando modelo...")
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n   AUC Score: {auc:.4f}")

print("\n   Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\n   Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 11. GUARDAR MODELO
print("\n11. Guardando modelo...")

# Crear directorio si no existe
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Guardar modelo
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_pipeline, f)

print(f"   ✓ Modelo guardado en: {MODEL_PATH}")

# 12. GUARDAR METADATOS
print("\n12. Guardando metadatos...")

metadata = {
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'auc_score': auc,
    'train_samples': X_train.shape[0],
    'test_samples': X_test.shape[0],
    'features_count': X.shape[1]
}

metadata_path = MODEL_PATH.replace('.pkl', '_metadata.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"   ✓ Metadatos guardados en: {metadata_path}")

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("="*60)
print(f"\nModelo listo para usar en: {MODEL_PATH}")
print(f"AUC Score: {auc:.4f}")
print("\n¡El modelo está listo para producción!")
