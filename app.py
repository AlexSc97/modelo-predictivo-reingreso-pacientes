from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import shap
from pathlib import Path

app = Flask(__name__, static_folder='static')
CORS(app)

# Rutas de los modelos
MODEL_PATH = Path(__file__).parent / 'models' / 'xgboost_model.joblib'

model = None

def load_model():
    """Cargar el modelo entrenado"""
    global model
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            print(f"✓ Modelo cargado desde {MODEL_PATH}")
        else:
            print(f"ERROR: No se encontró el archivo del modelo en {MODEL_PATH}")
            # No activamos modo demo, simplemente el modelo queda como None
            
    except Exception as e:
        print(f"ERROR CRÍTICO: No se pudo cargar el modelo: {e}")
        model = None

@app.route('/health')
def health():
    """Endpoint de verificación de estado"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


# Las 10 características más importantes
FEATURES = {
    'number_inpatient': {
        'type': 'integer',
        'min': 0,
        'max': 20,
        'description': 'Número de hospitalizaciones previas.',
        'label': 'Visitas Hospitalarias Previas'
    },
    'discharge_segment': {
        'type': 'categorical',
        'options': ['Otherwise', 'Discharged to home'],
        'description': 'Destino del paciente al ser dado de alta.',
        'label': 'Tipo de Alta'
    },
    'diabetesMed': {
        'type': 'categorical',
        'options': ['Yes', 'No'],
        'description': 'Si el paciente recibe medicación para la diabetes.',
        'label': 'Medicación para Diabetes'
    },
    'number_emergency': {
        'type': 'integer',
        'min': 0,
        'max': 20,
        'description': 'Número de visitas a emergencias en el año anterior.',
        'label': 'Visitas de Emergencia'
    },
    'metformin': {
        'type': 'categorical',
        'options': ['No', 'Steady', 'Up', 'Down'],
        'description': 'Uso de Metformina.',
        'label': 'Metformina'
    },
    'diag_1_group': {
        'type': 'categorical',
        'options': ['Diabetes', 'Other', 'Neoplasms', 'Circulatory', 'Respiratory', 'Injury', 'Musculoskeletal', 'Digestive', 'Genitourinary'],
        'description': 'Grupo de diagnóstico primario.',
        'label': 'Diagnóstico Primario'
    },
    'diag_2_group': {
        'type': 'categorical',
        'options': ['Diabetes', 'Other', 'Neoplasms', 'Circulatory', 'Respiratory', 'Injury', 'Musculoskeletal', 'Digestive', 'Genitourinary'],
        'description': 'Grupo de diagnóstico secundario.',
        'label': 'Diagnóstico Secundario'
    },
    'number_diagnoses': {
        'type': 'integer',
        'min': 1,
        'max': 16,
        'description': 'Número total de diagnósticos registrados.',
        'label': 'Número de Diagnósticos'
    },
    'age': {
        'type': 'categorical', # Tratamos la edad como categórica/ordinal para el dropdown
        'options': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        'description': 'Edad del paciente (punto medio del rango decenal).',
        'label': 'Edad'
    },
    'time_in_hospital': {
        'type': 'integer',
        'min': 1,
        'max': 14,
        'description': 'Días de estancia en el hospital.',
        'label': 'Días en Hospital'
    }
}

FEATURE_ORDER = [
    'number_inpatient',
    'discharge_segment',
    'diabetesMed',
    'number_emergency',
    'metformin',
    'diag_1_group',
    'diag_2_group',
    'number_diagnoses',
    'age',
    'time_in_hospital'
]

def update_feature_importance():
    """Actualizar la importancia de las características desde el modelo"""
    global FEATURES, model
    if model is not None:
        try:
            xgb_model = model.named_steps['xgb']
            # Obtener importancia (gain)
            importance = xgb_model.feature_importances_
            
            # Mapear importancia a las características originales
            # Esto es una aproximación ya que OneHotEncoder expande las características
            # Asignaremos la importancia máxima de las columnas codificadas a la característica original
            
            preprocessor = model.named_steps['preprocessor']
            feature_names_out = preprocessor.get_feature_names_out()
            
            # Crear mapa de feature original -> max importancia
            temp_importance = {k: 0.0 for k in FEATURES.keys()}
            
            for name, imp in zip(feature_names_out, importance):
                clean_name = name.replace('num__', '').replace('cat__', '')
                # Buscar a qué feature original pertenece
                for original_feat in FEATURES.keys():
                    if original_feat in clean_name:
                        temp_importance[original_feat] = max(temp_importance[original_feat], float(imp))
                        break
            
            # Normalizar a porcentaje del total
            total_imp = sum(temp_importance.values()) if temp_importance.values() else 1.0
            if total_imp > 0:
                for k in FEATURES:
                    FEATURES[k]['importance'] = (temp_importance[k] / total_imp) * 100
                    
        except Exception as e:
            print(f"Error actualizando importancia: {e}")

# Actualizar importancia al cargar
load_model()
update_feature_importance()

@app.route('/')
def index():
    """Página HTML principal"""
    return send_from_directory('static', 'index.html')

@app.route('/api/features')
def get_features():
    """Devolver metadatos de las características"""
    return jsonify({
        'features': FEATURES,
        'feature_order': FEATURE_ORDER
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predecir el riesgo de readmisión del paciente
    """
    try:
        # Verificar si el modelo está cargado
        if model is None:
            return jsonify({'error': 'El modelo no está cargado. Contacte al administrador.'}), 503

        # Obtener datos JSON
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        # Validar que todas las características requeridas estén presentes
        missing_features = [f for f in FEATURE_ORDER if f not in data]
        if missing_features:
            return jsonify({
                'error': 'Faltan características requeridas',
                'missing': missing_features
            }), 400
        
        # Crear DataFrame para la predicción (el pipeline espera un DataFrame)
        input_df = pd.DataFrame([data])
        
        # Realizar predicción con el modelo real
        # Obtener probabilidades: [prob_clase_0, prob_clase_1]
        probability = model.predict_proba(input_df)[0]
        prob_score = probability[1]
        
        # Lógica de 3 niveles de riesgo
        if prob_score >= 0.70:
            prediction = 2  # Alto Riesgo
            risk_level = 'high'
            prediction_label = 'Alto Riesgo de Readmisión'
        elif prob_score >= 0.51:
            prediction = 1  # Riesgo Medio
            risk_level = 'medium'
            prediction_label = 'Riesgo Medio de Readmisión'
        else:
            prediction = 0  # Bajo Riesgo
            risk_level = 'low'
            prediction_label = 'Bajo Riesgo de Readmisión'
            
        # --- EXPLICABILIDAD (SHAP) ---
        try:
            # 1. Transformar los datos de entrada usando el preprocesador del pipeline
            # Accedemos al paso 'preprocessor' del pipeline
            preprocessor = model.named_steps['preprocessor']
            X_transformed = preprocessor.transform(input_df)
            
            # 2. Obtener nombres de características transformadas
            feature_names = preprocessor.get_feature_names_out()
            
            # 3. Calcular valores SHAP
            # Accedemos al modelo XGBoost dentro del pipeline
            xgb_model = model.named_steps['xgb']
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_transformed)
            
            # shap_values puede ser una lista (para multiclase) o array. Para binaria suele ser array.
            # Si es lista, tomamos el índice 1 (clase positiva)
            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            else:
                vals = shap_values[0] # Para XGBoost binario a veces retorna directo
                
            # 4. Crear lista de impacto
            # Mapear nombres limpios (remover prefijos de OneHotEncoder si es posible para mejor visualización)
            impact_list = []
            for name, val in zip(feature_names, vals):
                # Limpiar nombres un poco para el frontend
                clean_name = name.replace('num__', '').replace('cat__', '')
                # Traducir/Mejorar nombres comunes
                if 'number_inpatient' in clean_name: clean_name = 'Visitas Hospitalarias'
                elif 'number_emergency' in clean_name: clean_name = 'Visitas Emergencia'
                elif 'time_in_hospital' in clean_name: clean_name = 'Días en Hospital'
                elif 'age' in clean_name: clean_name = 'Edad'
                elif 'discharge_segment' in clean_name: clean_name = 'Tipo de Alta'
                elif 'number_diagnoses' in clean_name: clean_name = 'Num. Diagnósticos'
                elif 'diabetesMed' in clean_name: clean_name = 'Medicación Diabetes'
                elif 'metformin' in clean_name: clean_name = 'Metformina'
                elif 'diag_1' in clean_name: clean_name = 'Diag. Primario'
                elif 'diag_2' in clean_name: clean_name = 'Diag. Secundario'
                
                impact_list.append({
                    'feature': clean_name,
                    'impact': float(val), # Convertir numpy.float32 a float nativo
                    'abs_impact': float(abs(val)) # Convertir numpy.float32 a float nativo
                })
            
            # Ordenar por impacto absoluto y tomar los top 5
            impact_list.sort(key=lambda x: x['abs_impact'], reverse=True)
            top_features = impact_list[:5]
            
        except Exception as e:
            print(f"Error calculando SHAP: {e}")
            top_features = []
        
        # Preparar respuesta
        result = {
            'prediction': int(prediction),
            'risk_level': risk_level,
            'prediction_label': prediction_label,
            'probability': {
                'low_risk': float(probability[0]),
                'high_risk': float(probability[1])
            },
            'risk_percentage': float(probability[1] * 100),
            'input_features': data,
            'top_features': top_features, # Agregamos los factores clave
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("API de Predicción de Readmisión de Pacientes")
    print("="*60)
    print(f"Modelo cargado: {model is not None}")
    print(f"Número de características: {len(FEATURE_ORDER)}")
    print(f"\nIniciando servidor en http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
