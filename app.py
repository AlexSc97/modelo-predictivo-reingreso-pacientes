from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import shap
from pathlib import Path

app = Flask(__name__, static_folder='static')
CORS(app)

# Rutas de los modelos - ACTUALIZADO PARA USAR modelo_produccion.pkl
MODEL_PATH = Path(__file__).parent / 'models' / 'modelo_produccion.pkl'

model = None

def load_model():
    """Cargar el modelo entrenado"""
    global model
    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Modelo cargado desde {MODEL_PATH}")
        else:
            print(f"ERROR: No se encontró el archivo del modelo en {MODEL_PATH}")
            print(f"Por favor, ejecute train_model_production.py primero")
            model = None
            
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


# Features según el modelo de producción
# Estas deben coincidir EXACTAMENTE con las usadas en el entrenamiento

# Features numéricas (orden importa)
NUMERICAL_FEATURES = [
    'time_in_hospital',
    'num_procedures',
    'number_diagnoses',
    'number_inpatient',
    'number_emergency'
]

# Features categóricas (orden importa)
CATEGORICAL_FEATURES = [
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

# Definición de features para el frontend (incluye todas las features)
FEATURES = {
    'time_in_hospital': {
        'type': 'integer',
        'min': 1,
        'max': 14,
        'description': 'Días de estancia en el hospital.',
        'label': 'Días en Hospital'
    },
    'num_procedures': {
        'type': 'integer',
        'min': 0,
        'max': 10,
        'description': 'Número de procedimientos realizados.',
        'label': 'Procedimientos'
    },
    'number_diagnoses': {
        'type': 'integer',
        'min': 1,
        'max': 16,
        'description': 'Número total de diagnósticos registrados.',
        'label': 'Número de Diagnósticos'
    },
    'number_inpatient': {
        'type': 'integer',
        'min': 0,
        'max': 20,
        'description': 'Número de hospitalizaciones previas.',
        'label': 'Visitas Hospitalarias Previas'
    },
    'number_emergency': {
        'type': 'integer',
        'min': 0,
        'max': 20,
        'description': 'Número de visitas a emergencias en el año anterior.',
        'label': 'Visitas de Emergencia'
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
    'age': {
        'type': 'categorical',
        'options': ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
        'description': 'Rango de edad del paciente.',
        'label': 'Edad'
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
    'A1Cresult': {
        'type': 'categorical',
        'options': ['None', 'Norm', '>7', '>8'],
        'description': 'Resultado del test A1C.',
        'label': 'A1C Result'
    },
    'insulin': {
        'type': 'categorical',
        'options': ['No', 'Steady', 'Up', 'Down'],
        'description': 'Uso de insulina.',
        'label': 'Insulina'
    },
    'weight': {
        'type': 'categorical',
        'options': ['?', '[0-25)', '[25-50)', '[50-75)', '[75-100)', '[100-125)', '[125-150)', '[150-175)', '[175-200)', '>200'],
        'description': 'Rango de peso del paciente.',
        'label': 'Peso'
    }
}

# Orden de las features (TODAS las features usadas en entrenamiento)
FEATURE_ORDER = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def update_feature_importance():
    """Actualizar la importancia de las características desde el modelo"""
    global FEATURES, model
    if model is not None:
        try:
            # Verificar que el modelo es un Pipeline
            if not hasattr(model, 'named_steps'):
                print("Advertencia: El modelo no es un Pipeline válido")
                return
            
            # Verificar que tiene el paso 'xgb'
            if 'xgb' not in model.named_steps:
                print(f"Advertencia: El pipeline no tiene paso 'xgb'. Pasos disponibles: {list(model.named_steps.keys())}")
                return
                
            xgb_model = model.named_steps['xgb']
            
            # Verificar que tiene feature_importances_
            if not hasattr(xgb_model, 'feature_importances_'):
                print("Advertencia: El modelo XGBoost no tiene feature_importances_")
                return
                
            importance = xgb_model.feature_importances_
            
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
            
            print("✓ Importancia de features actualizada correctamente")
                    
        except Exception as e:
            print(f"Error actualizando importancia: {e}")
            import traceback
            traceback.print_exc()

# Cargar modelo al iniciar
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


@app.route('/api/shap_plot', methods=['POST'])
def generate_shap_plot():
    """
    Generar gráfico SHAP waterfall interactivo como HTML
    """
    try:
        # Verificar si el modelo está cargado
        if model is None:
            return jsonify({'error': 'El modelo no está cargado. Por favor entrene el modelo primero ejecutando train_model_production.py'}), 503

        # Obtener datos JSON
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        # Validar características
        missing_features = [f for f in FEATURE_ORDER if f not in data]
        if missing_features:
            return jsonify({'error': 'Faltan características requeridas', 'missing': missing_features}), 400
        
        # Crear DataFrame con ORDEN CORRECTO
        input_df = pd.DataFrame([data])[FEATURE_ORDER]
        
        # Transformar datos
        preprocessor = model.named_steps['preprocessor']
        X_transformed = preprocessor.transform(input_df)
        
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        
        # Calcular SHAP
        xgb_model = model.named_steps['xgb']
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]
        
        # Obtener nombres originales y limpiarlos
        raw_feature_names = preprocessor.get_feature_names_out()
        clean_feature_names = []
        
        for name in raw_feature_names:
            # Limpiar prefijos
            name = name.replace('num__', '').replace('cat__', '')
            
            # Traducciones y limpieza específica
            if 'time_in_hospital' in name: name = 'Días Hospital'
            elif 'num_procedures' in name: name = 'Procedimientos'
            elif 'number_diagnoses' in name: name = 'Num Diagnósticos'
            elif 'number_inpatient' in name: name = 'Hospitalizaciones Previas'
            elif 'number_emergency' in name: name = 'Emergencias Previas'
            elif 'discharge_segment' in name: 
                if 'Discharged to home' in name: name = 'Alta: Domicilio'
                elif 'Otherwise' in name: name = 'Alta: Otro'
                else: name = 'Tipo de Alta'
            elif 'diabetesMed' in name: name = 'Medicación Diabetes'
            elif 'age' in name: name = f"Edad {name.split('_')[-1]}"
            elif 'metformin' in name: name = f"Metformina: {name.split('_')[-1]}"
            elif 'diag_1_group' in name: name = f"Diag 1: {name.split('_')[-1]}"
            elif 'diag_2_group' in name: name = f"Diag 2: {name.split('_')[-1]}"
            elif 'A1Cresult' in name: name = f"A1C: {name.split('_')[-1]}"
            elif 'insulin' in name: name = f"Insulina: {name.split('_')[-1]}"
            elif 'weight' in name: name = 'Peso'
            
            clean_feature_names.append(name)

        # Base value
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        # Crear explicación SHAP
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=base_value,
            data=X_transformed[0],
            feature_names=clean_feature_names
        )
        
        # Generar gráfico
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64
        
        # Configurar estilo para que coincida con la UI (Soft UI)
        plt.style.use('default') 
        fig = plt.figure(figsize=(10, 6), facecolor='none') # Fondo transparente
        
        # Generar el waterfall plot
        shap.plots.waterfall(explanation, show=False, max_display=10)
        
        # Ajustes visuales finos
        ax = plt.gca()
        ax.set_facecolor('none') # Fondo del eje transparente
        
        # Guardar
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', transparent=True)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        html_content = f'''
        <div style="width: 100%; display: flex; justify-content: center; align-items: center;">
            <img src="data:image/png;base64,{image_base64}" 
                 style="max-width: 100%; height: auto; border-radius: 8px;"
                 alt="Análisis de Factores SHAP">
        </div>
        '''
        
        return jsonify({
            'html': html_content,
            'base_value': float(base_value)
        })
        
    except Exception as e:
        print(f"Error generando gráfico SHAP: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predecir el riesgo de readmisión del paciente
    """
    try:
        # Verificar si el modelo está cargado
        if model is None:
            return jsonify({'error': 'El modelo no está cargado. Por favor entrene el modelo primero ejecutando train_model_production.py'}), 503

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
        
        # IMPORTANTE: Crear DataFrame con el ORDEN CORRECTO de features
        # Esto es CRÍTICO - el modelo espera las features en este orden exacto
        input_df = pd.DataFrame([data])[FEATURE_ORDER]
        
        # Realizar predicción con el modelo
        probability = model.predict_proba(input_df)[0]
        prob_score = float(probability[1])  # Probabilidad base del modelo
        
        # --- AJUSTE DE REGLAS EXPERTAS (CLINICAL OVERLAY) ---
        # El modelo ML puede ser conservador. Ajustamos basado en factores de riesgo clínicos conocidos.
        adjustment = 0.0
        
        # 1. Ajuste por Peso (Obesidad mórbida es factor de riesgo)
        weight_val = data.get('weight', '?')
        if weight_val in ['[100-125)', '[125-150)', '[150-175)', '[175-200)', '>200']:
            adjustment += 0.08  # +8% riesgo
            
        # 2. Ajuste por Descontrol Glucémico (A1C alto)
        a1c_val = data.get('A1Cresult', 'None')
        if a1c_val in ['>8', '>7']:
            adjustment += 0.05  # +5% riesgo
            
        # 3. Ajuste por Inestabilidad de Insulina
        insulin_val = data.get('insulin', 'No')
        if insulin_val in ['Up', 'Down']: # Cambios recientes en dosis
            adjustment += 0.04  # +4% riesgo
            
        # 4. Ajuste por Polifarmacia/Complejidad (muchos diagnósticos)
        if int(data.get('number_diagnoses', 0)) > 8:
             adjustment += 0.03

        # Aplicar ajuste (tope máximo 0.95)
        prob_score = min(prob_score + adjustment, 0.95)
        
        # Recalcular probabilidad complementaria
        probability_0 = 1.0 - prob_score
        probability = [probability_0, prob_score]
        
        # Lógica de 3 niveles de riesgo
        if prob_score >= 0.65: # Umbral ajustado
            prediction = 2  # Alto Riesgo
            risk_level = 'high'
            prediction_label = 'Alto Riesgo de Readmisión'
        elif prob_score >= 0.45: # Umbral ajustado para ser más sensible
            prediction = 1  # Riesgo Medio
            risk_level = 'medium'
            prediction_label = 'Riesgo Medio de Readmisión'
        else:
            prediction = 0  # Bajo Riesgo
            risk_level = 'low'
            prediction_label = 'Bajo Riesgo de Readmisión'
            
        # --- EXPLICABILIDAD (SHAP) ---
        shap_base_value = None
        try:
            preprocessor = model.named_steps['preprocessor']
            X_transformed = preprocessor.transform(input_df)
            
            # Convertir a denso si es disperso
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            
            feature_names = preprocessor.get_feature_names_out()
            
            # Calcular valores SHAP
            xgb_model = model.named_steps['xgb']
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_transformed)
            
            # Obtener base value
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                shap_base_value = float(base_value[1] if len(base_value) > 1 else base_value[0])
            else:
                shap_base_value = float(base_value)
            
            # Obtener valores SHAP
            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            else:
                vals = shap_values[0]
                
            # Crear lista de impacto con nombres mejorados
            impact_list = []
            for name, val in zip(feature_names, vals):
                clean_name = name.replace('num__', '').replace('cat__', '')
                # Traducir nombres
                if 'number_inpatient' in clean_name: clean_name = 'Visitas Hospitalarias'
                elif 'number_emergency' in clean_name: clean_name = 'Visitas Emergencia'
                elif 'time_in_hospital' in clean_name: clean_name = 'Días en Hospital'
                elif 'age' in clean_name: clean_name = 'Edad'
                elif 'discharge_segment' in clean_name: clean_name = 'Tipo de Alta'
                elif 'number_diagnoses' in clean_name: clean_name = 'Num. Diagnósticos'
                elif 'diabetesMed' in clean_name: clean_name = 'Medicación Diabetes'
                elif 'metformin' in clean_name: clean_name = 'Metformina'
                elif 'num_procedures' in clean_name: clean_name = 'Procedimientos'
                elif 'diag_1' in clean_name: clean_name = 'Diag. Primario'
                elif 'diag_2' in clean_name: clean_name = 'Diag. Secundario'
                elif 'A1Cresult' in clean_name: clean_name = 'A1C'
                elif 'insulin' in clean_name: clean_name = 'Insulina'
                elif 'weight' in clean_name: clean_name = 'Peso'
                
                impact_list.append({
                    'feature': clean_name,
                    'impact': float(val),
                    'abs_impact': float(abs(val))
                })
            
            # Ordenar por impacto absoluto y tomar top 5
            impact_list.sort(key=lambda x: x['abs_impact'], reverse=True)
            top_features = impact_list[:5]
            
        except Exception as e:
            print(f"Error calculando SHAP: {e}")
            import traceback
            traceback.print_exc()
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
            'top_features': top_features,
            'shap_base_value': shap_base_value,
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("API de Predicción de Readmisión de Pacientes")
    print("="*60)
    print(f"Modelo cargado: {model is not None}")
    if model is None:
        print("\n⚠️  ADVERTENCIA: El modelo no está cargado")
        print("   Por favor ejecute: python train_model_production.py")
    print(f"Número de características: {len(FEATURE_ORDER)}")
    print(f"\nIniciando servidor en http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
