"""
Flask API for Patient Readmission Prediction Model
Serves the frontend and provides prediction endpoints using the trained XGBoost model.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os
from pathlib import Path

app = Flask(__name__, static_folder='static')
CORS(app)

# Load the trained model
MODEL_PATH = Path(__file__).parent / 'models' / 'top10_model.pkl'
PIPELINE_PATH = Path(__file__).parent / 'models' / 'model_pipeline.pkl'

model = None
pipeline = None
DEMO_MODE = False

def load_model():
    """Load the trained model and pipeline"""
    global model, pipeline, DEMO_MODE
    try:
        # Try loading the top10 model first
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Model loaded from {MODEL_PATH}")
        
        # Try loading the pipeline
        if PIPELINE_PATH.exists():
            with open(PIPELINE_PATH, 'rb') as f:
                pipeline = pickle.load(f)
            print(f"✓ Pipeline loaded from {PIPELINE_PATH}")
            
        if model is None and pipeline is None:
            raise FileNotFoundError("No model files found")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not load model: {e}")
        print("⚠️  Running in DEMO MODE - predictions will be simulated")
        DEMO_MODE = True

# Load model on startup
load_model()

# Feature metadata - the 10 most important features
FEATURES = {
    'number_inpatient': {
        'type': 'integer',
        'min': 0,
        'max': 20,
        'description': 'Número de veces que el paciente fue hospitalizado como paciente interno en el año anterior. Un mayor número de hospitalizaciones previas está fuertemente asociado con mayor riesgo de readmisión, ya que indica condiciones médicas complejas o crónicas.',
        'short_label': 'Visitas Hospitalarias'
    },
    'discharge_disposition_id': {
        'type': 'integer',
        'min': 1,
        'max': 30,
        'description': 'Código que indica el destino del paciente al momento del alta hospitalaria (ej: casa, centro de rehabilitación, otro hospital). Valores más altos pueden indicar situaciones de mayor complejidad que aumentan el riesgo de readmisión.',
        'short_label': 'Tipo de Alta'
    },
    'number_emergency': {
        'type': 'integer',
        'min': 0,
        'max': 20,
        'description': 'Número de visitas a la sala de emergencias en el año anterior. Múltiples visitas de emergencia sugieren condiciones médicas inestables o mal controladas, lo que incrementa significativamente el riesgo de readmisión.',
        'short_label': 'Visitas de Emergencia'
    },
    'medical_specialty_Psychiatry': {
        'type': 'binary',
        'min': 0,
        'max': 1,
        'description': 'Indica si el médico tratante es especialista en Psiquiatría. Los pacientes con condiciones psiquiátricas pueden tener mayor complejidad en el manejo y adherencia al tratamiento, afectando el riesgo de readmisión.',
        'short_label': 'Especialidad: Psiquiatría'
    },
    'diag_1_group_Musculoskeletal': {
        'type': 'binary',
        'min': 0,
        'max': 1,
        'description': 'El diagnóstico principal pertenece al grupo musculoesquelético (artritis, fracturas, problemas de columna, etc.). Estas condiciones pueden requerir seguimiento prolongado y tienen patrones específicos de readmisión.',
        'short_label': 'Diagnóstico: Musculoesquelético'
    },
    'diag_2_group_Neoplasms': {
        'type': 'binary',
        'min': 0,
        'max': 1,
        'description': 'El diagnóstico secundario incluye neoplasmas (tumores benignos o malignos). La presencia de cáncer como condición secundaria aumenta significativamente la complejidad del caso y el riesgo de readmisión.',
        'short_label': 'Diagnóstico Secundario: Neoplasmas'
    },
    'medical_specialty_Oncology': {
        'type': 'binary',
        'min': 0,
        'max': 1,
        'description': 'Indica si el médico tratante es especialista en Oncología. Los pacientes oncológicos requieren cuidados especializados y tienen mayor riesgo de complicaciones que pueden llevar a readmisión.',
        'short_label': 'Especialidad: Oncología'
    },
    'medical_specialty_PhysicalMedicineandRehabilitation': {
        'type': 'binary',
        'min': 0,
        'max': 1,
        'description': 'Indica si el médico tratante es especialista en Medicina Física y Rehabilitación. Estos pacientes suelen tener condiciones que requieren recuperación prolongada y seguimiento continuo.',
        'short_label': 'Especialidad: Medicina Física y Rehabilitación'
    },
    'insulin_Down': {
        'type': 'binary',
        'min': 0,
        'max': 1,
        'description': 'Indica si la dosis de insulina fue reducida durante la hospitalización. Los cambios en la medicación para diabetes pueden afectar el control glucémico y aumentar el riesgo de complicaciones y readmisión.',
        'short_label': 'Dosis de Insulina Reducida'
    },
    'diag_1_group_Circulatory': {
        'type': 'binary',
        'min': 0,
        'max': 1,
        'description': 'El diagnóstico principal pertenece al sistema circulatorio (enfermedades cardíacas, hipertensión, problemas vasculares). Estas condiciones son altamente prevalentes en pacientes diabéticos y están asociadas con mayor riesgo de readmisión.',
        'short_label': 'Diagnóstico: Circulatorio'
    }
}

FEATURE_ORDER = [
    'number_inpatient',
    'discharge_disposition_id',
    'number_emergency',
    'medical_specialty_Psychiatry',
    'diag_1_group_Musculoskeletal',
    'diag_2_group_Neoplasms',
    'medical_specialty_Oncology',
    'medical_specialty_PhysicalMedicineandRehabilitation',
    'insulin_Down',
    'diag_1_group_Circulatory'
]


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None or pipeline is not None,
        'demo_mode': DEMO_MODE
    })


@app.route('/api/features')
def get_features():
    """Return feature metadata"""
    return jsonify({
        'features': FEATURES,
        'feature_order': FEATURE_ORDER
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict patient readmission risk
    
    Expected JSON payload:
    {
        "number_inpatient": 1,
        "discharge_disposition_id": 1,
        "number_emergency": 0,
        "medical_specialty_Psychiatry": 0,
        "diag_1_group_Musculoskeletal": 0,
        "diag_2_group_Neoplasms": 0,
        "medical_specialty_Oncology": 0,
        "medical_specialty_PhysicalMedicineandRehabilitation": 0,
        "insulin_Down": 0,
        "diag_1_group_Circulatory": 1
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate all required features are present
        missing_features = [f for f in FEATURE_ORDER if f not in data]
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing': missing_features
            }), 400
        
        # Validate feature values
        errors = []
        for feature_name in FEATURE_ORDER:
            value = data[feature_name]
            feature_meta = FEATURES[feature_name]
            
            # Check type
            if not isinstance(value, (int, float)):
                errors.append(f"{feature_name} must be a number")
                continue
            
            # Check range
            if value < feature_meta['min'] or value > feature_meta['max']:
                errors.append(
                    f"{feature_name} must be between {feature_meta['min']} and {feature_meta['max']}"
                )
        
        if errors:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400
        
        # Prepare input array in correct order
        input_array = np.array([[data[f] for f in FEATURE_ORDER]])
        
        # Make prediction
        if DEMO_MODE:
            # Demo mode: simulate prediction based on input features
            # Calculate a risk score based on the features
            risk_score = (
                data['number_inpatient'] * 0.15 +
                data['number_emergency'] * 0.13 +
                (data['discharge_disposition_id'] > 5) * 0.1 +
                data['medical_specialty_Psychiatry'] * 0.1 +
                data['diag_1_group_Musculoskeletal'] * 0.05 +
                data['diag_2_group_Neoplasms'] * 0.1 +
                data['medical_specialty_Oncology'] * 0.1 +
                data['medical_specialty_PhysicalMedicineandRehabilitation'] * 0.05 +
                data['insulin_Down'] * 0.1 +
                data['diag_1_group_Circulatory'] * 0.12
            )
            
            # Normalize to 0-1 range
            high_risk_prob = min(max(risk_score / 2.0, 0.1), 0.9)
            low_risk_prob = 1.0 - high_risk_prob
            
            prediction = 1 if high_risk_prob > 0.5 else 0
            probability = np.array([low_risk_prob, high_risk_prob])
            
        elif pipeline is not None:
            prediction = pipeline.predict(input_array)[0]
            probability = pipeline.predict_proba(input_array)[0]
        elif model is not None:
            prediction = model.predict(input_array)[0]
            probability = model.predict_proba(input_array)[0]
        else:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'prediction_label': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': {
                'low_risk': float(probability[0]),
                'high_risk': float(probability[1])
            },
            'risk_percentage': float(probability[1] * 100),
            'input_features': data,
            'demo_mode': DEMO_MODE
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 Patient Readmission Prediction API")
    print("="*60)
    print(f"Model loaded: {model is not None or pipeline is not None}")
    if DEMO_MODE:
        print("⚠️  DEMO MODE: Using simulated predictions")
    print(f"Number of features: {len(FEATURE_ORDER)} (Top 10 most important)")
    print(f"\nStarting server on http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
