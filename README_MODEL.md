# Modelo Predictivo de Reingreso de Pacientes (XGBoost)

Este directorio contiene el modelo XGBoost optimizado para predecir el reingreso de pacientes.

## Archivos

- `src/train_xgboost.py`: Script de Python para entrenar, optimizar y guardar el modelo. Este script reemplaza la lógica del notebook `features_xgboost.ipynb` para un flujo de trabajo más robusto y listo para producción.
- `models/xgboost_model.joblib`: El modelo entrenado y guardado (Pipeline completo con preprocesamiento).
- `models/confusion_matrix.png`: Gráfico de la matriz de confusión del modelo actual.

## Cómo entrenar el modelo

Para re-entrenar el modelo, ejecuta el siguiente comando desde la carpeta `src`:

```bash
python train_xgboost.py
```

Esto realizará lo siguiente:
1. Cargar los datos procesados.
2. Seleccionar las 10 mejores características.
3. Optimizar hiperparámetros con Optuna (maximizando AUC).
4. Entrenar el modelo final.
5. Guardar el modelo en `models/xgboost_model.joblib`.

## Integración con Flask

Para usar este modelo en una aplicación Flask, puedes cargarlo de la siguiente manera:

```python
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Cargar el modelo al iniciar la app
MODEL_PATH = 'models/xgboost_model.joblib'
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Convertir JSON a DataFrame
        # Asegúrate de que el JSON tenga las claves correspondientes a las características
        df = pd.DataFrame([data])
        
        # Realizar predicción (el pipeline se encarga del preprocesamiento)
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]
        
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0])
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## Características Requeridas

El modelo espera las siguientes columnas en los datos de entrada:
- `number_inpatient`
- `discharge_segment`
- `diabetesMed`
- `number_emergency`
- `metformin`
- `diag_1_group`
- `diag_2_group`
- `number_diagnoses`
- `age`
- `time_in_hospital`
