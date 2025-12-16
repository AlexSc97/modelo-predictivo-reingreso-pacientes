import pickle
import pandas as pd
from pathlib import Path

MODEL_PATH = Path('models/modelo_produccion.pkl')

print(f"Inspeccionando modelo en: {MODEL_PATH.absolute()}")

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print("Modelo cargado exitosamente.")
    
    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
        preprocessor = model.named_steps['preprocessor']
        print("\nTransformadores en el preprocesador:")
        for name, trans, cols in preprocessor.transformers_:
            print(f"  - {name}: {cols}")
            
    else:
        print("El modelo no tiene la estructura esperada (pipeline con preprocessor).")
        print(f"Tipo de modelo: {type(model)}")

except Exception as e:
    print(f"Error cargando modelo: {e}")
