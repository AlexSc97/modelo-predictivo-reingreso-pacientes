# 🏥 Predicción de Reingreso Hospitalario en Pacientes Diabéticos

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Random Forest](https://img.shields.io/badge/Model-Random_Forest-green)
![SMOTE](https://img.shields.io/badge/Imbalanced_Data-SMOTE-orange)
![Status](https://img.shields.io/badge/Focus-Healthcare_KPI-red)

Este proyecto aborda uno de los problemas más costosos en la gestión hospitalaria: el **reingreso de pacientes** (Readmission) antes de los 30 días. Utilizando un dataset clínico de 10 años (1999-2008), se desarrolló un modelo predictivo para identificar pacientes de alto riesgo y optimizar la asignación de recursos médicos.

## 🎯 Contexto y Problema de Negocio
Un reingreso hospitalario temprano suele indicar una falla en el tratamiento inicial o en el seguimiento post-alta.
* **Objetivo:** Predecir si un paciente diabético será readmitido en menos de 30 días.
* **Impacto:** Permitir al personal médico intervenir preventivamente en pacientes de alto riesgo antes de darles el alta.

## ⚙️ Metodología Técnica y Clínica

### 1. Limpieza con Criterio Médico
Se realizó un preprocesamiento riguroso guiado por lógica clínica:
* **Filtrado de Cohorte:** Se excluyeron registros de pacientes con alta por fallecimiento o traslado a hospicio (*Hospice*), ya que el reingreso es imposible en estos casos.
* **Manejo de Datos Faltantes:** Eliminación de variables con >50% de nulidad (`weight`, `payer_code`) y imputación estratégica.

### 2. Ingeniería de Características (Feature Engineering)
* **Agrupación de CIE-9:** Se simplificaron cientos de códigos de diagnóstico en categorías manejables.
* **Historial del Paciente:** Se dio peso a variables como `number_inpatient` (visitas previas) y `time_in_hospital`.
* **Interacción de Medicamentos:** Análisis de cambios en la medicación (`change`) y uso de insulina.

### 3. Manejo de Desbalance de Clases (SMOTE) ⚖️
Dado que los casos de reingreso positivo eran minoría, se aplicó **SMOTE (Synthetic Minority Over-sampling Technique)**.
* Esto generó datos sintéticos para la clase minoritaria, evitando que el modelo tuviera un sesgo hacia la clase mayoritaria (No Reingreso) y mejorando la sensibilidad del diagnóstico.

### 4. Modelado
Se implementó un **Random Forest Classifier** optimizado mediante **GridSearchCV**.
* **Métrica Clave:** Se priorizó el **Recall (Sensibilidad)** sobre el Accuracy, ya que en medicina es más costoso no detectar a un paciente en riesgo (Falso Negativo) que revisar a uno sano (Falso Positivo).

## 📊 Resultados y Hallazgos
El análisis de importancia de características (*Feature Importance*) reveló los predictores más fuertes:
1.  **`number_inpatient` (Visitas hospitalarias previas):** El predictor #1. Pacientes con historial de ingresos recientes tienen una probabilidad drásticamente mayor de volver.
2.  **`num_lab_procedures`:** Indica la complejidad del estado de salud del paciente.
3.  **`discharge_disposition_id`:** El lugar a donde se envía al paciente (casa, otra clínica) influye en el riesgo.

## 🛠️ Stack Tecnológico
* **Procesamiento:** Pandas, NumPy.
* **Machine Learning:** Scikit-Learn (Random Forest, GridSearchCV).
* **Técnicas Avanzadas:** Imbalanced-learn (SMOTE).
* **Visualización:** Seaborn, Matplotlib.

## 📂 Estructura
```text
├── data/                # Dataset clínico
├── notebooks/           # Notebook con EDA, SMOTE y Modelado
├── models/              # Modelo Random Forest serializado
└── README.md
