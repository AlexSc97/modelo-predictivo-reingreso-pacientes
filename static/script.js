// API Configuration
const API_BASE_URL = 'http://localhost:5001';

// DOM Elements
const form = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const resetBtn = document.getElementById('resetBtn');
const resultsCard = document.getElementById('resultsCard');
const errorCard = document.getElementById('errorCard');

// Feature names in order
const FEATURE_NAMES = [
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
];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('🏥 Patient Readmission Prediction App Initialized');
    checkAPIHealth();
});

// Check API Health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API Health Check Failed:', error);
        showError('No se puede conectar con el servidor. Asegúrese de que la API esté ejecutándose.');
    }
}

// Form Submit Handler
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Hide previous results/errors
    resultsCard.style.display = 'none';
    errorCard.style.display = 'none';

    // Show loading state
    submitBtn.classList.add('loading');

    try {
        // Collect form data
        const formData = collectFormData();
        console.log('Sending prediction request:', formData);

        // Make prediction request
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Error en la predicción');
        }

        console.log('Prediction result:', data);
        displayResults(data);

    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'Error al realizar la predicción. Por favor, intente nuevamente.');
    } finally {
        submitBtn.classList.remove('loading');
    }
});

// Reset Button Handler
resetBtn.addEventListener('click', () => {
    form.reset();
    resultsCard.style.display = 'none';
    errorCard.style.display = 'none';
});

// Collect Form Data
function collectFormData() {
    const data = {};

    FEATURE_NAMES.forEach(featureName => {
        const element = document.getElementById(featureName);

        if (element.type === 'checkbox') {
            data[featureName] = element.checked ? 1 : 0;
        } else if (element.type === 'number') {
            data[featureName] = parseFloat(element.value) || 0;
        }
    });

    return data;
}

// Display Results
function displayResults(data) {
    const isHighRisk = data.prediction === 1;
    const riskPercentage = data.risk_percentage.toFixed(1);
    const lowRiskPercentage = (data.probability.low_risk * 100).toFixed(1);
    const highRiskPercentage = (data.probability.high_risk * 100).toFixed(1);

    // Update risk indicator
    const riskIndicator = document.getElementById('riskIndicator');
    const riskIcon = document.getElementById('riskIcon');
    const riskLabel = document.getElementById('riskLabel');
    const riskPercentageEl = document.getElementById('riskPercentage');

    riskIndicator.className = 'risk-indicator ' + (isHighRisk ? 'high-risk' : 'low-risk');
    riskIcon.textContent = isHighRisk ? '⚠️' : '✅';
    riskLabel.textContent = data.prediction_label;
    riskPercentageEl.textContent = riskPercentage + '%';

    // Update probability bars
    document.getElementById('lowRiskValue').textContent = lowRiskPercentage + '%';
    document.getElementById('highRiskValue').textContent = highRiskPercentage + '%';

    const lowRiskBar = document.getElementById('lowRiskBar');
    const highRiskBar = document.getElementById('highRiskBar');

    // Animate bars
    setTimeout(() => {
        lowRiskBar.style.width = lowRiskPercentage + '%';
        highRiskBar.style.width = highRiskPercentage + '%';
    }, 100);

    // Update recommendation
    const recommendationText = document.getElementById('recommendationText');
    if (isHighRisk) {
        recommendationText.innerHTML = `
            <strong>El paciente presenta un riesgo alto de readmisión (${riskPercentage}%).</strong><br><br>
            Se recomienda:
            <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                <li>Seguimiento médico cercano post-alta</li>
                <li>Educación al paciente sobre manejo de su condición</li>
                <li>Coordinación con atención primaria</li>
                <li>Considerar programas de transición de cuidados</li>
            </ul>
        `;
    } else {
        recommendationText.innerHTML = `
            <strong>El paciente presenta un riesgo bajo de readmisión (${riskPercentage}%).</strong><br><br>
            Se recomienda:
            <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                <li>Seguimiento estándar post-alta</li>
                <li>Instrucciones claras de alta médica</li>
                <li>Contacto de seguimiento programado</li>
            </ul>
        `;
    }

    // Show results card with animation
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show Error
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorCard.style.display = 'block';
    errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Input Validation
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', (e) => {
        const min = parseFloat(e.target.min);
        const max = parseFloat(e.target.max);
        const value = parseFloat(e.target.value);

        if (value < min) e.target.value = min;
        if (value > max) e.target.value = max;
    });
});

// Add visual feedback for checkboxes
document.querySelectorAll('.checkbox-label').forEach(label => {
    label.addEventListener('click', () => {
        // Add a subtle animation
        label.style.transform = 'scale(0.98)';
        setTimeout(() => {
            label.style.transform = 'scale(1)';
        }, 100);
    });
});

console.log('✅ Script loaded successfully');
