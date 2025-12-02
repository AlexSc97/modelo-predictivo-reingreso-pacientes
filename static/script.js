const API_BASE_URL = '';

// DOM Elements
const form = document.getElementById('predictionForm');
const steps = document.querySelectorAll('.wizard-step');
const stepIndicators = document.querySelectorAll('.step');
const nextBtn = document.getElementById('nextBtn');
const prevBtn = document.getElementById('prevBtn');
const submitBtn = document.getElementById('submitBtn');
const formPanel = document.getElementById('formPanel');
const resultsPanel = document.getElementById('resultsPanel');
const backToFormBtn = document.getElementById('backToForm');
const resetBtn = document.getElementById('resetBtn'); // The new button in results

let currentStep = 0;
let FEATURES_METADATA = {};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateWizard();
    setupEventListeners();
    loadFeatureMetadata();
});

async function loadFeatureMetadata() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/features`);
        const data = await response.json();
        FEATURES_METADATA = data.features;
        initializeTooltips();
    } catch (error) {
        console.error('Error loading feature metadata:', error);
    }
}

function initializeTooltips() {
    // Map input IDs to feature names if they differ, but here they are mostly same
    // We need to attach tooltips to the .input-group containers

    const inputGroups = document.querySelectorAll('.input-group');

    inputGroups.forEach(group => {
        // Find the input/select inside to get the ID
        const input = group.querySelector('input, select');
        if (!input) return;

        const featureName = input.id || input.name;
        const meta = FEATURES_METADATA[featureName];

        if (meta) {
            // Create Tooltip Element
            const tooltip = document.createElement('div');
            tooltip.className = 'feature-tooltip';
            tooltip.innerHTML = `
                <div class="tooltip-header">Impacto del Modelo</div>
                <div class="tooltip-impact">${meta.importance ? meta.importance.toFixed(1) : 0}%</div>
                <div class="tooltip-desc">${meta.description}</div>
            `;

            group.appendChild(tooltip);

            // Events
            group.addEventListener('mouseenter', () => {
                tooltip.style.display = 'block';
            });

            group.addEventListener('mouseleave', () => {
                tooltip.style.display = 'none';
            });
        }
    });
}

function setupEventListeners() {
    // Navigation
    nextBtn.addEventListener('click', () => {
        if (validateStep(currentStep)) {
            currentStep++;
            updateWizard();
        }
    });

    prevBtn.addEventListener('click', () => {
        if (currentStep > 0) {
            currentStep--;
            updateWizard();
        }
    });

    // Form Submit
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handlePrediction();
    });

    // Back to Form (Icon)
    backToFormBtn.addEventListener('click', resetApp);

    // Reset Button (Main)
    if (resetBtn) {
        resetBtn.addEventListener('click', resetApp);
    }

    // Toggle Switch Logic for Diabetes Med
    const diabetesToggle = document.getElementById('diabetesMed_toggle');
    const diabetesInput = document.getElementById('diabetesMed');

    if (diabetesToggle) {
        diabetesToggle.addEventListener('change', (e) => {
            diabetesInput.value = e.target.checked ? 'Yes' : 'No';
        });
    }

    // Logic for No Secondary Diagnosis
    const noDiag2Checkbox = document.getElementById('no_diag_2_checkbox');
    const diag2Select = document.getElementById('diag_2_group');

    if (noDiag2Checkbox && diag2Select) {
        noDiag2Checkbox.addEventListener('change', (e) => {
            const wrapper = diag2Select.closest('.select-wrapper-animated');
            if (e.target.checked) {
                diag2Select.disabled = true;
                if (wrapper) wrapper.classList.add('disabled-input');
            } else {
                diag2Select.disabled = false;
                if (wrapper) wrapper.classList.remove('disabled-input');
            }
        });
    }
}

function resetApp() {
    resultsPanel.style.display = 'none';
    formPanel.style.display = 'flex';
    form.reset();
    currentStep = 0;
    updateWizard();

    // Reset sliders outputs
    document.querySelectorAll('output').forEach(out => {
        // Find corresponding input to get min value or default
        const input = out.previousElementSibling.querySelector('input');
        if (input) out.textContent = input.min || 0;
    });

    // Reset Diag 2 state
    const diag2Select = document.getElementById('diag_2_group');
    if (diag2Select) {
        diag2Select.disabled = false;
        diag2Select.style.opacity = '1';
        const wrapper = diag2Select.closest('.select-wrapper-animated');
        if (wrapper) wrapper.classList.remove('disabled-input');
    }
}

function updateWizard() {
    // Update Steps Visibility
    steps.forEach((step, index) => {
        if (index === currentStep) {
            step.classList.add('active');
        } else {
            step.classList.remove('active');
        }
    });

    // Update Sidebar Indicators
    stepIndicators.forEach((indicator, index) => {
        const stepNum = parseInt(indicator.dataset.step) - 1;
        if (stepNum === currentStep) {
            indicator.classList.add('active');
            indicator.classList.remove('completed');
        } else if (stepNum < currentStep) {
            indicator.classList.add('completed');
            indicator.classList.remove('active');
        } else {
            indicator.classList.remove('active', 'completed');
        }
    });

    // Update Buttons
    prevBtn.disabled = currentStep === 0;

    if (currentStep === steps.length - 1) {
        nextBtn.style.display = 'none';
        submitBtn.style.display = 'flex';
    } else {
        nextBtn.style.display = 'block';
        submitBtn.style.display = 'none';
    }
}

function validateStep(stepIndex) {
    const currentStepEl = steps[stepIndex];
    const inputs = currentStepEl.querySelectorAll('input[required], select[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value) {
            isValid = false;
            input.classList.add('error');
        } else {
            input.classList.remove('error');
        }
    });

    return isValid;
}

async function handlePrediction() {
    const originalBtnContent = submitBtn.innerHTML;
    submitBtn.innerHTML = '<span class="material-icons-round spin">sync</span> Analizando...';
    submitBtn.disabled = true;

    try {
        const formData = collectFormData();
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Error en predicción');

        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        alert('Error al procesar la solicitud: ' + error.message);
    } finally {
        submitBtn.innerHTML = originalBtnContent;
        submitBtn.disabled = false;
    }
}

function collectFormData() {
    const data = {};
    const formData = new FormData(form);

    for (let [key, value] of formData.entries()) {
        if (['number_inpatient', 'number_emergency', 'time_in_hospital', 'number_diagnoses', 'age'].includes(key)) {
            data[key] = parseInt(value);
        } else {
            data[key] = value;
        }
    }

    // Handle No Secondary Diagnosis Checkbox
    const noDiag2Checkbox = document.getElementById('no_diag_2_checkbox');
    if (noDiag2Checkbox && noDiag2Checkbox.checked) {
        data['diag_2_group'] = 'None';
    }

    return data;
}

function displayResults(data) {
    formPanel.style.display = 'none';
    resultsPanel.style.display = 'flex';

    const riskPercent = data.risk_percentage;
    const circle = document.querySelector('.circle-fill');
    const riskText = document.getElementById('riskPercentage');
    const riskStatus = document.getElementById('riskStatus');

    setTimeout(() => {
        circle.setAttribute('stroke-dasharray', `${riskPercent}, 100`);

        const outcomeElement = document.getElementById('predictionOutcome');
        // Usamos la probabilidad de alto riesgo devuelta por la API
        if (data.probability.high_risk >= 0.51) {
            outcomeElement.textContent = "PREDICCIÓN: EL PACIENTE SERÁ READMITIDO";
            outcomeElement.style.color = 'var(--danger)';
        } else {
            outcomeElement.textContent = "PREDICCIÓN: EL PACIENTE NO SERÁ READMITIDO";
            outcomeElement.style.color = 'var(--success)';
        }

        if (data.risk_level === 'high') {
            circle.style.stroke = 'var(--danger)';
            riskStatus.style.color = 'var(--danger)';
            riskStatus.textContent = 'Riesgo Alto';
        } else if (data.risk_level === 'medium') {
            circle.style.stroke = 'var(--warning)';
            riskStatus.style.color = 'var(--warning)';
            riskStatus.textContent = 'Riesgo Moderado';
        } else {
            circle.style.stroke = 'var(--success)';
            riskStatus.style.color = 'var(--success)';
            riskStatus.textContent = 'Riesgo Bajo';
        }
    }, 100);

    animateValue(riskText, 0, riskPercent, 1500);

    const impactList = document.getElementById('impactList');
    impactList.innerHTML = '';

    if (data.top_features) {
        // Normalize based on the highest impact for relative bar sizing
        const maxImpact = Math.max(...data.top_features.map(f => f.abs_impact));

        data.top_features.slice(0, 5).forEach((feature, index) => {
            const width = (feature.abs_impact / maxImpact) * 100;
            const item = document.createElement('div');
            item.className = 'factor-item';

            // Determine color based on impact direction (if available) or just use primary
            // Assuming positive impact increases risk (danger), negative decreases (success)
            // If impact is absolute, we might just use primary color.
            // Let's use a gradient or specific color.
            const barColor = feature.impact > 0 ? 'var(--danger)' : 'var(--success)';

            item.innerHTML = `
                <div class="factor-header">
                    <span>${feature.feature}</span>
                    <span class="factor-value" style="color: ${barColor}">${feature.impact.toFixed(2)}</span>
                </div>
                <div class="factor-bar-bg">
                    <div class="factor-bar-fill" style="width: 0%; background: ${barColor}"></div>
                </div>
            `;
            impactList.appendChild(item);

            // Animate bar width
            setTimeout(() => {
                const bar = item.querySelector('.factor-bar-fill');
                bar.style.width = `${width}%`;
            }, 200 + (index * 100)); // Staggered animation
        });
    }

    const recText = document.getElementById('recommendationText');
    if (data.risk_level === 'high') {
        recText.innerHTML = "<strong>Atención Inmediata:</strong> El modelo indica una alta probabilidad de reingreso. Se recomienda encarecidamente revisar el plan de medicación, considerar un ingreso hospitalario preventivo o establecer un monitoreo domiciliario intensivo. Verificar adherencia al tratamiento y soporte social.";
    } else if (data.risk_level === 'medium') {
        recText.innerHTML = "<strong>Precaución:</strong> Existe un riesgo moderado. Se sugiere programar una cita de seguimiento en los próximos 7 días y reforzar la educación del paciente sobre los signos de alarma. Revisar posibles interacciones medicamentosas.";
    } else {
        recText.innerHTML = "<strong>Bajo Riesgo:</strong> El paciente puede continuar con el plan de alta estándar. Se recomienda mantener las citas de control habituales y fomentar hábitos de vida saludables. Riesgo bajo de readmisión en 30 días.";
    }

    // Cargar gráfico SHAP
    loadShapPlot(data.input_features);
}

async function loadShapPlot(inputFeatures) {
    const shapContainer = document.getElementById('shapPlotContainer');

    try {
        const response = await fetch(`${API_BASE_URL}/api/shap_plot`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputFeatures)
        });

        const shapData = await response.json();

        if (!response.ok) throw new Error(shapData.error || 'Error generando SHAP plot');

        // Insertar HTML del gráfico con animación
        shapContainer.innerHTML = shapData.html;
        shapContainer.classList.add('loaded');

    } catch (error) {
        console.error('Error cargando SHAP plot:', error);
        shapContainer.innerHTML = `
            <div class="error-message">
                <span class="material-icons-round">error_outline</span>
                <p>No se pudo generar el gráfico SHAP. ${error.message}</p>
            </div>
        `;
    }
}

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start) + "%";
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
