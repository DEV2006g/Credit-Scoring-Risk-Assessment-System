/*
  Credit Scoring & Risk Assessment System
  Interactive Frontend Logic: Scorecard Engine, Dynamic SVG Gauge, Presets & What-If Sandbox
*/

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('creditForm');
    const initialState = document.getElementById('initialState');
    const resultCard = document.getElementById('resultCard');

    // Preset Profiles
    const presets = {
        prime: {
            checking_status: 'A13',
            duration: 12,
            credit_history: 'A32',
            purpose: 'A40',
            credit_amount: 2500,
            savings_status: 'A64',
            employment: 'A75',
            installment_rate: 2,
            personal_status: 'A93',
            other_parties: 'A101',
            residence_since: 4,
            property_magnitude: 'A121',
            age: 45,
            other_payment_plans: 'A143',
            housing: 'A152',
            existing_credits: 2,
            job: 'A173',
            num_dependents: 1,
            own_telephone: 'A192',
            foreign_worker: 'A201'
        },
        fair: {
            checking_status: 'A12',
            duration: 24,
            credit_history: 'A32',
            purpose: 'A42',
            credit_amount: 4200,
            savings_status: 'A62',
            employment: 'A73',
            installment_rate: 3,
            personal_status: 'A92',
            other_parties: 'A101',
            residence_since: 2,
            property_magnitude: 'A122',
            age: 32,
            other_payment_plans: 'A143',
            housing: 'A151',
            existing_credits: 1,
            job: 'A173',
            num_dependents: 1,
            own_telephone: 'A191',
            foreign_worker: 'A201'
        },
        subprime: {
            checking_status: 'A11',
            duration: 48,
            credit_history: 'A34',
            purpose: 'A41',
            credit_amount: 8500,
            savings_status: 'A61',
            employment: 'A71',
            installment_rate: 4,
            personal_status: 'A93',
            other_parties: 'A101',
            residence_since: 1,
            property_magnitude: 'A124',
            age: 23,
            other_payment_plans: 'A141',
            housing: 'A151',
            existing_credits: 2,
            job: 'A172',
            num_dependents: 1,
            own_telephone: 'A191',
            foreign_worker: 'A201'
        }
    };

    function loadPreset(profileKey) {
        const p = presets[profileKey];
        if (!p) return;

        for (const [key, val] of Object.entries(p)) {
            const input = document.getElementById(key);
            if (input) {
                input.value = val;
            }
        }
    }

    // Bind Preset Buttons
    const primeBtn = document.getElementById('presetPrimeBtn');
    const fairBtn = document.getElementById('presetFairBtn');
    const subprimeBtn = document.getElementById('presetSubprimeBtn');

    if (primeBtn) primeBtn.addEventListener('click', () => loadPreset('prime'));
    if (fairBtn) fairBtn.addEventListener('click', () => loadPreset('fair'));
    if (subprimeBtn) subprimeBtn.addEventListener('click', () => loadPreset('subprime'));

    let currentBaseApplicant = null;

    // Form Submission & Risk Assessment
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();

            const formData = new FormData(form);
            const applicantData = {};
            formData.forEach((value, key) => {
                applicantData[key] = value;
            });

            currentBaseApplicant = { ...applicantData };

            const submitBtn = document.getElementById('submitScoreBtn');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Evaluating Credit Profile...';

            fetch('/api/score', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(applicantData)
            })
            .then(res => res.json())
            .then(response => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnText;

                if (response.error) {
                    alert('Evaluation error: ' + response.error);
                    return;
                }

                displayScorecard(response.data);
                initWhatIfControls(applicantData);
            })
            .catch(err => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnText;
                alert('Network error: ' + err.message);
            });
        });
    }

    function displayScorecard(data) {
        if (initialState) initialState.style.display = 'none';
        if (resultCard) resultCard.style.display = 'block';

        // 1. Update Score Display & Gauge
        const scoreValElem = document.getElementById('scoreValue');
        const score = data.credit_score;
        scoreValElem.textContent = score;

        // Animated Radial Gauge calculation:
        // Score scale: 300 to 850 (range: 550)
        // Angle range: -90 deg to +90 deg (total 180 deg)
        const scoreFraction = Math.max(0, Math.min(1, (score - 300) / 550));
        const rotationAngle = -90 + (scoreFraction * 180);
        
        const needle = document.getElementById('gaugeNeedle');
        if (needle) {
            needle.style.transform = `rotate(${rotationAngle}deg)`;
        }

        const meter = document.getElementById('gaugeMeter');
        if (meter) {
            // Arc length: ~235.6
            const totalDash = 235.6;
            const offset = totalDash - (scoreFraction * totalDash);
            meter.style.strokeDashoffset = offset;
            meter.style.stroke = data.color;
        }

        // 2. Risk Tier Badge
        const tierBadge = document.getElementById('riskTierBadge');
        if (tierBadge) {
            tierBadge.textContent = `${data.risk_tier} Tier`;
            tierBadge.className = `badge badge-lg ${data.badge_class}`;
        }

        // 3. Decision Banner
        const banner = document.getElementById('decisionBanner');
        const decisionText = document.getElementById('decisionText');
        const decisionSummary = document.getElementById('decisionSummary');
        const decisionIcon = document.getElementById('decisionIcon');

        if (banner && decisionText && decisionSummary) {
            decisionText.textContent = `Decision: ${data.decision}`;
            decisionSummary.textContent = data.summary;

            banner.className = 'decision-banner';
            if (data.decision === 'Approved') {
                banner.classList.add('banner-approved');
                decisionIcon.innerHTML = '<i class="fa-solid fa-circle-check text-success"></i>';
            } else if (data.decision.includes('Conditional') || data.decision.includes('Manual')) {
                banner.classList.add('banner-conditional');
                decisionIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation text-warning"></i>';
            } else {
                banner.classList.add('banner-declined');
                decisionIcon.innerHTML = '<i class="fa-solid fa-circle-xmark text-danger"></i>';
            }
        }

        // 4. Metric Tiles
        document.getElementById('pdValue').textContent = `${data.default_probability}%`;
        document.getElementById('aprValue').textContent = data.suggested_apr;
        document.getElementById('limitValue').textContent = data.max_approved_limit > 0 ? `${data.max_approved_limit.toLocaleString()} DM` : 'N/A';
        document.getElementById('costDecision').textContent = data.cost_optimized_decision;

        // 5. Positive Factors
        const posList = document.getElementById('positiveDriversList');
        posList.innerHTML = '';
        data.positive_factors.forEach(item => {
            const li = document.createElement('li');
            li.className = 'factor-item pos';
            li.innerHTML = `
                <span class="factor-title text-success"><i class="fa-solid fa-check"></i> ${item.feature}</span>
                <span class="factor-detail">${item.detail}</span>
            `;
            posList.appendChild(li);
        });

        // 6. Risk Flags
        const flagList = document.getElementById('riskFlagsList');
        flagList.innerHTML = '';
        data.risk_flags.forEach(item => {
            const li = document.createElement('li');
            li.className = 'factor-item neg';
            li.innerHTML = `
                <span class="factor-title text-danger"><i class="fa-solid fa-triangle-exclamation"></i> ${item.feature}</span>
                <span class="factor-detail">${item.detail}</span>
            `;
            flagList.appendChild(li);
        });

        // 7. Recommendations
        const recsList = document.getElementById('recsList');
        recsList.innerHTML = '';
        data.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recsList.appendChild(li);
        });

        // 8. Timestamp
        const now = new Date();
        document.getElementById('decisionTimestamp').textContent = `Assessed at ${now.toLocaleTimeString()} (${now.toLocaleDateString()})`;
    }

    // What-If Simulation Controls
    const simDuration = document.getElementById('simDuration');
    const simAmount = document.getElementById('simAmount');
    const simGuarantor = document.getElementById('simGuarantor');
    const runSimBtn = document.getElementById('runSimulationBtn');

    const simDurationVal = document.getElementById('simDurationVal');
    const simAmountVal = document.getElementById('simAmountVal');
    const simDeltaBox = document.getElementById('simDeltaBox');

    function initWhatIfControls(applicant) {
        if (!simDuration || !simAmount) return;

        simDuration.value = applicant.duration || 24;
        simDurationVal.textContent = `${simDuration.value} Months`;

        simAmount.value = applicant.credit_amount || 3000;
        simAmountVal.textContent = `${Number(simAmount.value).toLocaleString()} DM`;

        if (simGuarantor) {
            simGuarantor.value = applicant.other_parties || 'A101';
        }

        if (simDeltaBox) simDeltaBox.style.display = 'none';
    }

    if (simDuration) {
        simDuration.addEventListener('input', () => {
            simDurationVal.textContent = `${simDuration.value} Months`;
        });
    }

    if (simAmount) {
        simAmount.addEventListener('input', () => {
            simAmountVal.textContent = `${Number(simAmount.value).toLocaleString()} DM`;
        });
    }

    if (runSimBtn) {
        runSimBtn.addEventListener('click', () => {
            if (!currentBaseApplicant) return;

            const modifications = {
                duration: parseFloat(simDuration.value),
                credit_amount: parseFloat(simAmount.value),
                other_parties: simGuarantor ? simGuarantor.value : currentBaseApplicant.other_parties
            };

            fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    base: currentBaseApplicant,
                    modifications: modifications
                })
            })
            .then(res => res.json())
            .then(res => {
                if (res.error) {
                    alert('Simulation error: ' + res.error);
                    return;
                }
                displaySimulationResult(res.data);
            })
            .catch(err => {
                alert('Simulation error: ' + err.message);
            });
        });
    }

    function displaySimulationResult(simData) {
        if (!simDeltaBox) return;
        simDeltaBox.style.display = 'flex';

        document.getElementById('simScoreValue').textContent = simData.simulated_score;
        const scoreDeltaElem = document.getElementById('simScoreDelta');
        const scoreDiff = simData.score_change;
        scoreDeltaElem.textContent = scoreDiff >= 0 ? `+${scoreDiff} pts` : `${scoreDiff} pts`;
        scoreDeltaElem.className = `sim-delta ${scoreDiff >= 0 ? 'delta-pos' : 'delta-neg'}`;

        document.getElementById('simPdValue').textContent = `${simData.simulated_prob}%`;
        const pdDeltaElem = document.getElementById('simPdDelta');
        const pdDiff = simData.prob_change;
        pdDeltaElem.textContent = pdDiff <= 0 ? `${pdDiff}%` : `+${pdDiff}%`;
        pdDeltaElem.className = `sim-delta ${pdDiff <= 0 ? 'delta-pos' : 'delta-neg'}`;

        const decisionElem = document.getElementById('simDecisionValue');
        decisionElem.textContent = simData.simulated_decision;
        decisionElem.className = `badge ${simData.simulated_decision === 'Approved' ? 'badge-success' : 'badge-warning'}`;
    }
});
