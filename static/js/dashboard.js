/**
 * TruthLens.NLP - Analytics Dashboard & History Logs Controller
 */

let distributionChartInstance = null;
let comparisonChartInstance = null;
let featuresChartInstance = null;
let currentFilter = 'all';

document.addEventListener('DOMContentLoaded', () => {
    initDashboard();

    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            initDashboard();
        });
    }

    // Filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.getAttribute('data-filter');
            loadHistory();
        });
    });

    // Search input
    const historySearch = document.getElementById('historySearch');
    if (historySearch) {
        let debounceTimer;
        historySearch.addEventListener('input', () => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                loadHistory();
            }, 300);
        });
    }

    // Clear history button
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', async () => {
            if (confirm("Are you sure you want to clear all recorded prediction logs?")) {
                try {
                    const res = await fetch('/api/history/clear', { method: 'POST' });
                    if (res.ok) {
                        loadHistory();
                    }
                } catch (err) {
                    console.error("Failed to clear history:", err);
                }
            }
        });
    }
});

async function initDashboard() {
    await loadMetricsAndCharts();
    await loadHistory();
}

async function loadMetricsAndCharts() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) return;
        const data = await response.json();
        const meta = data.metadata || {};
        const comparison = data.comparison || [];

        renderDistributionChart(meta);
        renderComparisonChart(comparison);
        renderFeaturesChart(meta);
        updateConfusionMatrix(meta);
    } catch (err) {
        console.error("Failed to load metrics for charts:", err);
    }
}

function renderDistributionChart(meta) {
    const ctx = document.getElementById('distributionChart');
    if (!ctx) return;

    const realCount = meta.real_samples_total || 2970;
    const fakeCount = meta.fake_samples_total || 2794;

    if (distributionChartInstance) {
        distributionChartInstance.destroy();
    }

    distributionChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Real News Articles', 'Fake News Articles'],
            datasets: [{
                data: [realCount, fakeCount],
                backgroundColor: ['#10b981', '#ef4444'],
                borderColor: '#111827',
                borderWidth: 2,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#9ca3af',
                        padding: 16,
                        font: { family: 'Inter', size: 12 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const val = context.raw || 0;
                            const total = realCount + fakeCount;
                            const pct = ((val / total) * 100).toFixed(1);
                            return ` ${context.label}: ${val.toLocaleString()} (${pct}%)`;
                        }
                    }
                }
            },
            cutout: '68%'
        }
    });
}

function renderComparisonChart(comparison) {
    const ctx = document.getElementById('comparisonChart');
    if (!ctx) return;

    if (comparisonChartInstance) {
        comparisonChartInstance.destroy();
    }

    const labels = comparison.map(m => m.model_name);
    const accuracies = comparison.map(m => m.accuracy);
    const precisions = comparison.map(m => m.precision);
    const recalls = comparison.map(m => m.recall);
    const f1Scores = comparison.map(m => m.f1_score);

    comparisonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.length ? labels : ['Logistic Regression', 'Naive Bayes', 'Linear SVM'],
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accuracies.length ? accuracies : [97.75, 94.19, 97.92],
                    backgroundColor: '#06b6d4',
                    borderRadius: 4
                },
                {
                    label: 'F1-Score (%)',
                    data: f1Scores.length ? f1Scores : [97.74, 94.19, 97.92],
                    backgroundColor: '#3b82f6',
                    borderRadius: 4
                },
                {
                    label: 'Precision (%)',
                    data: precisions.length ? precisions : [97.74, 94.19, 97.92],
                    backgroundColor: '#10b981',
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 90,
                    max: 100,
                    grid: { color: 'rgba(55, 65, 81, 0.4)' },
                    ticks: { color: '#9ca3af', callback: val => val + '%' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#9ca3af', font: { family: 'Inter', size: 11 } }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#9ca3af', font: { family: 'Inter', size: 11 } }
                }
            }
        }
    });
}

function renderFeaturesChart(meta) {
    const ctx = document.getElementById('featuresChart');
    if (!ctx) return;

    if (featuresChartInstance) {
        featuresChartInstance.destroy();
    }

    const topFeatures = meta.top_features || {};
    const fakeIndicators = (topFeatures.fake_indicators || []).slice(0, 6);
    const realIndicators = (topFeatures.real_indicators || []).slice(0, 6);

    const labels = [];
    const values = [];
    const colors = [];

    // Add fake indicators (positive)
    fakeIndicators.forEach(item => {
        labels.push(`${item.word} (Fake)`);
        values.push(item.weight);
        colors.push('#ef4444');
    });

    // Add real indicators (positive magnitude for comparison)
    realIndicators.forEach(item => {
        labels.push(`${item.word} (Real)`);
        values.push(item.weight);
        colors.push('#10b981');
    });

    featuresChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Relative Feature Coefficient Weight',
                data: values,
                backgroundColor: colors,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(55, 65, 81, 0.4)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e5e7eb', font: { family: 'Inter', size: 11 } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` Weight: ${ctx.raw}`
                    }
                }
            }
        }
    });
}

function updateConfusionMatrix(meta) {
    const tm = meta.test_metrics || {};
    const cm = tm.confusion_matrix || {};

    const elRR = document.getElementById('cmRealReal');
    const elRF = document.getElementById('cmRealFake');
    const elFR = document.getElementById('cmFakeReal');
    const elFF = document.getElementById('cmFakeFake');

    if (elRR && cm.true_real_pred_real !== undefined) elRR.textContent = cm.true_real_pred_real;
    if (elRF && cm.true_real_pred_fake !== undefined) elRF.textContent = cm.true_real_pred_fake;
    if (elFR && cm.true_fake_pred_real !== undefined) elFR.textContent = cm.true_fake_pred_real;
    if (elFF && cm.true_fake_pred_fake !== undefined) elFF.textContent = cm.true_fake_pred_fake;
}

async function loadHistory() {
    const tableBody = document.getElementById('historyTableBody');
    if (!tableBody) return;

    const searchInput = document.getElementById('historySearch');
    const searchVal = searchInput ? searchInput.value.trim() : '';

    let url = `/api/history?limit=50&filter=${encodeURIComponent(currentFilter)}`;
    if (searchVal) {
        url += `&search=${encodeURIComponent(searchVal)}`;
    }

    try {
        const response = await fetch(url);
        if (!response.ok) return;
        const data = await response.json();
        const items = data.history || [];

        if (items.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center text-secondary py-4 fst-italic">
                        No prediction records found. Run an analysis on the "Detect News" page to log items here.
                    </td>
                </tr>
            `;
            return;
        }

        let html = '';
        items.forEach(item => {
            const isReal = item.predicted_class === 0;
            const badgeClass = isReal ? 'bg-soft-green text-green border-green-subtle' : 'bg-soft-danger text-danger border-danger-subtle';
            const icon = isReal ? 'fa-shield-check' : 'fa-triangle-exclamation';

            html += `
                <tr>
                    <td class="text-secondary small">#${item.id}</td>
                    <td class="text-secondary small text-nowrap">${item.timestamp}</td>
                    <td>
                        <div class="text-white small fw-semibold text-truncate" style="max-width: 320px;" title="${escapeHtml(item.title || item.snippet)}">
                            ${escapeHtml(item.title || item.snippet || 'Untitled')}
                        </div>
                    </td>
                    <td>
                        <span class="badge bg-dark-subtle text-secondary border border-subtle smaller text-capitalize">
                            ${escapeHtml(item.input_mode)}
                        </span>
                    </td>
                    <td>
                        <span class="badge ${badgeClass} border px-2 py-1 small">
                            <i class="fa-solid ${icon} me-1"></i> ${escapeHtml(item.prediction)}
                        </span>
                    </td>
                    <td>
                        <strong class="text-cyan small">${item.confidence}%</strong>
                    </td>
                    <td>
                        ${item.top_indicator ? `<span class="badge bg-card-subtle text-light border border-subtle font-monospace smaller">${escapeHtml(item.top_indicator)}</span>` : '<span class="text-muted smaller">-</span>'}
                    </td>
                </tr>
            `;
        });

        tableBody.innerHTML = html;
    } catch (err) {
        console.error("Failed to load history items:", err);
        tableBody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-danger py-3">
                    Failed to load prediction history.
                </td>
            </tr>
        `;
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
