/**
 * TruthLens.NLP - Fake News Detection Interactive Controller
 */

// Sample Demonstration Datasets
const SAMPLE_NEWS = {
    'real-1': {
        headline: "Global Climate Summit Reaches Historic Multi-Nation Carbon Agreement in Geneva",
        article: "Diplomats and environment ministers from over 190 nations concluded two weeks of intensive negotiations on Wednesday, finalizing a binding multilateral framework to reduce industrial carbon emissions by 45 percent over the coming decade. United Nations representatives confirmed that the accord establishes formal inspection protocols, annual transparency reports, and multilateral financial assistance funds for developing economies."
    },
    'real-2': {
        headline: "NASA Perseverance Rover Discovers Organic Carbon Compounds in Ancient Martian Crater",
        article: "Scientists at NASA's Jet Propulsion Laboratory reported on Thursday that sedimentary rock samples drilled from the Jezero Crater contain complex carbon-based molecules, suggesting that prebiotic aqueous conditions may have existed on Mars billions of years ago. The findings were corroborated by spectroscopic instruments and submitted for international peer review."
    },
    'fake-1': {
        headline: "SHOCKING: Secret Miracle Herb Cures All Terminal Diseases Overnight, Big Pharma In Panic!",
        article: "Mainstream media and corrupt medical elites do not want you to know this mind-blowing truth! A brave independent researcher just leaked secret laboratory videos exposing how a rare mountain root completely wipes out cancer, diabetes, and heart disease in less than 24 hours. Featured video exposes the covert billionaire agenda to suppress natural remedies. Watch the full explosive video before they delete it forever!"
    },
    'fake-2': {
        headline: "BOMBSHELL: Declassified Files Prove Shape-Shifting Aliens Infiltrated High Government Offices",
        article: "Insiders within deep state agencies have finally confirmed what patriots suspected all along. Whistleblower footage recorded during a secret midnight ceremony reveals high-ranking federal officials communicating directly with extraterrestrial reptilian overlords. Share this urgent report with everyone you know before the government shuts down the internet!"
    }
};

document.addEventListener('DOMContentLoaded', () => {
    const headlineInput = document.getElementById('headlineInput');
    const articleInput = document.getElementById('articleInput');
    const headlineCounter = document.getElementById('headlineCounter');
    const articleCounter = document.getElementById('articleCounter');
    const modeBadge = document.getElementById('modeBadge');
    const clearBtn = document.getElementById('clearBtn');
    const detectForm = document.getElementById('detectForm');
    const submitBtn = document.getElementById('submitBtn');
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');

    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultBanner = document.getElementById('resultBanner');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const confidenceValue = document.getElementById('confidenceValue');
    const realProbBar = document.getElementById('realProbBar');
    const fakeProbBar = document.getElementById('fakeProbBar');
    const realProbText = document.getElementById('realProbText');
    const fakeProbText = document.getElementById('fakeProbText');
    const analysisSummary = document.getElementById('analysisSummary');
    const fakeIndicatorsList = document.getElementById('fakeIndicatorsList');
    const realIndicatorsList = document.getElementById('realIndicatorsList');
    const statWords = document.getElementById('statWords');
    const statSentences = document.getElementById('statSentences');
    const statDiversity = document.getElementById('statDiversity');
    const disclaimerText = document.getElementById('disclaimerText');

    // Update character counters and input mode badge
    function updateCountersAndMode() {
        const hText = (headlineInput.value || '').trim();
        const aText = (articleInput.value || '').trim();

        headlineCounter.textContent = `${headlineInput.value.length} / 300`;
        articleCounter.textContent = `${articleInput.value.length.toLocaleString()} / 25,000`;

        if (hText && aText) {
            modeBadge.textContent = 'Combined (Headline + Article)';
            modeBadge.className = 'badge bg-soft-info text-info border border-subtle';
        } else if (hText) {
            modeBadge.textContent = 'Headline Only';
            modeBadge.className = 'badge bg-soft-cyan text-cyan border border-subtle';
        } else if (aText) {
            modeBadge.textContent = 'Article Body Only';
            modeBadge.className = 'badge bg-soft-cyan text-cyan border border-subtle';
        } else {
            modeBadge.textContent = 'Waiting for input';
            modeBadge.className = 'badge bg-dark-subtle text-secondary border border-subtle';
        }
    }

    headlineInput.addEventListener('input', updateCountersAndMode);
    articleInput.addEventListener('input', updateCountersAndMode);

    // Sample button listeners
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleKey = btn.getAttribute('data-sample');
            const sampleData = SAMPLE_NEWS[sampleKey];
            if (sampleData) {
                headlineInput.value = sampleData.headline;
                articleInput.value = sampleData.article;
                updateCountersAndMode();
                hideError();
                headlineInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        });
    });

    // Clear form listener
    clearBtn.addEventListener('click', () => {
        headlineInput.value = '';
        articleInput.value = '';
        updateCountersAndMode();
        hideError();
        resultsContainer.classList.add('d-none');
        resultsPlaceholder.classList.remove('d-none');
    });

    function showError(msg) {
        errorMessage.textContent = msg;
        errorAlert.classList.remove('d-none');
    }

    function hideError() {
        errorAlert.classList.add('d-none');
    }

    function setLoading(isLoading) {
        const btnText = submitBtn.querySelector('.btn-text');
        const btnSpinner = submitBtn.querySelector('.btn-spinner');

        if (isLoading) {
            submitBtn.disabled = true;
            btnText.classList.add('d-none');
            btnSpinner.classList.remove('d-none');
        } else {
            submitBtn.disabled = false;
            btnText.classList.remove('d-none');
            btnSpinner.classList.add('d-none');
        }
    }

    // Form submit listener
    detectForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();

        const headline = headlineInput.value.trim();
        const article = articleInput.value.trim();

        if (!headline && !article) {
            showError("Please enter a headline, paste an article, or pick a sample above.");
            return;
        }

        const combinedLength = (headline + " " + article).trim().length;
        if (combinedLength < 10) {
            showError("Input is too brief. Please enter at least 10 characters for NLP analysis.");
            return;
        }

        setLoading(true);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ headline, article })
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                showError(data.error || "Analysis could not be completed.");
                setLoading(false);
                return;
            }

            // Render prediction results
            renderResults(data);
        } catch (err) {
            console.error("Prediction API Error:", err);
            showError("Network connection error. Ensure the Flask server is running.");
        } finally {
            setLoading(false);
        }
    });

    function renderResults(data) {
        // Swap placeholder with result container
        resultsPlaceholder.classList.add('d-none');
        resultsContainer.classList.remove('d-none');

        const isReal = data.predicted_class === 0;

        // Banner styling
        resultBanner.className = `p-3 rounded-4 mb-4 result-banner ${isReal ? 'real-news' : 'fake-news'}`;
        resultTitle.textContent = data.prediction;
        resultIcon.innerHTML = isReal 
            ? '<i class="fa-solid fa-shield-check text-green"></i>' 
            : '<i class="fa-solid fa-triangle-exclamation text-danger"></i>';

        // Confidence & Dual Probability Bars
        confidenceValue.textContent = `${data.confidence}%`;
        const realProb = data.probabilities.real;
        const fakeProb = data.probabilities.fake;

        realProbBar.style.width = `${realProb}%`;
        realProbBar.setAttribute('aria-valuenow', realProb);
        fakeProbBar.style.width = `${fakeProb}%`;
        fakeProbBar.setAttribute('aria-valuenow', fakeProb);

        realProbText.textContent = `${realProb}%`;
        fakeProbText.textContent = `${fakeProb}%`;

        // Summary
        analysisSummary.textContent = data.analysis_summary;

        // Linguistic metrics
        if (data.linguistics) {
            statWords.textContent = data.linguistics.word_count.toLocaleString();
            statSentences.textContent = data.linguistics.sentence_count.toLocaleString();
            statDiversity.textContent = `${data.linguistics.lexical_diversity}%`;
        }

        // Disclaimer
        if (data.disclaimer) {
            disclaimerText.textContent = data.disclaimer;
        }

        // Explainability Chips
        fakeIndicatorsList.innerHTML = '';
        realIndicatorsList.innerHTML = '';

        const fakeTokens = (data.explainability && data.explainability.fake_indicators) || [];
        const realTokens = (data.explainability && data.explainability.real_indicators) || [];

        if (fakeTokens.length > 0) {
            fakeTokens.forEach(item => {
                const badge = document.createElement('span');
                badge.className = 'chip-badge-fake';
                badge.innerHTML = `<strong>${item.token}</strong> <span class="smaller opacity-75">+${item.weight}</span>`;
                fakeIndicatorsList.appendChild(badge);
            });
        } else {
            fakeIndicatorsList.innerHTML = '<span class="smaller text-secondary fst-italic">No strong sensationalism markers identified.</span>';
        }

        if (realTokens.length > 0) {
            realTokens.forEach(item => {
                const badge = document.createElement('span');
                badge.className = 'chip-badge-real';
                badge.innerHTML = `<strong>${item.token}</strong> <span class="smaller opacity-75">-${item.weight}</span>`;
                realIndicatorsList.appendChild(badge);
            });
        } else {
            realIndicatorsList.innerHTML = '<span class="smaller text-secondary fst-italic">No formal journalistic markers identified.</span>';
        }

        // Smooth scroll to result on mobile
        if (window.innerWidth < 992) {
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
});
