/**
 * Multi-Dish Meal Analyser — Client-Side Logic.
 *
 * Flow: set up 1-5 dishes with images → Analyse → view results → New Analysis.
 */

const MAX_DISHES = 5;
const MAX_IMAGES = 3;
const ACCEPTED = 'image/*';

let dishCount = 0;
let dishData = {};  // { dishId: { name, files: File[] } }

/* ── Initialise ───────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    addDish();
    document.getElementById('addDishBtn')
        .addEventListener('click', addDish);
    document.getElementById('analyseBtn')
        .addEventListener('click', analyseMeal);
    document.getElementById('newAnalysisBtn')
        .addEventListener('click', resetAll);
});

/* ── Dish Management ──────────────────────────────── */
function addDish() {
    if (Object.keys(dishData).length >= MAX_DISHES) return;

    const id = dishCount++;
    dishData[id] = { name: '', files: [] };

    const card = document.createElement('div');
    card.className = 'dish-card';
    card.id = `dish-${id}`;
    card.innerHTML = `
        <div class="dish-header">
            <div class="dish-number">${Object.keys(dishData).length}</div>
            <input type="text" class="dish-name-input"
                   placeholder="Dish name (optional)"
                   oninput="dishData[${id}].name = this.value">
            <button class="remove-dish-btn"
                    onclick="removeDish(${id})"
                    title="Remove dish">&times;</button>
        </div>
        <div class="upload-area">
            <div class="image-previews" id="previews-${id}"></div>
            <button class="upload-btn" id="uploadBtn-${id}"
                    onclick="triggerUpload(${id})">
                <span class="icon">📷</span>
                Add photo
            </button>
            <input type="file" id="fileInput-${id}"
                   accept="${ACCEPTED}" multiple hidden
                   onchange="handleFiles(${id}, this.files)">
        </div>
        <div class="image-count" id="count-${id}">
            0 / ${MAX_IMAGES} images
        </div>
    `;

    document.getElementById('dishesContainer').appendChild(card);
    updateUI();
}

function removeDish(id) {
    delete dishData[id];
    const el = document.getElementById(`dish-${id}`);
    if (el) {
        el.style.animation = 'slideIn 0.2s ease reverse';
        setTimeout(() => {
            el.remove();
            renumberDishes();
            updateUI();
        }, 200);
    }
}

function renumberDishes() {
    const cards = document.querySelectorAll('.dish-card');
    cards.forEach((card, i) => {
        card.querySelector('.dish-number').textContent = i + 1;
    });
}

/* ── Image Upload ─────────────────────────────────── */
function triggerUpload(id) {
    document.getElementById(`fileInput-${id}`).click();
}

function handleFiles(id, fileList) {
    const dish = dishData[id];
    if (!dish) return;

    for (const file of fileList) {
        if (dish.files.length >= MAX_IMAGES) break;
        dish.files.push(file);
    }

    renderPreviews(id);
    updateUI();
}

function removeImage(dishId, fileIdx) {
    dishData[dishId].files.splice(fileIdx, 1);
    renderPreviews(dishId);
    updateUI();
}

function renderPreviews(id) {
    const container = document.getElementById(`previews-${id}`);
    const dish = dishData[id];
    container.innerHTML = '';

    dish.files.forEach((file, idx) => {
        const div = document.createElement('div');
        div.className = 'image-preview';

        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.alt = file.name;

        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-img';
        removeBtn.innerHTML = '✕';
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            removeImage(id, idx);
        };

        div.appendChild(img);
        div.appendChild(removeBtn);
        container.appendChild(div);
    });

    const count = document.getElementById(`count-${id}`);
    count.textContent = `${dish.files.length} / ${MAX_IMAGES} images`;

    const btn = document.getElementById(`uploadBtn-${id}`);
    if (dish.files.length >= MAX_IMAGES) {
        btn.classList.add('disabled');
    } else {
        btn.classList.remove('disabled');
    }
}

/* ── UI State ─────────────────────────────────────── */
function updateUI() {
    const dishIds = Object.keys(dishData);
    const hasImages = dishIds.some(id => dishData[id].files.length > 0);

    document.getElementById('addDishBtn').disabled =
        dishIds.length >= MAX_DISHES;
    document.getElementById('analyseBtn').disabled = !hasImages;
}

/* ── Reset ────────────────────────────────────────── */
function resetAll() {
    // Clear all data
    dishData = {};
    dishCount = 0;

    // Clear UI
    document.getElementById('dishesContainer').innerHTML = '';
    document.getElementById('results').classList.remove('active');
    document.getElementById('errorMsg').classList.remove('active');

    // Show setup section, hide restart button
    document.getElementById('setupSection').classList.remove('hidden');
    document.getElementById('newAnalysisBtn').classList.add('hidden');

    // Add first dish card
    addDish();
}

/* ── API Call ─────────────────────────────────────── */
async function analyseMeal() {
    const dishIds = Object.keys(dishData)
        .filter(id => dishData[id].files.length > 0);

    if (dishIds.length === 0) return;

    showLoading(true);
    hideResults();
    hideError();

    const formData = new FormData();
    dishIds.forEach((id, idx) => {
        const dish = dishData[id];
        dish.files.forEach(file => {
            formData.append(`dish_${idx}_images`, file);
        });
        const name = dish.name || `Dish ${idx + 1}`;
        formData.append(`dish_${idx}_name`, name);
    });

    try {
        const resp = await fetch('/api/analyse_meal', {
            method: 'POST',
            body: formData,
        });

        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.error || 'Analysis failed');
        }

        renderResults(data);

        // Hide setup section, show restart button
        document.getElementById('setupSection').classList.add('hidden');
        document.getElementById('newAnalysisBtn').classList.remove('hidden');

    } catch (err) {
        showError(err.message);
    } finally {
        showLoading(false);
    }
}

/* ── Render Results ───────────────────────────────── */
function renderResults(data) {
    const container = document.getElementById('results');

    const dishesHtml = data.dishes.map((dish) => `
        <div class="dish-result">
            <div class="dish-result-header">
                <span class="dish-result-name">${dish.name}</span>
                <span class="dish-result-images">
                    ${dish.num_images} image${dish.num_images > 1 ? 's' : ''}
                </span>
            </div>
            <div class="macro-grid">
                <div class="macro-item">
                    <div class="macro-value">${dish.weight_g}g</div>
                    <div class="macro-label">Weight</div>
                </div>
                <div class="macro-item">
                    <div class="macro-value">${dish.carbs_g}g</div>
                    <div class="macro-label">Carbs</div>
                </div>
                <div class="macro-item">
                    <div class="macro-value">${dish.protein_g}g</div>
                    <div class="macro-label">Protein</div>
                </div>
                <div class="macro-item">
                    <div class="macro-value">${dish.fat_g}g</div>
                    <div class="macro-label">Fat</div>
                </div>
            </div>
        </div>
    `).join('');

    const t = data.totals;
    const b = data.bolus_recommendation;

    const totalsHtml = `
        <div class="totals-card">
            <h3>🍽️ Meal Totals</h3>
            <div class="totals-macro-grid">
                <div class="totals-macro-item">
                    <div class="totals-macro-value">${t.weight_g}g</div>
                    <div class="macro-label">Weight</div>
                </div>
                <div class="totals-macro-item">
                    <div class="totals-macro-value">${t.carbs_g}g</div>
                    <div class="macro-label">Carbs</div>
                </div>
                <div class="totals-macro-item">
                    <div class="totals-macro-value">${t.protein_g}g</div>
                    <div class="macro-label">Protein</div>
                </div>
                <div class="totals-macro-item">
                    <div class="totals-macro-value">${t.fat_g}g</div>
                    <div class="macro-label">Fat</div>
                </div>
            </div>
            <div class="fpu-row">
                <div class="fpu-item">
                    <div class="fpu-value">${b.fpu}</div>
                    <div class="macro-label">FPU</div>
                </div>
                <div class="fpu-item">
                    <div class="fpu-value">${b.equivalent_carbs_g}g</div>
                    <div class="macro-label">Equiv. Carbs</div>
                </div>
                <div class="fpu-item">
                    <div class="fpu-value">${b.total_active_carbs_g}g</div>
                    <div class="macro-label">Active Carbs</div>
                </div>
            </div>
        </div>
    `;

    const bolusHtml = `
        <div class="bolus-card">
            <h3>💉 Combo Bolus Recommendation</h3>
            <div class="bolus-split">
                <div class="bolus-section">
                    <div class="bolus-section-title">
                        ⚡ Immediate (${b.immediate_pct}%)
                    </div>
                    <div class="bolus-units">
                        ${b.immediate_units}<span>u</span>
                    </div>
                    <div class="bolus-detail">
                        ${b.immediate_carbs_g}g carbs — at meal time
                    </div>
                </div>
                <div class="bolus-section">
                    <div class="bolus-section-title">
                        ⏳ Extended (${b.extended_pct}%)
                    </div>
                    <div class="bolus-units">
                        ${b.extended_units}<span>u</span>
                    </div>
                    <div class="bolus-detail">
                        ${b.extended_carbs_g}g equiv. carbs —
                        over ${b.extension_duration_hours} hours
                    </div>
                </div>
            </div>
            <div class="bolus-total">
                <span class="bolus-total-label">Total insulin</span>
                <span class="bolus-total-value">
                    ${b.total_insulin_units}u
                </span>
            </div>
            <div class="bolus-disclaimer">
                ⚠️ This is a decision-support tool, not medical advice.
                Always verify with your endocrinologist. Consider IOB
                (insulin on board) and activity level before dosing.
            </div>
        </div>
    `;

    document.getElementById('dishResults').innerHTML = dishesHtml;
    document.getElementById('mealTotals').innerHTML =
        totalsHtml + bolusHtml;

    container.classList.add('active');
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Helpers ──────────────────────────────────────── */
function showLoading(show) {
    document.getElementById('loading')
        .classList.toggle('active', show);
    document.getElementById('analyseBtn').disabled = show;
}

function hideResults() {
    document.getElementById('results').classList.remove('active');
}

function showError(msg) {
    const el = document.getElementById('errorMsg');
    el.textContent = msg;
    el.classList.add('active');
}

function hideError() {
    document.getElementById('errorMsg').classList.remove('active');
}
