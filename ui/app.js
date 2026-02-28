/**
 * Multi-Dish Meal Analyser — Client-Side Logic.
 *
 * Each dish can be:
 *   - "single"   → user types food name + weight → USDA lookup
 *   - "composed" → user uploads image(s) → Nutrition5k model
 */

const MAX_DISHES = 5;
const MAX_IMAGES = 3;

let dishCount = 0;
let dishData = {};
// { id: { mode: 'single'|'composed', name, files: File[], weight } }

let foodList = [];  // cached for autocomplete

/* ── Initialise ───────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    addDish();
    document.getElementById('addDishBtn')
        .addEventListener('click', addDish);
    document.getElementById('analyseBtn')
        .addEventListener('click', analyseMeal);
    document.getElementById('newAnalysisBtn')
        .addEventListener('click', resetAll);

    // Pre-fetch food list for autocomplete
    fetch('/api/foods')
        .then(r => r.json())
        .then(list => { foodList = list; })
        .catch(() => { });
});

/* ── Dish Management ──────────────────────────────── */
function addDish() {
    if (Object.keys(dishData).length >= MAX_DISHES) return;

    const id = dishCount++;
    dishData[id] = { mode: 'single', name: '', files: [], weight: '' };

    const card = document.createElement('div');
    card.className = 'dish-card';
    card.id = `dish-${id}`;
    card.innerHTML = `
        <div class="dish-header">
            <div class="dish-number">${Object.keys(dishData).length}</div>

            <!-- Mode toggle -->
            <div class="mode-toggle">
                <button class="mode-btn active" id="modeBtn-single-${id}"
                        onclick="setMode(${id}, 'single')">
                    🥑 Single Item
                </button>
                <button class="mode-btn" id="modeBtn-composed-${id}"
                        onclick="setMode(${id}, 'composed')">
                    🍽️ Composed Dish
                </button>
            </div>

            <button class="remove-dish-btn"
                    onclick="removeDish(${id})"
                    title="Remove dish">&times;</button>
        </div>

        <!-- Single item inputs -->
        <div class="single-inputs" id="singleInputs-${id}">
            <input type="text" class="dish-name-input"
                   id="nameInput-${id}"
                   placeholder="Food name (e.g. tomato, chicken breast)"
                   list="foodSuggestions-${id}"
                   oninput="dishData[${id}].name = this.value">
            <datalist id="foodSuggestions-${id}">
                ${foodList.map(f => `<option value="${f}">`).join('')}
            </datalist>
            <div class="weight-input-row">
                <input type="number" class="weight-input"
                       id="weightInput-${id}"
                       placeholder="Weight (g)"
                       min="1" step="1"
                       oninput="dishData[${id}].weight = this.value">
                <span class="weight-unit">g</span>
            </div>
        </div>

        <!-- Composed dish inputs (image upload) -->
        <div class="composed-inputs hidden" id="composedInputs-${id}">
            <input type="text" class="dish-name-input"
                   placeholder="Dish name (optional)"
                   oninput="dishData[${id}].name = this.value">
            <div class="upload-area">
                <div class="image-previews" id="previews-${id}"></div>
                <button class="upload-btn" id="uploadBtn-${id}"
                        onclick="triggerUpload(${id})">
                    <span class="icon">📷</span>
                    Add photo
                </button>
                <input type="file" id="fileInput-${id}"
                       accept="image/*" multiple hidden
                       onchange="handleFiles(${id}, this.files)">
            </div>
            <div class="image-count" id="count-${id}">
                0 / ${MAX_IMAGES} images
            </div>
        </div>
    `;

    document.getElementById('dishesContainer').appendChild(card);
    updateUI();
}

function setMode(id, mode) {
    dishData[id].mode = mode;
    dishData[id].files = [];
    dishData[id].name = '';
    dishData[id].weight = '';

    // Toggle UI
    const single = document.getElementById(`singleInputs-${id}`);
    const composed = document.getElementById(`composedInputs-${id}`);
    const btnSingle = document.getElementById(`modeBtn-single-${id}`);
    const btnComposed = document.getElementById(`modeBtn-composed-${id}`);

    if (mode === 'single') {
        single.classList.remove('hidden');
        composed.classList.add('hidden');
        btnSingle.classList.add('active');
        btnComposed.classList.remove('active');
    } else {
        single.classList.add('hidden');
        composed.classList.remove('hidden');
        btnSingle.classList.remove('active');
        btnComposed.classList.add('active');
    }

    // Clear inputs
    const nameInput = document.getElementById(`nameInput-${id}`);
    if (nameInput) nameInput.value = '';
    const weightInput = document.getElementById(`weightInput-${id}`);
    if (weightInput) weightInput.value = '';
    renderPreviews(id);
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
    if (!container) return;
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
    if (count) {
        count.textContent = `${dish.files.length} / ${MAX_IMAGES} images`;
    }

    const btn = document.getElementById(`uploadBtn-${id}`);
    if (btn) {
        btn.classList.toggle('disabled', dish.files.length >= MAX_IMAGES);
    }
}

/* ── UI State ─────────────────────────────────────── */
function updateUI() {
    const dishIds = Object.keys(dishData);
    const hasData = dishIds.some(id => {
        const d = dishData[id];
        if (d.mode === 'single') {
            return d.name.trim() && d.weight;
        }
        return d.files.length > 0;
    });

    document.getElementById('addDishBtn').disabled =
        dishIds.length >= MAX_DISHES;
    document.getElementById('analyseBtn').disabled = !hasData;
}

/* ── Reset ────────────────────────────────────────── */
function resetAll() {
    dishData = {};
    dishCount = 0;
    document.getElementById('dishesContainer').innerHTML = '';
    document.getElementById('results').classList.remove('active');
    document.getElementById('errorMsg').classList.remove('active');
    document.getElementById('setupSection').classList.remove('hidden');
    document.getElementById('newAnalysisBtn').classList.add('hidden');
    addDish();
}

/* ── API Call ─────────────────────────────────────── */
async function analyseMeal() {
    const dishIds = Object.keys(dishData).filter(id => {
        const d = dishData[id];
        if (d.mode === 'single') return d.name.trim() && d.weight;
        return d.files.length > 0;
    });

    if (dishIds.length === 0) return;

    showLoading(true);
    hideResults();
    hideError();

    const formData = new FormData();
    dishIds.forEach((id, idx) => {
        const dish = dishData[id];
        formData.append(`dish_${idx}_mode`, dish.mode);
        formData.append(`dish_${idx}_name`,
            dish.name || `Dish ${idx + 1}`);

        if (dish.mode === 'single') {
            formData.append(`dish_${idx}_weight`, dish.weight);
        } else {
            dish.files.forEach(file => {
                formData.append(`dish_${idx}_images`, file);
            });
        }
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

        // Collect preview URLs for composed dishes
        const previewUrls = dishIds.map(id => {
            const files = dishData[id].files;
            return files.length > 0
                ? URL.createObjectURL(files[0])
                : null;
        });
        renderResults(data, previewUrls);

        document.getElementById('setupSection').classList.add('hidden');
        document.getElementById('newAnalysisBtn')
            .classList.remove('hidden');

    } catch (err) {
        showError(err.message);
    } finally {
        showLoading(false);
    }
}

/* ── Render Results ───────────────────────────────── */
function renderResults(data, previewUrls) {
    const container = document.getElementById('results');

    const dishesHtml = data.dishes.map((dish, i) => {
        const thumbSrc = previewUrls && previewUrls[i]
            ? previewUrls[i] : '';
        const thumbHtml = thumbSrc
            ? `<img class="dish-result-thumb" src="${thumbSrc}" alt="">`
            : '';
        const sourceTag = dish.source
            ? `<span class="source-tag">${dish.source === 'usda_api' ? '🇺🇸 USDA' : '📖 Local DB'}</span>`
            : '';
        const descTag = dish.usda_description && dish.usda_description !== dish.name
            ? `<span class="usda-desc">${dish.usda_description}</span>`
            : '';
        return `
        <div class="dish-result">
            <div class="dish-result-header">
                ${thumbHtml}
                <span class="dish-result-name">${dish.name}</span>
                ${sourceTag}
                ${descTag}
                <span class="dish-result-images">
                    ${dish.num_images > 0
                ? `${dish.num_images} image${dish.num_images > 1 ? 's' : ''}`
                : `${dish.weight_g}g`}
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
    `;
    }).join('');

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
