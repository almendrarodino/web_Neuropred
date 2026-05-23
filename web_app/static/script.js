// ================= STATE & CONFIG =================
const API_URL = "http://localhost:8000";
const TUMOR_THRESHOLD = 0.0610;

let currentToken = localStorage.getItem("token");
let currentPatientId = null;
let currentPatient = null;
let currentPredictions = [];
let currentFile = null;
let selectedHistoryPrediction = null;
let currentUser = null;

const views = {
    login: document.getElementById("login-view"),
    dashboard: document.getElementById("dashboard-view"),
    patientsList: document.getElementById("patients-list-container"),
    patientDetail: document.getElementById("patient-detail-container")
};

const forms = {
    login: document.getElementById("login-form"),
    newPatient: document.getElementById("form-new-patient"),
    editPatient: document.getElementById("form-edit-patient"),
    editNotes: document.getElementById("form-edit-notes"),
    uploadStudy: document.getElementById("form-upload-study"),
    userProfile: document.getElementById("form-user-profile")
};

const modals = {
    newPatient: document.getElementById("modal-new-patient"),
    editPatient: document.getElementById("modal-edit-patient"),
    editNotes: document.getElementById("modal-edit-notes"),
    newStudy: document.getElementById("modal-new-study"),
    viewHistory: document.getElementById("modal-view-history"),
    userProfile: document.getElementById("modal-user-profile")
};

const toastEl = document.getElementById("toast");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const btnAnalyze = document.getElementById("btn-analyze");

document.addEventListener("DOMContentLoaded", () => {
    setupEventListeners();
    if (currentToken) {
        checkAuthAndLoadDashboard();
    } else {
        showView("login");
    }
});

function setupEventListeners() {
    forms.login.addEventListener("submit", handleLoginSubmit);
    forms.newPatient.addEventListener("submit", handleCreatePatient);
    forms.editPatient.addEventListener("submit", handleEditPatient);
    forms.editNotes.addEventListener("submit", handleEditNotes);
    forms.uploadStudy.addEventListener("submit", handleUploadStudy);
    forms.userProfile.addEventListener("submit", handleUserProfileSubmit);

    document.getElementById("logout-btn").addEventListener("click", (e) => {
        e.preventDefault();
        logout();
    });

    document.getElementById("go-to-register").addEventListener("click", toggleRegisterView);
    document.getElementById("user-profile-btn").addEventListener("click", openUserProfileModal);
    document.getElementById("btn-new-patient").addEventListener("click", () => openModal(modals.newPatient));
    document.getElementById("btn-back-to-patients").addEventListener("click", goHome);
    document.getElementById("nav-patients").addEventListener("click", (e) => {
        e.preventDefault();
        goHome();
    });
    document.querySelectorAll(".app-logo").forEach((logo) => logo.addEventListener("click", goHome));
    document.querySelectorAll(".close-modal").forEach((btn) => btn.addEventListener("click", hideAllModals));

    document.getElementById("search-patient").addEventListener("input", (e) => {
        loadPatients(e.target.value, document.getElementById("filter-tumor").value, document.getElementById("sort-patients").value);
    });
    document.getElementById("filter-tumor").addEventListener("change", (e) => {
        loadPatients(document.getElementById("search-patient").value, e.target.value, document.getElementById("sort-patients").value);
    });
    document.getElementById("sort-patients").addEventListener("change", (e) => {
        loadPatients(document.getElementById("search-patient").value, document.getElementById("filter-tumor").value, e.target.value);
    });

    document.getElementById("btn-edit-patient").addEventListener("click", openEditPatientModal);
    document.getElementById("btn-delete-patient").addEventListener("click", deleteCurrentPatient);
    document.getElementById("btn-download-report").addEventListener("click", downloadCurrentReport);
    document.getElementById("btn-edit-notes").addEventListener("click", openNotesModal);
    document.getElementById("patient-notes-card").addEventListener("click", openNotesModal);
    document.getElementById("btn-new-study").addEventListener("click", () => {
        resetUploadModal();
        openModal(modals.newStudy);
    });

    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) handleFileSelect(e.target.files[0]);
    });
    document.getElementById("btn-remove-file").addEventListener("click", clearSelectedFile);
    document.getElementById("btn-download-history-image").addEventListener("click", () => {
        if (selectedHistoryPrediction) downloadPredictionImage(selectedHistoryPrediction);
    });
    document.getElementById("btn-delete-history").addEventListener("click", () => {
        if (selectedHistoryPrediction) deletePrediction(selectedHistoryPrediction.id);
    });
}

function showView(viewName) {
    Object.values(views).forEach((view) => view.classList.add("hidden"));
    if (viewName === "login") {
        views.login.classList.remove("hidden");
        return;
    }
    views.dashboard.classList.remove("hidden");
    if (viewName === "patientDetail") {
        views.patientDetail.classList.remove("hidden");
    } else {
        views.patientsList.classList.remove("hidden");
    }
}

function goHome() {
    if (!currentToken) {
        showView("login");
        return;
    }
    currentPatientId = null;
    currentPatient = null;
    showView("dashboard");
    loadPatients(document.getElementById("search-patient").value, document.getElementById("filter-tumor").value, document.getElementById("sort-patients").value);
}

function openModal(modalEl) {
    modalEl.classList.remove("hidden");
}

function closeModal(modalEl) {
    modalEl.classList.add("hidden");
}

function hideAllModals() {
    Object.values(modals).forEach(closeModal);
}

function showToast(message, type = "info") {
    toastEl.textContent = message;
    toastEl.style.background = type === "error" ? "var(--danger)" : (type === "success" ? "var(--success)" : "var(--text-main)");
    toastEl.classList.remove("hidden");
    setTimeout(() => toastEl.classList.add("hidden"), 3500);
}

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    }[char]));
}

function authHeaders(extra = {}) {
    return { ...extra, Authorization: `Bearer ${currentToken}` };
}

async function apiFetch(path, options = {}) {
    const res = await fetch(`${API_URL}${path}`, {
        ...options,
        headers: authHeaders(options.headers || {})
    });
    if (!res.ok) {
        let detail = "Error de solicitud";
        try {
            const data = await res.json();
            detail = data.detail || detail;
        } catch (_) {
            detail = await res.text();
        }
        throw new Error(detail);
    }
    return res;
}

async function handleLoginSubmit(e) {
    e.preventDefault();
    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value;
    const isRegistering = document.getElementById("go-to-register").textContent === "Iniciar Sesión";

    if (isRegistering) {
        try {
            const res = await fetch(`${API_URL}/register`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password })
            });
            if (!res.ok) {
                const data = await res.json();
                showToast(data.detail || "Error al registrar", "error");
                return;
            }
            showToast("Registro exitoso. Iniciando sesión...", "success");
        } catch (_) {
            showToast("Error de conexión", "error");
            return;
        }
    }

    try {
        const formData = new URLSearchParams();
        formData.append("username", username);
        formData.append("password", password);
        const res = await fetch(`${API_URL}/token`, { method: "POST", body: formData });
        if (!res.ok) {
            showToast("Credenciales incorrectas", "error");
            return;
        }
        const data = await res.json();
        currentToken = data.access_token;
        localStorage.setItem("token", currentToken);
        checkAuthAndLoadDashboard();
    } catch (_) {
        showToast("Error de conexión", "error");
    }
}

async function checkAuthAndLoadDashboard() {
    try {
        const res = await apiFetch("/me");
        const user = await res.json();
        currentUser = user;
        const displayName = user.display_name || user.username;
        document.getElementById("user-name-display").textContent = displayName;
        document.getElementById("user-initial").textContent = displayName.charAt(0).toUpperCase();
        showView("dashboard");
        loadPatients();
    } catch (_) {
        logout();
    }
}

function logout() {
    currentToken = null;
    localStorage.removeItem("token");
    showView("login");
}

function toggleRegisterView(e) {
    e.preventDefault();
    const btn = e.target;
    const title = document.querySelector(".login-left h1");
    const submitBtn = forms.login.querySelector("button");
    if (btn.textContent === "Registrarse") {
        title.textContent = "Nuevo Médico";
        submitBtn.textContent = "Crear Cuenta";
        btn.textContent = "Iniciar Sesión";
        btn.parentElement.childNodes[0].nodeValue = "¿Ya tienes cuenta? ";
    } else {
        title.textContent = "Acceso Médico";
        submitBtn.textContent = "Ingresar al Sistema";
        btn.textContent = "Registrarse";
        btn.parentElement.childNodes[0].nodeValue = "¿No tienes cuenta? ";
    }
}

function openUserProfileModal() {
    if (!currentUser) return;
    document.getElementById("up-first-name").value = currentUser.first_name || "";
    document.getElementById("up-last-name").value = currentUser.last_name || "";
    document.getElementById("up-dni").value = currentUser.dni || "";
    document.getElementById("up-profession").value = currentUser.profession || "";
    document.getElementById("up-email").value = currentUser.email || "";
    openModal(modals.userProfile);
}

async function handleUserProfileSubmit(e) {
    e.preventDefault();
    const payload = {
        first_name: document.getElementById("up-first-name").value.trim(),
        last_name: document.getElementById("up-last-name").value.trim(),
        dni: document.getElementById("up-dni").value.trim(),
        profession: document.getElementById("up-profession").value.trim(),
        email: document.getElementById("up-email").value.trim()
    };
    try {
        const res = await apiFetch("/me", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        currentUser = await res.json();
        const displayName = currentUser.display_name || currentUser.username;
        document.getElementById("user-name-display").textContent = displayName;
        document.getElementById("user-initial").textContent = displayName.charAt(0).toUpperCase();
        closeModal(modals.userProfile);
        showToast("Datos personales guardados", "success");
    } catch (error) {
        showToast(error.message || "No se pudieron guardar los datos", "error");
    }
}

async function loadPatients(search = "", tumorType = "", sort = "recent") {
    try {
        const params = new URLSearchParams();
        if (search) params.set("search", search);
        if (tumorType) params.set("tumor_type", tumorType);
        if (sort) params.set("sort", sort);
        const res = await apiFetch(`/patients?${params.toString()}`);
        renderPatientsList(await res.json());
    } catch (error) {
        showToast(error.message || "Error cargando pacientes", "error");
    }
}

function renderPatientsList(patients) {
    const grid = document.getElementById("patients-grid");
    grid.innerHTML = "";
    if (patients.length === 0) {
        grid.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 3rem; color: var(--text-muted)">No se encontraron pacientes. Haz clic en "Nuevo Paciente" para comenzar.</div>`;
        return;
    }

    patients.forEach((patient) => {
        const initials = `${patient.first_name || patient.full_name || "P"}`.charAt(0).toUpperCase();
        const card = document.createElement("div");
        card.className = "patient-card";
        card.onclick = () => loadPatientDetail(patient.id);
        card.innerHTML = `
            <div class="patient-card-header">
                <div class="pc-avatar">${escapeHTML(initials)}</div>
                <div>
                    <h3>${escapeHTML(patient.full_name)}</h3>
                    <p>DNI: ${escapeHTML(patient.dni || "-")}</p>
                </div>
            </div>
            <div class="patient-card-stats">
                <div class="stat-item">
                    <span class="stat-label">Edad</span>
                    <span class="stat-value">${escapeHTML(patient.age)} años</span>
                </div>
            </div>
        `;
        grid.appendChild(card);
    });
}

async function handleCreatePatient(e) {
    e.preventDefault();
    const payload = readPatientForm("np");
    try {
        const res = await apiFetch("/patients", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        closeModal(modals.newPatient);
        forms.newPatient.reset();
        showToast("Paciente creado", "success");
        const patient = await res.json();
        loadPatientDetail(patient.id);
    } catch (error) {
        showToast(error.message || "Error al crear paciente", "error");
    }
}

function readPatientForm(prefix) {
    return {
        first_name: document.getElementById(`${prefix}-first-name`).value.trim(),
        last_name: document.getElementById(`${prefix}-last-name`).value.trim(),
        age: parseInt(document.getElementById(`${prefix}-age`).value, 10),
        dni: document.getElementById(`${prefix}-dni`).value.trim(),
        email: document.getElementById(`${prefix}-email`).value.trim(),
        notes: prefix === "np" ? document.getElementById("np-notes").value.trim() : undefined
    };
}

async function loadPatientDetail(id) {
    currentPatientId = id;
    try {
        const res = await apiFetch(`/patients/${id}`);
        const data = await res.json();
        currentPatient = data.patient;
        currentPredictions = data.predictions;
        renderPatientDetail(data.patient, data.predictions);
        showView("patientDetail");
    } catch (error) {
        showToast(error.message || "Paciente no encontrado", "error");
        showView("dashboard");
    }
}

function renderPatientDetail(patient, predictions) {
    document.getElementById("detail-patient-name").textContent = patient.full_name;
    document.getElementById("detail-patient-id").textContent = `#${patient.id}`;
    document.getElementById("detail-patient-age").textContent = patient.age ?? "-";
    document.getElementById("detail-patient-dni").textContent = patient.dni || "-";
    document.getElementById("detail-patient-email").textContent = patient.email || "-";
    document.getElementById("detail-patient-notes").textContent = patient.notes || "Ninguna nota clínica registrada.";

    const list = document.getElementById("history-list");
    list.innerHTML = "";
    if (predictions.length === 0) {
        list.innerHTML = '<p style="color:var(--text-muted)">No hay estudios registrados para este paciente.</p>';
        return;
    }

    predictions.forEach((prediction) => list.appendChild(historyItem(prediction)));
}

function historyItem(prediction) {
    const item = document.createElement("div");
    item.className = "history-item";
    item.onclick = () => viewHistoricalStudy(prediction);
    const dateStr = formatDate(prediction.timestamp);
    const probsHtml = probabilityLines(prediction);
    item.innerHTML = `
        <div class="hi-image"><img src="${API_URL}${escapeHTML(prediction.image_path)}" alt="MRI"></div>
        <div class="hi-content">
            <div class="hi-header">
                ${getBadgeHTML(prediction.final_class)}
                <span class="hi-date">${escapeHTML(dateStr)}</span>
            </div>
            <div style="font-size: 0.875rem;"><strong>P(Tumor):</strong> ${(prediction.p_tumor * 100).toFixed(2)}%</div>
            ${probsHtml}
            ${prediction.study_notes ? `<div class="hi-notes">${escapeHTML(prediction.study_notes)}</div>` : ""}
            <div class="inline-actions">
                <button type="button" class="btn-text" data-action="download">Descargar imagen</button>
                <button type="button" class="btn-text danger-text" data-action="delete">Eliminar diagnóstico</button>
            </div>
        </div>
    `;
    item.querySelector('[data-action="download"]').addEventListener("click", (e) => {
        e.stopPropagation();
        downloadPredictionImage(prediction);
    });
    item.querySelector('[data-action="delete"]').addEventListener("click", (e) => {
        e.stopPropagation();
        deletePrediction(prediction.id);
    });
    return item;
}

function formatDate(value) {
    const date = new Date(value);
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

function getBadgeHTML(finalClass) {
    const map = {
        glioma: { text: "Glioma Detectado", class: "glioma" },
        meningioma: { text: "Meningioma Detectado", class: "meningioma" },
        pituitary: { text: "Tumor Pituitario Detectado", class: "pituitary" },
        no_tumor: { text: "Sin Tumor Evidente", class: "no_tumor" }
    };
    const mapped = map[finalClass] || { text: "Desconocido", class: "none" };
    return `<span class="badge ${mapped.class}">${mapped.text}</span>`;
}

function probabilityLines(prediction) {
    if (prediction.p_tumor < TUMOR_THRESHOLD || !prediction.probabilities_json) return "";
    const rows = Object.entries(prediction.probabilities_json)
        .map(([key, value]) => `<div>${escapeHTML(key)}: ${(value * 100).toFixed(1)}%</div>`)
        .join("");
    return `<div style="margin-top: 10px; font-size: 0.8rem;">${rows}</div>`;
}

function openEditPatientModal() {
    if (!currentPatient) return;
    document.getElementById("ep-first-name").value = currentPatient.first_name || "";
    document.getElementById("ep-last-name").value = currentPatient.last_name || "";
    document.getElementById("ep-age").value = currentPatient.age || "";
    document.getElementById("ep-dni").value = currentPatient.dni || "";
    document.getElementById("ep-email").value = currentPatient.email || "";
    openModal(modals.editPatient);
}

async function handleEditPatient(e) {
    e.preventDefault();
    if (!currentPatientId) return;
    const payload = readPatientForm("ep");
    delete payload.notes;
    try {
        const res = await apiFetch(`/patients/${currentPatientId}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        currentPatient = await res.json();
        closeModal(modals.editPatient);
        showToast("Paciente actualizado", "success");
        loadPatientDetail(currentPatientId);
    } catch (error) {
        showToast(error.message || "Error al actualizar paciente", "error");
    }
}

function openNotesModal(e) {
    if (e) e.stopPropagation();
    if (!currentPatient) return;
    document.getElementById("patient-notes-input").value = currentPatient.notes || "";
    openModal(modals.editNotes);
}

async function handleEditNotes(e) {
    e.preventDefault();
    try {
        const notes = document.getElementById("patient-notes-input").value;
        await apiFetch(`/patients/${currentPatientId}/notes`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ notes })
        });
        closeModal(modals.editNotes);
        showToast("Notas guardadas", "success");
        loadPatientDetail(currentPatientId);
    } catch (error) {
        showToast(error.message || "Error al guardar notas", "error");
    }
}

async function deleteCurrentPatient() {
    if (!currentPatientId || !currentPatient) return;
    if (!confirm(`¿Estás seguro de eliminar a ${currentPatient.full_name} y todos sus diagnósticos?`)) return;
    try {
        await apiFetch(`/patients/${currentPatientId}`, { method: "DELETE" });
        showToast("Paciente eliminado", "success");
        goHome();
    } catch (error) {
        showToast(error.message || "Error al eliminar paciente", "error");
    }
}

async function deletePrediction(predictionId) {
    if (!currentPatientId) return;
    if (!confirm("¿Estás seguro de eliminar este diagnóstico?")) return;
    try {
        await apiFetch(`/patients/${currentPatientId}/predictions/${predictionId}`, { method: "DELETE" });
        hideAllModals();
        showToast("Diagnóstico eliminado", "success");
        loadPatientDetail(currentPatientId);
    } catch (error) {
        showToast(error.message || "Error al eliminar diagnóstico", "error");
    }
}

function handleFileSelect(file) {
    const validExts = [".jpg", ".jpeg", ".png", ".mat", ".dcm", ".dicom", ".zip"];
    const ext = file.name.includes(".") ? file.name.substring(file.name.lastIndexOf(".")).toLowerCase() : "";
    if (!validExts.includes(ext)) {
        showToast("Formato no soportado. Usa JPG, PNG, MAT, DICOM o ZIP DICOM.", "error");
        return;
    }

    currentFile = file;
    dropZone.classList.add("hidden");
    document.getElementById("file-preview-container").classList.remove("hidden");
    document.getElementById("selected-file-name").textContent = file.name;

    if ([".jpg", ".jpeg", ".png"].includes(ext)) {
        document.getElementById("image-preview").src = URL.createObjectURL(file);
    } else {
        document.getElementById("image-preview").src = "/static/logo.png";
    }
    btnAnalyze.disabled = false;
}

function clearSelectedFile() {
    currentFile = null;
    fileInput.value = "";
    document.getElementById("file-preview-container").classList.add("hidden");
    dropZone.classList.remove("hidden");
    btnAnalyze.disabled = true;
}

function resetUploadModal() {
    clearSelectedFile();
    document.getElementById("study-notes").value = "";
    document.getElementById("results-waiting").classList.remove("hidden");
    document.getElementById("results-loading").classList.add("hidden");
    document.getElementById("results-success").classList.add("hidden");
}

async function handleUploadStudy(e) {
    e.preventDefault();
    if (!currentFile || !currentPatientId) return;

    document.getElementById("results-waiting").classList.add("hidden");
    document.getElementById("results-loading").classList.remove("hidden");
    document.getElementById("results-success").classList.add("hidden");
    btnAnalyze.disabled = true;

    const formData = new FormData();
    formData.append("file", currentFile);
    const notes = document.getElementById("study-notes").value;
    if (notes) formData.append("notes", notes);

    try {
        const res = await fetch(`${API_URL}/patients/${currentPatientId}/predict`, {
            method: "POST",
            headers: { Authorization: `Bearer ${currentToken}` },
            body: formData
        });
        document.getElementById("results-loading").classList.add("hidden");
        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            throw new Error(data.detail || "Error al analizar imagen");
        }
        const data = await res.json();
        renderPredictionResults(data);
        loadPatientDetail(currentPatientId);
    } catch (error) {
        document.getElementById("results-loading").classList.add("hidden");
        document.getElementById("results-waiting").classList.remove("hidden");
        showToast(error.message || "Error de conexión", "error");
        btnAnalyze.disabled = false;
    }
}

function renderPredictionResults(data) {
    const successPanel = document.getElementById("results-success");
    successPanel.classList.remove("hidden");
    const badgeMap = {
        glioma: { text: "GLIOMA", danger: true, title: "Tumor Glioma Detectado" },
        meningioma: { text: "MENINGIOMA", danger: true, title: "Tumor Meningioma Detectado" },
        pituitary: { text: "PITUITARIO", danger: true, title: "Tumor Pituitario Detectado" },
        no_tumor: { text: "SIN TUMOR", danger: false, title: "Cerebro Sano / Sin Tumor" }
    };
    const meta = badgeMap[data.final_class] || badgeMap.no_tumor;
    const badge = document.getElementById("res-badge");
    badge.textContent = meta.text;
    badge.className = `result-badge ${meta.danger ? "danger" : "success"}`;
    document.getElementById("res-title").textContent = meta.title;

    const pTumorPct = (data.p_tumor * 100).toFixed(1);
    document.getElementById("res-p-tumor").textContent = `${pTumorPct}%`;
    const bar = document.getElementById("bar-p-tumor");
    bar.style.width = `${pTumorPct}%`;
    bar.className = `fill ${meta.danger ? "danger" : "success"}`;

    const subtypesContainer = document.getElementById("res-subtypes");
    subtypesContainer.innerHTML = "<h4>Probabilidades de Subtipo [Stage 2]</h4>";
    if (data.final_class !== "no_tumor" && data.probs) {
        Object.entries(data.probs).forEach(([cls, prob]) => {
            const pct = (prob * 100).toFixed(1);
            const isWinner = cls === data.final_class;
            subtypesContainer.innerHTML += `
                <div class="prob-container mt-4">
                    <div class="prob-header" style="${isWinner ? "font-weight:700;color:var(--danger)" : ""}">
                        <span>${escapeHTML(cls.toUpperCase())}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="progress-bar"><div class="fill ${isWinner ? "danger" : ""}" style="width: ${pct}%"></div></div>
                </div>`;
        });
    } else {
        subtypesContainer.innerHTML += `<p style="color:var(--text-muted); font-size: 0.875rem;">Análisis de subtipos no requerido (Umbral P(Tumor) < ${TUMOR_THRESHOLD.toFixed(4)}).</p>`;
    }
}

function viewHistoricalStudy(prediction) {
    selectedHistoryPrediction = prediction;
    document.getElementById("history-image-view").src = API_URL + prediction.image_path;
    document.getElementById("history-notes-view").textContent = prediction.study_notes || "Sin notas clínicas registradas.";

    const badgeMap = {
        glioma: { text: "GLIOMA", danger: true, title: "Tumor Glioma Detectado" },
        meningioma: { text: "MENINGIOMA", danger: true, title: "Tumor Meningioma Detectado" },
        pituitary: { text: "PITUITARIO", danger: true, title: "Tumor Pituitario Detectado" },
        no_tumor: { text: "SIN TUMOR", danger: false, title: "Cerebro Sano / Sin Tumor" }
    };
    const meta = badgeMap[prediction.final_class] || { text: "DESCONOCIDO", danger: false, title: "Resultado Desconocido" };
    const badge = document.getElementById("history-badge-view");
    badge.textContent = meta.text;
    badge.className = `result-badge ${meta.danger ? "danger" : "success"}`;
    document.getElementById("history-title-view").textContent = meta.title;

    const pTumorPct = (prediction.p_tumor * 100).toFixed(1);
    document.getElementById("history-ptumor-view").textContent = `${pTumorPct}%`;
    const bar = document.getElementById("history-bar-ptumor-view");
    bar.style.width = `${pTumorPct}%`;
    bar.className = `fill ${meta.danger ? "danger" : "success"}`;

    const subtypesContainer = document.getElementById("history-subtypes-view");
    subtypesContainer.innerHTML = "<h4>Probabilidades de Subtipo [Stage 2]</h4>";
    if (prediction.final_class !== "no_tumor" && prediction.probabilities_json) {
        Object.entries(prediction.probabilities_json).forEach(([cls, prob]) => {
            const pct = (prob * 100).toFixed(1);
            const isWinner = cls === prediction.final_class;
            subtypesContainer.innerHTML += `
                <div class="prob-container mt-4">
                    <div class="prob-header" style="${isWinner ? "font-weight:700;color:var(--danger)" : ""}">
                        <span>${escapeHTML(cls.toUpperCase())}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="progress-bar"><div class="fill ${isWinner ? "danger" : ""}" style="width: ${pct}%"></div></div>
                </div>`;
        });
    } else {
        subtypesContainer.innerHTML += '<p style="color:var(--text-muted); font-size: 0.875rem;">Análisis de subtipos no requerido.</p>';
    }
    openModal(modals.viewHistory);
}

async function downloadPredictionImage(prediction) {
    const filename = prediction.original_filename || `estudio_${prediction.id}.jpg`;
    await downloadWithAuth(`/predictions/${prediction.id}/image/download`, filename);
}

async function downloadCurrentReport() {
    if (!currentPatientId) return;
    await downloadWithAuth(`/patients/${currentPatientId}/report`, `informe_neuropred_paciente_${currentPatientId}.pdf`);
}

async function downloadWithAuth(path, filename) {
    try {
        const res = await apiFetch(path);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    } catch (error) {
        showToast(error.message || "No se pudo descargar el archivo", "error");
    }
}
