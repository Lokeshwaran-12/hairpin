// Hairpin Guard AI - Main Frontend Controller
// High-Contrast White Theme with Sidebar, Scanning Effect & Live Video Detection

let currentImageBase64 = null;
let currentDetectData = null;
let telemetryInterval = null;

// DOM Element Registry
const dom = {
    // Sidebar Navigation Tabs
    tabImageBtn: document.getElementById('tab-image-btn'),
    tabVideoBtn: document.getElementById('tab-video-btn'),
    tabLogsBtn: document.getElementById('tab-logs-btn'),
    sectionImage: document.getElementById('section-image'),
    sectionVideo: document.getElementById('section-video'),
    sectionLogs: document.getElementById('section-logs'),

    // Image Upload & Controls
    imageUpload: document.getElementById('image-upload'),
    btnUseSample: document.getElementById('btn-use-sample'),
    selectedFileName: document.getElementById('selected-file-name'),
    confSlider: document.getElementById('conf-slider'),
    confVal: document.getElementById('conf-val'),
    viewStyleSelect: document.getElementById('view-style-select'),
    btnDownload: document.getElementById('btn-download'),

    // Scanning Effect Overlay
    scanOverlay: document.getElementById('scan-overlay'),
    mainDetectedImage: document.getElementById('main-detected-image'),
    imageBadge: document.getElementById('image-badge'),

    // Image Detection Status Cards
    valTotalCount: document.getElementById('val-total-count'),
    valBreakdown: document.getElementById('val-breakdown'),
    valRiskLevel: document.getElementById('val-risk-level'),
    valTtc: document.getElementById('val-ttc'),
    signalALight: document.getElementById('signal-a-light'),
    signalAText: document.getElementById('signal-a-text'),
    signalADesc: document.getElementById('signal-a-desc'),
    signalBLight: document.getElementById('signal-b-light'),
    signalBText: document.getElementById('signal-b-text'),
    signalBDesc: document.getElementById('signal-b-desc'),
    tableResultsBody: document.getElementById('table-results-body'),
    imageEmergencyBanner: document.getElementById('image-emergency-banner'),
    imageEmergencyBannerDesc: document.getElementById('image-emergency-banner-desc'),

    // Live Video Elements
    videoUploadInput: document.getElementById('video-upload-input'),
    btnSyntheticMode: document.getElementById('btn-synthetic-mode'),
    btnRefreshStream: document.getElementById('btn-refresh-stream'),
    videoActiveBadge: document.getElementById('video-active-badge'),
    liveStreamPlayer: document.getElementById('live-stream-player'),
    streamLiveTag: document.getElementById('stream-live-tag'),
    videoEmergencyBanner: document.getElementById('video-emergency-banner'),
    videoEmergencyBannerDesc: document.getElementById('video-emergency-banner-desc'),

    // Centric Detection Line Controls & Metrics
    btnToggleCentric: document.getElementById('btn-toggle-centric'),
    btnToggleLeftLane: document.getElementById('btn-toggle-left-lane'),
    centricLineSlider: document.getElementById('centric-line-slider'),
    centricPosVal: document.getElementById('centric-pos-val'),
    btnResetCentric: document.getElementById('btn-reset-centric'),
    centricStatusBadge: document.getElementById('centric-status-badge'),
    videoRedAlertBanner: document.getElementById('video-red-alert-banner'),
    redAlertBannerDesc: document.getElementById('red-alert-banner-desc'),
    centricTotalCrossings: document.getElementById('centric-total-crossings'),
    centricDetailStatus: document.getElementById('centric-detail-status'),
    centricUphillCrossings: document.getElementById('centric-uphill-crossings'),
    centricDownhillCrossings: document.getElementById('centric-downhill-crossings'),
    centricLeftCrossings: document.getElementById('centric-left-crossings'),
    centricLeftStatus: document.getElementById('centric-left-status'),
    centricActiveTouching: document.getElementById('centric-active-touching'),

    // Live Video Telemetry Cards
    streamValTotal: document.getElementById('stream-val-total'),
    streamValBreakdown: document.getElementById('stream-val-breakdown'),
    streamValRisk: document.getElementById('stream-val-risk'),
    streamValTtc: document.getElementById('stream-val-ttc'),
    streamSignalALight: document.getElementById('stream-signal-a-light'),
    streamSignalAText: document.getElementById('stream-signal-a-text'),
    streamSignalADesc: document.getElementById('stream-signal-a-desc'),
    streamSignalBLight: document.getElementById('stream-signal-b-light'),
    streamSignalBText: document.getElementById('stream-signal-b-text'),
    streamSignalBDesc: document.getElementById('stream-signal-b-desc'),

    // Logs
    btnReloadLogs: document.getElementById('btn-reload-logs'),
    tableLogsBody: document.getElementById('table-logs-body')
};

// -----------------------------------------------------------------------------
// 1. Initialization
// -----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    initImageSection();
    initVideoSection();
    initLogsSection();

    // Load initial sample highway image
    loadSample();
});

// -----------------------------------------------------------------------------
// 2. Sidebar Navigation
// -----------------------------------------------------------------------------
function initSidebar() {
    const tabs = [
        { btn: dom.tabImageBtn, section: dom.sectionImage },
        { btn: dom.tabVideoBtn, section: dom.sectionVideo },
        { btn: dom.tabLogsBtn, section: dom.sectionLogs }
    ];

    tabs.forEach(({ btn, section }) => {
        btn.addEventListener('click', () => {
            tabs.forEach(t => {
                t.btn.classList.remove('active');
                t.section.classList.remove('active');
            });
            btn.classList.add('active');
            section.classList.add('active');

            // Manage video telemetry polling
            if (btn === dom.tabVideoBtn) {
                startVideoTelemetry();
            } else {
                stopVideoTelemetry();
            }

            if (btn === dom.tabLogsBtn) {
                loadLogs();
            }
        });
    });
}

// -----------------------------------------------------------------------------
// 3. Image Detection & Scanning Effect
// -----------------------------------------------------------------------------
function initImageSection() {
    dom.imageUpload.addEventListener('change', async (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            dom.selectedFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
            try {
                currentImageBase64 = await fileToBase64(file);
                // Immediately display raw image preview under scanning effect
                dom.mainDetectedImage.src = currentImageBase64;
                runDetection();
            } catch (err) {
                console.error('File read error:', err);
                alert('Could not read image file.');
            }
        }
    });

    dom.btnUseSample.addEventListener('click', () => {
        loadSample();
    });

    let debounceTimer = null;
    dom.confSlider.addEventListener('input', (e) => {
        dom.confVal.textContent = `${e.target.value}%`;
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            runDetection();
        }, 250);
    });

    dom.viewStyleSelect.addEventListener('change', () => {
        updateImageSource();
    });

    dom.btnDownload.addEventListener('click', () => {
        if (!dom.mainDetectedImage.src) return;
        const a = document.createElement('a');
        a.href = dom.mainDetectedImage.src;
        a.download = `yolov8_hairpin_detect_${Date.now()}.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });
}

function showScanningEffect(show) {
    if (show) {
        dom.scanOverlay.style.display = 'flex';
        dom.imageBadge.textContent = '⚡ Scanning Vehicle Geometry...';
        dom.imageBadge.className = 'status-tag tag-live';
    } else {
        dom.scanOverlay.style.display = 'none';
        dom.imageBadge.className = 'status-tag';
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = (err) => reject(err);
        reader.readAsDataURL(file);
    });
}

async function loadSample() {
    dom.selectedFileName.textContent = 'Loading images.jpeg...';
    try {
        const res = await fetch('/api/sample');
        if (!res.ok) throw new Error('Sample endpoint unavailable');
        const data = await res.json();

        currentImageBase64 = data.image;
        dom.mainDetectedImage.src = data.image;
        dom.selectedFileName.textContent = `${data.filename} (${data.width}x${data.height} px)`;
        runDetection();
    } catch (err) {
        console.warn('Sample load warning:', err);
        dom.selectedFileName.textContent = 'Upload an image to start';
    }
}

async function runDetection() {
    if (!currentImageBase64) return;

    // Trigger visual scanning animation
    showScanningEffect(true);

    const payload = {
        image: currentImageBase64,
        confidence: parseFloat((dom.confSlider.value / 100).toFixed(2)),
        iou: 0.45,
        classes: 'car,motorcycle,bus,truck',
        enable_clahe: true,
        box_thickness: 2,
        show_conf: true,
        show_centroids: true
    };

    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || errData.error || `HTTP ${response.status}`);
        }

        const data = await response.json();
        currentDetectData = data;

        // Hide scanning effect once detection results arrive
        showScanningEffect(false);
        renderOutput(data);
    } catch (err) {
        showScanningEffect(false);
        console.error('Detection error:', err);
        dom.imageBadge.textContent = 'Detection Error';
        alert('Detection Error: ' + err.message);
    }
}

function renderOutput(data) {
    // 1. Vehicle Counts
    dom.valTotalCount.textContent = data.summary.total;
    const parts = [];
    if (data.summary.car) parts.push(`${data.summary.car} Car${data.summary.car > 1 ? 's' : ''}`);
    if (data.summary.truck) parts.push(`${data.summary.truck} Truck${data.summary.truck > 1 ? 's' : ''}`);
    if (data.summary.bus) parts.push(`${data.summary.bus} Bus${data.summary.bus > 1 ? 'es' : ''}`);
    if (data.summary.motorcycle) parts.push(`${data.summary.motorcycle} Bike${data.summary.motorcycle > 1 ? 's' : ''}`);
    dom.valBreakdown.textContent = parts.join(' | ') || 'No vehicles detected';

    dom.imageBadge.textContent = `${data.summary.total} Vehicles Detected & Tracked`;

    // 2. Risk Badge
    const risk = data.risk_assessment;
    dom.valRiskLevel.className = 'risk-pill-lg';
    if (risk.level === 0) {
        dom.valRiskLevel.classList.add('pill-green');
        dom.valRiskLevel.textContent = 'LEVEL 0 - SAFE';
    } else if (risk.level === 1) {
        dom.valRiskLevel.classList.add('pill-yellow');
        dom.valRiskLevel.textContent = 'LEVEL 1 - CAUTION';
    } else if (risk.level === 2) {
        dom.valRiskLevel.classList.add('pill-yellow');
        dom.valRiskLevel.textContent = 'LEVEL 2 - WARNING';
    } else {
        dom.valRiskLevel.classList.add('pill-red');
        dom.valRiskLevel.textContent = 'LEVEL 3 - CRITICAL';
    }
    dom.valTtc.textContent = risk.min_ttc ? `Min TTC: ${risk.min_ttc.toFixed(1)}s` : 'Trajectory clearance verified';

    // 3. Dual Signals
    setSignalUI(dom.signalALight, dom.signalAText, dom.signalADesc, risk.signal_a_state, risk.signal_a_message);
    setSignalUI(dom.signalBLight, dom.signalBText, dom.signalBDesc, risk.signal_b_state, risk.signal_b_message);

    // Emergency Preemption Alert Banner for Image Mode
    const isImagePreempted = Boolean(risk.emergency_preemption_active) || Boolean(data.summary && data.summary.ambulance > 0);
    if (dom.imageEmergencyBanner) {
        dom.imageEmergencyBanner.style.display = isImagePreempted ? 'flex' : 'none';
        if (isImagePreempted && dom.imageEmergencyBannerDesc) {
            const count = (data.summary && data.summary.ambulance) || risk.emergency_vehicle_count || 1;
            dom.imageEmergencyBannerDesc.textContent = `🚑 ${count} Ambulance detected — Priority clearance engaged. Signals preempted!`;
        }
    }

    // 4. Update Output Image
    updateImageSource();

    // 5. Telemetry Table
    renderTable(data.detections);

    dom.btnDownload.disabled = false;
}

function setSignalUI(dotEl, textEl, descEl, state, msg) {
    dotEl.className = 'signal-dot-lg';
    if (state === 'RED') {
        dotEl.classList.add('dot-red');
    } else if (state === 'AMBER') {
        dotEl.classList.add('dot-amber');
    } else if (state === 'YELLOW') {
        dotEl.classList.add('dot-yellow');
    } else {
        dotEl.classList.add('dot-green');
    }
    textEl.textContent = state;
    descEl.textContent = msg || (state === 'GREEN' ? 'Clear - Normal speed' : 'Opposing traffic active');
}

function updateImageSource() {
    if (!currentDetectData) return;
    const style = dom.viewStyleSelect.value;
    if (style === 'boxes') {
        dom.mainDetectedImage.src = currentDetectData.images.boxes_only;
    } else if (style === 'hud') {
        dom.mainDetectedImage.src = currentDetectData.images.full_hud;
    } else {
        dom.mainDetectedImage.src = currentDetectData.images.original;
    }
}

function renderTable(detections) {
    dom.tableResultsBody.innerHTML = '';
    if (!detections || detections.length === 0) {
        dom.tableResultsBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No vehicle detections available.</td></tr>';
        return;
    }

    detections.forEach(d => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>#${d.index}</td>
            <td><strong>${d.class_name}</strong></td>
            <td>${d.confidence}%</td>
            <td>[${d.box.join(', ')}]</td>
            <td>${d.width} &times; ${d.height} px</td>
            <td>(${d.center[0]}, ${d.center[1]})</td>
        `;
        dom.tableResultsBody.appendChild(tr);
    });
}

// -----------------------------------------------------------------------------
// 4. Live Video Detection & Video File Upload
// -----------------------------------------------------------------------------
function initVideoSection() {
    // Video File Upload handler
    dom.videoUploadInput.addEventListener('change', async (e) => {
        if (!e.target.files || !e.target.files[0]) return;
        const file = e.target.files[0];

        dom.videoActiveBadge.textContent = `Uploading & Initializing: ${file.name}...`;

        const formData = new FormData();
        formData.append('video', file);

        try {
            const res = await fetch('/api/video/upload', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.error || errData.details || `Upload failed with HTTP ${res.status}`);
            }

            const data = await res.json();
            dom.videoActiveBadge.textContent = `Active Video: ${data.filename}`;
            dom.streamLiveTag.textContent = `● LIVE DETECTION: ${data.filename.toUpperCase()}`;

            // Refresh video player to stream with visible YOLO bounding boxes
            dom.liveStreamPlayer.src = `/api/stream/feed?t=${Date.now()}`;
        } catch (err) {
            console.error('Video upload failed:', err);
            dom.videoActiveBadge.textContent = 'Upload failed. Retrying in 2s...';
            // Auto retry once if initial load was caught during model startup
            setTimeout(async () => {
                try {
                    dom.videoActiveBadge.textContent = `Retrying upload: ${file.name}...`;
                    const retryRes = await fetch('/api/video/upload', {
                        method: 'POST',
                        body: formData
                    });
                    if (retryRes.ok) {
                        const data = await retryRes.json();
                        dom.videoActiveBadge.textContent = `Active Video: ${data.filename}`;
                        dom.streamLiveTag.textContent = `● LIVE DETECTION: ${data.filename.toUpperCase()}`;
                        dom.liveStreamPlayer.src = `/api/stream/feed?t=${Date.now()}`;
                        return;
                    }
                } catch (retryErr) {
                    // fall through
                }
                dom.videoActiveBadge.textContent = 'Upload failed. Please click upload again.';
                alert('Video Upload Error: ' + err.message);
            }, 2000);
        }
    });

    // Curve Simulation button
    dom.btnSyntheticMode.addEventListener('click', async () => {
        try {
            await fetch('/api/stream/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source: 'synthetic' })
            });
            dom.videoActiveBadge.textContent = 'Active: Simulation Highway Curve';
            dom.streamLiveTag.textContent = '● LIVE DETECTION: SIMULATION CURVE';
            dom.liveStreamPlayer.src = `/api/stream/feed?t=${Date.now()}`;
        } catch (err) {
            console.error('Switch error:', err);
        }
    });

    // Reload stream button
    dom.btnRefreshStream.addEventListener('click', () => {
        dom.liveStreamPlayer.src = `/api/stream/feed?t=${Date.now()}`;
    });

    // Centric Detection Line Controls
    let centricLineEnabled = true;
    let centricDebounce = null;

    if (dom.btnToggleCentric) {
        dom.btnToggleCentric.addEventListener('click', async () => {
            centricLineEnabled = !centricLineEnabled;
            dom.btnToggleCentric.innerHTML = centricLineEnabled ?
                '<span class="btn-icon">📏</span> Centric Detection Line: ON' :
                '<span class="btn-icon">📏</span> Centric Detection Line: OFF';
            dom.btnToggleCentric.className = centricLineEnabled ? 'btn btn-primary btn-lg' : 'btn btn-secondary btn-lg';

            try {
                await fetch('/api/video/centric-line', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        enabled: centricLineEnabled,
                        y_ratio: parseFloat(dom.centricLineSlider.value) / 100
                    })
                });
            } catch (err) {
                console.warn('Centric line config error:', err);
            }
        });
    }

    if (dom.centricLineSlider) {
        dom.centricLineSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            dom.centricPosVal.textContent = `${val}%`;
            clearTimeout(centricDebounce);
            centricDebounce = setTimeout(async () => {
                try {
                    await fetch('/api/video/centric-line', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            enabled: centricLineEnabled,
                            y_ratio: parseFloat(val) / 100
                        })
                    });
                } catch (err) {
                    console.warn('Slider update error:', err);
                }
            }, 150);
        });
    }

    // Toggle Left Lane Only Detection Mode
    let leftLaneOnlyMode = false;
    if (dom.btnToggleLeftLane) {
        dom.btnToggleLeftLane.addEventListener('click', async () => {
            leftLaneOnlyMode = !leftLaneOnlyMode;
            dom.btnToggleLeftLane.innerHTML = leftLaneOnlyMode ?
                '<span class="btn-icon">🎯</span> Left Lane Mode: ONLY LEFT' :
                '<span class="btn-icon">🎯</span> Left Lane Mode: ALL LANES';
            dom.btnToggleLeftLane.className = leftLaneOnlyMode ? 'btn btn-primary btn-lg' : 'btn btn-secondary btn-lg';

            try {
                await fetch('/api/video/left-lane-mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: leftLaneOnlyMode })
                });
            } catch (err) {
                console.warn('Left lane mode error:', err);
            }
        });
    }

    if (dom.btnResetCentric) {
        dom.btnResetCentric.addEventListener('click', async () => {
            try {
                await fetch('/api/video/centric-line/reset', { method: 'POST' });
                if (dom.centricTotalCrossings) dom.centricTotalCrossings.textContent = '0';
                if (dom.centricUphillCrossings) dom.centricUphillCrossings.textContent = '0';
                if (dom.centricDownhillCrossings) dom.centricDownhillCrossings.textContent = '0';
                if (dom.centricLeftCrossings) dom.centricLeftCrossings.textContent = '0';
                if (dom.centricActiveTouching) dom.centricActiveTouching.textContent = '0';
                if (dom.centricLeftStatus) dom.centricLeftStatus.textContent = 'Safe (0 Left Passing)';
                if (dom.videoRedAlertBanner) dom.videoRedAlertBanner.style.display = 'none';
                if (dom.liveStreamPlayer && dom.liveStreamPlayer.parentElement) {
                    dom.liveStreamPlayer.parentElement.classList.remove('red-alert-active');
                }
            } catch (err) {
                console.warn('Reset error:', err);
            }
        });
    }
}

function startVideoTelemetry() {
    stopVideoTelemetry();
    pollVideoTelemetry();
    telemetryInterval = setInterval(pollVideoTelemetry, 1000);
}

function stopVideoTelemetry() {
    if (telemetryInterval) {
        clearInterval(telemetryInterval);
        telemetryInterval = null;
    }
}

async function pollVideoTelemetry() {
    try {
        const res = await fetch('/api/stream/telemetry');
        if (!res.ok) return;
        const data = await res.json();

        // Active vehicles count
        dom.streamValTotal.textContent = data.counts ? data.counts.total : (data.vehicles ? data.vehicles.length : 0);

        const counts = data.counts || {};
        const parts = [];
        if (counts.arm_a) parts.push(`${counts.arm_a} in Arm A`);
        if (counts.arm_b) parts.push(`${counts.arm_b} in Arm B`);
        if (counts.apex) parts.push(`${counts.apex} in Apex`);
        dom.streamValBreakdown.textContent = parts.join(' | ') || 'Real-time vehicle tracking active';

        // Risk pill
        const lvl = data.risk_level || 0;
        dom.streamValRisk.className = 'risk-pill-lg';
        if (lvl === 0) {
            dom.streamValRisk.classList.add('pill-green');
            dom.streamValRisk.textContent = 'LEVEL 0 - SAFE';
        } else if (lvl === 1) {
            dom.streamValRisk.classList.add('pill-yellow');
            dom.streamValRisk.textContent = 'LEVEL 1 - CAUTION';
        } else if (lvl === 2) {
            dom.streamValRisk.classList.add('pill-yellow');
            dom.streamValRisk.textContent = 'LEVEL 2 - WARNING';
        } else {
            dom.streamValRisk.classList.add('pill-red');
            dom.streamValRisk.textContent = 'LEVEL 3 - CRITICAL (RED ALERT)';
        }
        dom.streamValTtc.textContent = data.min_ttc ? `Min TTC: ${data.min_ttc.toFixed(1)}s` : 'Active stream clear';

        // Centric Line live passing metrics & Red Alert State
        if (data.centric_line) {
            const cl = data.centric_line;
            if (dom.centricTotalCrossings) dom.centricTotalCrossings.textContent = cl.total_crossings;
            if (dom.centricUphillCrossings) dom.centricUphillCrossings.textContent = cl.uphill_crossings;
            if (dom.centricDownhillCrossings) dom.centricDownhillCrossings.textContent = cl.downhill_crossings;
            if (dom.centricLeftCrossings) dom.centricLeftCrossings.textContent = cl.left_lane_crossings || 0;
            if (dom.centricActiveTouching) dom.centricActiveTouching.textContent = cl.active_touching;

            // Handle Red Alert Banner & Video Player Border Glow
            const isRedAlert = Boolean(cl.red_alert);
            if (dom.videoRedAlertBanner) {
                dom.videoRedAlertBanner.style.display = isRedAlert ? 'flex' : 'none';
                if (isRedAlert && dom.redAlertBannerDesc) {
                    dom.redAlertBannerDesc.textContent = cl.red_alert_reason || 'Vehicle detected crossing centric trigger line in LEFT LANE!';
                }
            }

            if (dom.liveStreamPlayer && dom.liveStreamPlayer.parentElement) {
                if (isRedAlert) {
                    dom.liveStreamPlayer.parentElement.classList.add('red-alert-active');
                    dom.liveStreamPlayer.parentElement.classList.remove('amber-alert-active');
                } else {
                    dom.liveStreamPlayer.parentElement.classList.remove('red-alert-active');
                }
            }

            if (dom.centricLeftStatus) {
                if (isRedAlert) {
                    dom.centricLeftStatus.textContent = '🚨 RED ALERT ACTIVE!';
                    dom.centricLeftStatus.style.color = '#ef4444';
                    dom.centricLeftStatus.style.fontWeight = '700';
                } else {
                    dom.centricLeftStatus.textContent = `${cl.left_lane_crossings || 0} Left Passings (Safe)`;
                    dom.centricLeftStatus.style.color = '';
                    dom.centricLeftStatus.style.fontWeight = '';
                }
            }

            if (dom.centricStatusBadge) {
                if (isRedAlert) {
                    dom.centricStatusBadge.textContent = '🚨 RED ALERT: LEFT PASSING CAR ON CENTRIC LINE!';
                    dom.centricStatusBadge.style.borderColor = '#ef4444';
                    dom.centricStatusBadge.style.color = '#ef4444';
                    dom.centricStatusBadge.style.backgroundColor = '#fee2e2';
                    dom.centricStatusBadge.style.fontWeight = '800';
                } else if (cl.active_touching > 0) {
                    dom.centricStatusBadge.textContent = `⚡ ${cl.active_touching} Vehicle(s) Passing Centric Line Now!`;
                    dom.centricStatusBadge.style.borderColor = '#0284c7';
                    dom.centricStatusBadge.style.color = '#0284c7';
                    dom.centricStatusBadge.style.backgroundColor = '';
                    dom.centricStatusBadge.style.fontWeight = '700';
                } else {
                    dom.centricStatusBadge.textContent = `Centric Line Active • ${cl.total_crossings} Verified Passings`;
                    dom.centricStatusBadge.style.borderColor = '';
                    dom.centricStatusBadge.style.color = '';
                    dom.centricStatusBadge.style.backgroundColor = '';
                    dom.centricStatusBadge.style.fontWeight = '';
                }
            }
        }

        // Phase 2: Emergency Priority Preemption Banner & Border Glow for Live Video
        const ep = data.emergency_preemption || {};
        const isVideoPreempted = Boolean(ep.active);
        if (dom.videoEmergencyBanner) {
            dom.videoEmergencyBanner.style.display = isVideoPreempted ? 'flex' : 'none';
            if (isVideoPreempted && dom.videoEmergencyBannerDesc) {
                dom.videoEmergencyBannerDesc.textContent = `🚑 ${ep.count || 1} Ambulance active on corridor — Preemption in effect: signals overridden for emergency priority!`;
            }
        }

        if (dom.liveStreamPlayer && dom.liveStreamPlayer.parentElement) {
            if (isVideoPreempted && !Boolean(data.centric_line && data.centric_line.red_alert)) {
                dom.liveStreamPlayer.parentElement.classList.add('amber-alert-active');
            } else if (!isVideoPreempted) {
                dom.liveStreamPlayer.parentElement.classList.remove('amber-alert-active');
            }
        }

        // Signals
        if (data.signal_a) {
            setSignalUI(dom.streamSignalALight, dom.streamSignalAText, dom.streamSignalADesc, data.signal_a.state, data.signal_a.message);
        }
        if (data.signal_b) {
            setSignalUI(dom.streamSignalBLight, dom.streamSignalBText, dom.streamSignalBDesc, data.signal_b.state, data.signal_b.message);
        }
    } catch (err) {
        // Silently tolerate transient poll drops
    }
}

// -----------------------------------------------------------------------------
// 5. Incident Event Logs Section
// -----------------------------------------------------------------------------
function initLogsSection() {
    dom.btnReloadLogs.addEventListener('click', loadLogs);
}

async function loadLogs() {
    dom.tableLogsBody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">Loading incident records from database...</td></tr>';
    try {
        const res = await fetch('/api/logs?limit=50');
        const logs = await res.json();
        dom.tableLogsBody.innerHTML = '';

        if (!logs || logs.length === 0) {
            dom.tableLogsBody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No high-risk incident events logged yet. System in normal flow.</td></tr>';
            return;
        }

        logs.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${row.timestamp}</strong></td>
                <td>${row.risk_label} (${row.risk_level})</td>
                <td>${row.min_ttc ? row.min_ttc.toFixed(1) + 's' : '--'}</td>
                <td>${row.vehicles_count}</td>
                <td>${row.vehicle_classes || 'N/A'}</td>
                <td><strong>${row.signal_a_state}</strong></td>
                <td><strong>${row.signal_b_state}</strong></td>
                <td>${row.hazard_description || 'Routine passage'}</td>
            `;
            dom.tableLogsBody.appendChild(tr);
        });
    } catch (err) {
        dom.tableLogsBody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">Error loading logs: ${err.message}</td></tr>`;
    }
}
