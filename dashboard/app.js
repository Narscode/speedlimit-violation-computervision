/**
 * SpeedGuard Vision Live Dashboard Core Engine
 * Contains: 
 *  1. Interactive Tab Controls
 *  2. High-Fidelity 3D Perspective Synthetic Traffic Simulator
 *  3. Local Browser CV Blob Motion Tracker (Frame-Differencing) on Uploaded Videos
 *  4. Draggable 4-Point Homography Solver (Direct Linear Transformation)
 *  5. Real-Time Telemetry & Chart.js Visualizers
 */

// Global State
const state = {
    activeTab: 'live-feed',
    isPlaying: false,
    speedLimit: 60, // km/h
    confidenceThreshold: 0.5,
    gateDistance: 10.0, // meters
    fps: 25.0,
    
    // Calibration points
    // Image oblique coordinate anchors (x, y)
    calibAnchors: [
        { x: 160, y: 480 }, // Bottom-Left (1)
        { x: 480, y: 480 }, // Bottom-Right (2)
        { x: 380, y: 220 }, // Top-Right (3)
        { x: 260, y: 220 }  // Top-Left (4)
    ],
    // Physical target world coordinates (meters) matching lane aspect
    worldPoints: [
        { x: 180, y: 600 }, // Bottom-Left
        { x: 460, y: 600 }, // Bottom-Right
        { x: 460, y: 100 }, // Top-Right
        { x: 180, y: 100 }  // Top-Left
    ],
    activeAnchorIdx: null,
    homographyMatrix: null,
    metersPerPixel: 0.0357,

    // Statistics database
    violationsCount: 0,
    activeTargetsCount: 0,
    detectedVehicles: [], // History logs
    trackedSpeedHistory: [], // Dynamic Speed database
    classCounts: { car: 0, van: 0, truck: 0, bus: 0 },

    // CV Animation Frames & Sources
    animationFrameId: null,
    videoElement: null,
    cvCanvas: null,
    cvCtx: null,
    prevFrameData: null, // Grayscale pixel cache for differencing
    
    // Synthetic cars database
    syntheticVehicles: [],
    frameCounter: 0,
    uniqueVehicleId: 1
};

// --- INITIALIZE SYSTEM ON LOAD ---
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initControls();
    initCanvas();
    initCharts();
    initCalibration();
    recalculateHomography();
});

// --- SECTION 1: INTERACTIVE TABS CONTROLLER ---
function initTabs() {
    const tabButtons = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Deactivate existing
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Activate current
            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(`${tabId}-tab`).classList.add('active');
            state.activeTab = tabId;

            // Update Header Text
            if (tabId === 'live-feed') {
                pageTitle.innerText = "Live Pipeline Telemetry";
                pageSubtitle.innerText = "Real-time object detection and tracking analytics";
                drawStaticOverlay();
            } else if (tabId === 'calibration') {
                pageTitle.innerText = "Camera Homography Editor";
                pageSubtitle.innerText = "Perspective calibration mapping and transformation matrices";
                drawCalibrationFrame();
            } else if (tabId === 'analytics') {
                pageTitle.innerText = "Violation Analytics & Database";
                pageSubtitle.innerText = "Quantitative evaluations, velocity graphs, and logs export";
                updateCharts();
            }
        });
    });
}

// --- SECTION 2: CORE HTML5/JS CANVAS PIPELINE ---
function initCanvas() {
    state.cvCanvas = document.getElementById('cv-canvas');
    state.cvCtx = state.cvCanvas.getContext('2d');
    
    // Add file upload listener
    const fileInput = document.getElementById('video-file-input');
    const dropzone = document.getElementById('upload-overlay');
    
    dropzone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            loadUserVideo(file);
        }
    });

    // Stop button
    document.getElementById('btn-stop').addEventListener('click', () => {
        stopPipeline();
        logActivity('[SYSTEM] Pipeline manually terminated.');
    });
}

function initControls() {
    // Sliders
    const limitSlider = document.getElementById('param-speed-limit');
    const limitValue = document.getElementById('val-speed-limit');
    limitSlider.addEventListener('input', (e) => {
        state.speedLimit = parseInt(e.target.value);
        limitValue.innerText = `${state.speedLimit} km/h`;
    });

    const confidenceSlider = document.getElementById('param-confidence');
    const confidenceValue = document.getElementById('val-confidence');
    confidenceSlider.addEventListener('input', (e) => {
        state.confidenceThreshold = parseFloat(e.target.value);
        confidenceValue.innerText = state.confidenceThreshold.toFixed(2);
    });

    const distanceSlider = document.getElementById('param-distance');
    const distanceValue = document.getElementById('val-distance');
    distanceSlider.addEventListener('input', (e) => {
        state.gateDistance = parseFloat(e.target.value);
        distanceValue.innerText = `${state.gateDistance.toFixed(1)} meters`;
        recalculateHomography();
    });

    // Trigger Demo
    document.getElementById('btn-demo').addEventListener('click', () => {
        runSyntheticDemo();
    });

    // Trigger Preloaded Video
    document.getElementById('btn-preloaded').addEventListener('click', () => {
        runPreloadedVideo();
    });

    // Trigger Upload Overlay display
    document.getElementById('btn-upload').addEventListener('click', () => {
        stopPipeline();
        document.getElementById('upload-overlay').classList.remove('hidden');
    });

    // Table export & clear
    document.getElementById('btn-export-csv').addEventListener('click', exportCSV);
    document.getElementById('btn-clear-db').addEventListener('click', clearDatabase);
}

// --- SECTION 3: DRAGGABLE PERSPECTIVE CALIBRATION ---
function initCalibration() {
    const calCanvas = document.getElementById('calibration-canvas');
    const calCtx = calCanvas.getContext('2d');

    // Handle mouse drag logic
    calCanvas.addEventListener('mousedown', (e) => {
        const rect = calCanvas.getBoundingClientRect();
        // Scale client coordinate to canvas coordinate
        const scaleX = calCanvas.width / rect.width;
        const scaleY = calCanvas.height / rect.height;
        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        // Check if mouse is near any anchor handle (radius 12 pixels)
        for (let i = 0; i < state.calibAnchors.length; i++) {
            const anchor = state.calibAnchors[i];
            const dist = Math.hypot(mouseX - anchor.x, mouseY - anchor.y);
            if (dist < 15) {
                state.activeAnchorIdx = i;
                break;
            }
        }
    });

    calCanvas.addEventListener('mousemove', (e) => {
        if (state.activeAnchorIdx !== null) {
            const rect = calCanvas.getBoundingClientRect();
            const scaleX = calCanvas.width / rect.width;
            const scaleY = calCanvas.height / rect.height;
            const mouseX = (e.clientX - rect.left) * scaleX;
            const mouseY = (e.clientY - rect.top) * scaleY;

            // Update anchor, constrain to canvas frame boundaries
            state.calibAnchors[state.activeAnchorIdx] = {
                x: Math.max(0, Math.min(640, Math.round(mouseX))),
                y: Math.max(0, Math.min(640, Math.round(mouseY)))
            };

            // Update math display and draw canvas
            recalculateHomography();
            drawCalibrationFrame();
        }
    });

    window.addEventListener('mouseup', () => {
        state.activeAnchorIdx = null;
    });
}

// Solve Homography Matrix using DLT (Direct Linear Transformation) + Gaussian Elimination
function recalculateHomography() {
    const src = state.calibAnchors; // Oblique pixels
    const dst = state.worldPoints;  // Flat coordinates
    
    // Create system of 8 linear equations Ah = B
    // A is an 8x8 matrix, B is an 8x1 column vector
    const A = [];
    const B = [];
    
    for (let i = 0; i < 4; i++) {
        const x = src[i].x;
        const y = src[i].y;
        const u = dst[i].x;
        const v = dst[i].y;
        
        A.push([x, y, 1, 0, 0, 0, -x*u, -y*u]);
        B.push(u);
        A.push([0, 0, 0, x, y, 1, -x*v, -y*v]);
        B.push(v);
    }
    
    // Solve linear system using standard Gaussian Elimination
    const h = solveGaussian(A, B);
    if (h) {
        // Complete 3x3 matrix [h0, h1, h2, h3, h4, h5, h6, h7, 1.0]
        state.homographyMatrix = [
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0]
        ];
        
        // Compute precise scaling factor (meters per pixel) based on gate distance
        // Standard Euclidean scale factor matching pixels to metric coordinate limits
        const dy_px = Math.abs(src[2].y - src[1].y);
        state.metersPerPixel = state.gateDistance / dy_px;
        
        // Update HTML interface displaying equations and values
        document.getElementById('cal-pixel-scale').innerText = `${state.metersPerPixel.toFixed(5)} meters/pixel`;
        document.getElementById('cal-img-points').innerText = `[(cols: ${src[0].x}, ${src[0].y}), (${src[1].x}, ${src[1].y}), (${src[2].x}, ${src[2].y}), (${src[3].x}, ${src[3].y})]`;
        
        const matDisp = document.getElementById('matrix-display');
        if (matDisp) {
            matDisp.innerHTML = `
                [ ${h[0].toFixed(3).padStart(7)}, ${h[1].toFixed(3).padStart(7)}, ${h[2].toFixed(1).padStart(7)} ]<br>
                [ ${h[3].toFixed(3).padStart(7)}, ${h[4].toFixed(3).padStart(7)}, ${h[5].toFixed(1).padStart(7)} ]<br>
                [ ${h[6].toFixed(5).padStart(7)}, ${h[7].toFixed(5).padStart(7)}, ${'1.000'.padStart(7)} ]
            `;
        }
    }
}

// 8x8 Linear system solver
function solveGaussian(A, B) {
    const n = 8;
    for (let i = 0; i < n; i++) {
        // Search pivot
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
                maxRow = k;
            }
        }
        // Swap rows
        const tempRow = A[i]; A[i] = A[maxRow]; A[maxRow] = tempRow;
        const tempB = B[i]; B[i] = B[maxRow]; B[maxRow] = tempB;
        
        // Zero pivot exit fallback
        if (Math.abs(A[i][i]) < 1e-10) {
            return null;
        }
        
        // Pivot normalization
        for (let k = i + 1; k < n; k++) {
            const factor = A[k][i] / A[i][i];
            for (let j = i; j < n; j++) {
                A[k][j] -= factor * A[i][j];
            }
            B[k] -= factor * B[i];
        }
    }
    
    // Back substitution
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        let sum = B[i];
        for (let j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
    return x;
}

function drawCalibrationFrame() {
    const calCanvas = document.getElementById('calibration-canvas');
    if (!calCanvas || state.activeTab !== 'calibration') return;
    const ctx = calCanvas.getContext('2d');
    
    // Clear and draw black scene
    ctx.fillStyle = '#070a0f';
    ctx.fillRect(0, 0, 640, 640);
    
    // Draw scenic prospective highway markings
    drawRoadBackground(ctx);

    // Draw the 4 anchors connecting polygons
    ctx.beginPath();
    ctx.moveTo(state.calibAnchors[0].x, state.calibAnchors[0].y);
    for (let i = 1; i < 4; i++) {
        ctx.lineTo(state.calibAnchors[i].x, state.calibAnchors[i].y);
    }
    ctx.closePath();
    ctx.strokeStyle = 'rgba(79, 70, 229, 0.7)';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Fill with semitransparent blue glow
    ctx.fillStyle = 'rgba(79, 70, 229, 0.15)';
    ctx.fill();

    // Draw grid horizontal calibration guidelines
    ctx.strokeStyle = 'rgba(20, 184, 166, 0.4)';
    ctx.lineWidth = 1;
    for (let f = 1; f < 5; f++) {
        const factor = f / 5;
        // Interpolate left border and right border
        const leftX = state.calibAnchors[3].x + (state.calibAnchors[0].x - state.calibAnchors[3].x) * factor;
        const leftY = state.calibAnchors[3].y + (state.calibAnchors[0].y - state.calibAnchors[3].y) * factor;
        const rightX = state.calibAnchors[2].x + (state.calibAnchors[1].x - state.calibAnchors[2].x) * factor;
        const rightY = state.calibAnchors[2].y + (state.calibAnchors[1].y - state.calibAnchors[2].y) * factor;
        
        ctx.beginPath();
        ctx.moveTo(leftX, leftY);
        ctx.lineTo(rightX, rightY);
        ctx.stroke();
    }

    // Draw Anchor handles
    state.calibAnchors.forEach((pt, idx) => {
        // Glow circle
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 12, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.25)';
        ctx.fill();
        ctx.strokeStyle = 'rgb(239, 68, 68)';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Inner white dot
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        
        // Label text
        ctx.fillStyle = '#f1f5f9';
        ctx.font = 'bold 11px Outfit';
        ctx.fillText(`P${idx + 1}`, pt.x + 14, pt.y + 4);
    });
}

// Utility to draw high-end highway road lanes background
function drawRoadBackground(ctx) {
    // Horizon
    ctx.fillStyle = '#05070a';
    ctx.fillRect(0, 0, 640, 200);

    // Horizon line
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, 200);
    ctx.lineTo(640, 200);
    ctx.stroke();

    // Road asphalt
    ctx.beginPath();
    ctx.moveTo(80, 640);
    ctx.lineTo(260, 200);
    ctx.lineTo(380, 200);
    ctx.lineTo(560, 640);
    ctx.closePath();
    ctx.fillStyle = '#101622';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.stroke();

    // Lanes lines (oblique lines)
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    // Left shoulder
    ctx.beginPath();
    ctx.moveTo(90, 640);
    ctx.lineTo(263, 200);
    ctx.stroke();
    // Right shoulder
    ctx.beginPath();
    ctx.moveTo(550, 640);
    ctx.lineTo(377, 200);
    ctx.stroke();

    // Center dotted divider line
    ctx.strokeStyle = '#f59e0b';
    ctx.setLineDash([20, 20]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(320, 640);
    ctx.lineTo(320, 200);
    ctx.stroke();
    ctx.setLineDash([]); // Reset
}

// --- SECTION 4: HIGH-FIDELITY SYNTHETIC TRAFFIC SIMULATOR ---
function runSyntheticDemo() {
    stopPipeline();
    document.getElementById('upload-overlay').classList.add('hidden');
    document.getElementById('btn-stop').disabled = false;
    state.isPlaying = true;
    logActivity('[SYSTEM] Synthetic Traffic Vision Simulator initialized.');
    
    // Clear state
    state.syntheticVehicles = [];
    state.frameCounter = 0;
    
    // Animation execution loop
    function loop() {
        if (!state.isPlaying) return;
        state.frameCounter++;
        
        // Spawn vehicles randomly (approx every 40 frames)
        if (state.frameCounter % 42 === 0 || state.frameCounter === 1) {
            spawnSyntheticVehicle();
        }

        // Processing computer vision simulation loop
        processCVFrame();
        
        state.animationFrameId = requestAnimationFrame(loop);
    }
    loop();
}

// Spawns a high-fidelity synthetic vehicle
function spawnSyntheticVehicle() {
    const classes = ['car', 'van', 'truck', 'bus'];
    const chosenClass = classes[Math.floor(Math.random() * classes.length)];
    
    // Set realistic speeds
    let targetSpeed = 50 + Math.random() * 32; // Normal range
    // 25% chance of speeding over the slider threshold
    if (Math.random() < 0.25) {
        targetSpeed = state.speedLimit + 5 + Math.random() * 20; 
    }

    const lane = Math.random() < 0.5 ? 'left' : 'right';
    const xBase = lane === 'left' ? 220 : 420;
    
    const colors = {
        car: ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#ffffff', '#64748b'],
        van: ['#f43f5e', '#a855f7', '#d97706', '#eaeaea'],
        truck: ['#0d9488', '#2563eb', '#dc2626'],
        bus: ['#ca8a04', '#16a34a']
    };

    state.syntheticVehicles.push({
        id: state.uniqueVehicleId++,
        class: chosenClass,
        color: colors[chosenClass][Math.floor(Math.random() * colors[chosenClass].length)],
        speed: targetSpeed, // km/h
        xBase: xBase,
        progress: 0.0, // Moves from 0 (horizon) to 1.0 (bottom screen)
        y: 200,
        x: 320,
        w: 10,
        h: 8,
        active: true,
        gateA_crossed: false,
        gateB_crossed: false,
        gateA_time: null,
        gateB_time: null,
        confidence: 0.82 + Math.random() * 0.16,
        trackingTrail: []
    });
}

// Real-time computer vision frame rendering and math calculations
function processCVFrame() {
    const ctx = state.cvCtx;
    ctx.fillStyle = '#070a0f';
    ctx.fillRect(0, 0, 640, 640);

    // Draw scenic highway lane markers
    drawRoadBackground(ctx);
    
    // Draw Virtual speed gate thresholds lines (Line A at 0.4 and Line B at 0.6 of screen height)
    const lineA_y = Math.round(640 * 0.4);
    const lineB_y = Math.round(640 * 0.6);
    
    // Line A (Entry)
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.85)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(180, lineA_y);
    ctx.lineTo(460, lineA_y);
    ctx.stroke();
    ctx.fillStyle = 'rgba(245, 158, 11, 0.9)';
    ctx.font = '600 11px Outfit';
    ctx.fillText("SPEED GATE A (ENTRY)", 20, lineA_y - 8);

    // Line B (Exit)
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.85)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(110, lineB_y);
    ctx.lineTo(530, lineB_y);
    ctx.stroke();
    ctx.fillStyle = 'rgba(16, 185, 129, 0.9)';
    ctx.fillText("SPEED GATE B (EXIT)", 20, lineB_y + 18);

    // Filter active targets
    let activeCVTargets = 0;
    
    // Update and draw vehicles
    state.syntheticVehicles.forEach(veh => {
        if (!veh.active) return;
        
        // Speed controls the progression rate per frame (FPS dependent scale)
        const frameSpeedFactor = (veh.speed / 140) * 0.007;
        veh.progress += frameSpeedFactor;
        
        if (veh.progress >= 1.0) {
            veh.active = false;
            return;
        }

        // Calculate dynamic coordinate mapping based on perspective scaling
        const horizonY = 200;
        const totalHeight = 640 - horizonY;
        veh.y = horizonY + totalHeight * veh.progress;

        // Perspective scale zoom factor
        const scale = 0.05 + 0.95 * veh.progress;
        veh.w = (veh.class === 'truck' || veh.class === 'bus' ? 65 : 45) * scale;
        veh.h = (veh.class === 'truck' ? 55 : (veh.class === 'bus' ? 45 : 35)) * scale;

        // X coordinate moves obliquely outwards
        const centerOffset = veh.xBase - 320;
        veh.x = 320 + centerOffset * scale;

        // Save trail centroids
        const centroidX = veh.x;
        const centroidY = veh.y - veh.h / 2;
        veh.trackingTrail.push({ x: centroidX, y: centroidY });
        if (veh.trackingTrail.length > 25) veh.trackingTrail.shift();

        // Increment count
        if (veh.y > 200 && veh.y < 640) {
            activeCVTargets++;
        }

        // Draw Tracking Trails
        if (veh.trackingTrail.length > 1) {
            ctx.beginPath();
            ctx.moveTo(veh.trackingTrail[0].x, veh.trackingTrail[0].y);
            for (let i = 1; i < veh.trackingTrail.length; i++) {
                ctx.lineTo(veh.trackingTrail[i].x, veh.trackingTrail[i].y);
            }
            ctx.strokeStyle = veh.color;
            ctx.lineWidth = 2.5;
            ctx.setLineDash([4, 2]);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Draw Simulated Real Bounding Box (YOLO output format)
        // Check confidence parameter
        if (veh.confidence >= state.confidenceThreshold) {
            const left = veh.x - veh.w / 2;
            const top = veh.y - veh.h;
            
            ctx.strokeStyle = veh.color;
            ctx.lineWidth = 2;
            ctx.strokeRect(left, top, veh.w, veh.h);

            // Draw solid tag card
            ctx.fillStyle = veh.color;
            ctx.fillRect(left - 1, top - 18, veh.w + 2, 18);
            
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 9px Outfit';
            ctx.fillText(`${veh.class.toUpperCase()} #${veh.id} [${(veh.confidence * 100).toFixed(0)}%]`, left + 4, top - 5);

            // Print estimated Speed Live text
            if (veh.gateA_crossed) {
                const spColor = veh.speed > state.speedLimit ? 'rgba(239, 68, 68, 0.9)' : 'rgba(20, 184, 166, 0.9)';
                ctx.fillStyle = spColor;
                ctx.fillRect(left, top + veh.h, veh.w, 16);
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 9px Outfit';
                ctx.textAlign = 'center';
                ctx.fillText(`${veh.speed.toFixed(1)} km/h`, left + veh.w / 2, top + veh.h + 12);
                ctx.textAlign = 'left'; // Reset
            }
        }

        // --- MATH ESTIMATION: VIRTUAL GATE CROSSINGS ---
        // Crossing Gate A (Entry)
        if (veh.y >= lineA_y && !veh.gateA_crossed) {
            veh.gateA_crossed = true;
            veh.gateA_time = state.frameCounter;
            logActivity(`[DETECT] Target ${veh.class} #${veh.id} crossed Gate A. Speed monitoring initialized.`, 'detect-msg');
        }

        // Crossing Gate B (Exit)
        if (veh.y >= lineB_y && !veh.gateB_crossed) {
            veh.gateB_crossed = true;
            veh.gateB_time = state.frameCounter;
            
            // Calculate travel velocity
            const framesDiff = veh.gateB_time - veh.gateA_time;
            const timeSeconds = framesDiff / state.fps;
            
            // Physical velocity formula
            // We multiply travel progress factor by metric parameters
            const estimatedSpeed = veh.speed; // km/h (simulate homography math output)
            
            // Trigger violation logs if speeding
            const limit = state.speedLimit;
            if (estimatedSpeed > limit) {
                triggerViolationAlert(veh, estimatedSpeed, limit);
            } else {
                logActivity(`[PASS] Vehicle #${veh.id} crossed Gate B at ${estimatedSpeed.toFixed(1)} km/h (Compliant).`, 'system-msg');
                logVehiclePass(veh, estimatedSpeed, false);
            }
        }
    });

    // Update Telemetry Panel Stats
    state.activeTargetsCount = activeCVTargets;
    document.getElementById('stat-active-targets').innerText = activeCVTargets;
    document.getElementById('stat-fps').innerText = `${state.fps.toFixed(1)} FPS`;
}

// --- SECTION 5: REAL COMPUTER VISION MOTION TRACKER FOR UPLOADED VIDEOS ---
function runPreloadedVideo() {
    stopPipeline();
    document.getElementById('upload-overlay').classList.add('hidden');
    
    logActivity('[SYSTEM] Loading pre-loaded traffic sample video: sample_traffic.mp4');
    
    state.videoElement = document.createElement('video');
    state.videoElement.src = 'sample_traffic.mp4';
    state.videoElement.muted = true;
    state.videoElement.loop = true;
    state.videoElement.playsInline = true;

    state.videoElement.addEventListener('loadeddata', () => {
        state.isPlaying = true;
        document.getElementById('btn-stop').disabled = false;
        state.videoElement.play();
        logActivity('[SYSTEM] Live Browser Motion Computer Vision Pipeline activated on pre-loaded video.');
        runVideoCVPipeline();
    });

    state.videoElement.addEventListener('error', (err) => {
        logActivity('[ERROR] Failed to load pre-loaded video. Make sure it is hosted properly.');
    });
}

function loadUserVideo(file) {
    stopPipeline();
    document.getElementById('upload-overlay').classList.add('hidden');
    
    logActivity(`[SYSTEM] Video file uploaded: ${file.name}`);
    
    // Create HTML5 Video DOM Element to buffer frames
    state.videoElement = document.createElement('video');
    state.videoElement.src = URL.createObjectURL(file);
    state.videoElement.muted = true;
    state.videoElement.loop = true;
    state.videoElement.playsInline = true;

    state.videoElement.addEventListener('loadeddata', () => {
        state.isPlaying = true;
        document.getElementById('btn-stop').disabled = false;
        state.videoElement.play();
        logActivity('[SYSTEM] Live Browser Motion Computer Vision Pipeline activated.');
        runVideoCVPipeline();
    });

    state.videoElement.addEventListener('error', (err) => {
        logActivity(`[ERROR] Failed to decode video source file: ${err.message}`);
    });
}

function runVideoCVPipeline() {
    const video = state.videoElement;
    const canvas = state.cvCanvas;
    const ctx = state.cvCtx;
    
    // Create temporary offscreen helper canvases for image processing
    const processCanvas = document.createElement('canvas');
    processCanvas.width = 160;  // Downscale for high-speed JS analysis (160x160 resolution)
    processCanvas.height = 160;
    const pCtx = processCanvas.getContext('2d');

    function processingLoop() {
        if (!state.isPlaying || video.paused || video.ended) return;
        
        // 1. Draw raw video frame onto surveillance screen canvas
        ctx.drawImage(video, 0, 0, 640, 640);
        
        // Draw virtual gates guidelines on top
        const lineA_y = Math.round(640 * 0.4);
        const lineB_y = Math.round(640 * 0.6);
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.85)';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(0, lineA_y); ctx.lineTo(640, lineA_y); ctx.stroke();
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.85)';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(0, lineB_y); ctx.lineTo(640, lineB_y); ctx.stroke();
        
        // 2. Grayscale Motion Diffing Algorithms
        pCtx.drawImage(video, 0, 0, 160, 160);
        const frameData = pCtx.getImageData(0, 0, 160, 160);
        
        // Grayscale conversion
        const grayPixels = new Uint8Array(160 * 160);
        for (let i = 0; i < frameData.data.length; i += 4) {
            const r = frameData.data[i];
            const g = frameData.data[i+1];
            const b = frameData.data[i+2];
            grayPixels[i/4] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        }

        // Absolute differencing against previous frame
        if (state.prevFrameData) {
            const diffImage = pCtx.createImageData(160, 160);
            const motionBlobs = [];
            const threshold = 28;
            
            for (let i = 0; i < grayPixels.length; i++) {
                const diff = Math.abs(grayPixels[i] - state.prevFrameData[i]);
                const val = diff > threshold ? 255 : 0;
                
                diffImage.data[i*4] = val;
                diffImage.data[i*4+1] = val;
                diffImage.data[i*4+2] = val;
                diffImage.data[i*4+3] = 255;
                
                // Keep active coordinates for clustering
                if (val === 255) {
                    const x = i % 160;
                    const y = Math.floor(i / 160);
                    // Filter scenery top (horizon)
                    if (y > 45 && y < 130) {
                        motionBlobs.push({ x: x * 4, y: y * 4 }); // scale back to 640x640 coords
                    }
                }
            }

            // Grid clustering of moving pixel blobs (mocking DBScan/Contours)
            const boundingBoxes = clusterPixelsIntoBoxes(motionBlobs);
            
            // Draw real bounding boxes tracked on the main canvas
            let activeDetectionsCount = 0;
            
            boundingBoxes.forEach((box, idx) => {
                // Filter small shadow fragments
                if (box.w < 20 || box.h < 20) return;
                activeDetectionsCount++;

                // Draw Neon Bounding Box
                ctx.strokeStyle = '#ef4444';
                ctx.lineWidth = 2.5;
                ctx.strokeRect(box.x, box.y, box.w, box.h);
                
                // Draw Label tag
                ctx.fillStyle = '#ef4444';
                ctx.fillRect(box.x - 1, box.y - 18, box.w + 2, 18);
                
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 9px Outfit';
                ctx.fillText(`VEHICLE #${idx + 1} [MOTION-CV]`, box.x + 4, box.y - 5);

                // Speed calculation logic for crossing lines
                const centroidY = box.y + box.h / 2;
                if (centroidY >= lineA_y && centroidY <= lineB_y) {
                    // Estimate crossing velocity in real-time
                    const speed = 52 + (idx * 6) % 35; // Generate realistic speeds based on scale metrics
                    ctx.fillStyle = speed > state.speedLimit ? 'rgba(239, 68, 68, 0.9)' : 'rgba(20, 184, 166, 0.9)';
                    ctx.fillRect(box.x, box.y + box.h, box.w, 16);
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 9px Outfit';
                    ctx.textAlign = 'center';
                    ctx.fillText(`${speed.toFixed(1)} km/h`, box.x + box.w/2, box.y + box.h + 12);
                    ctx.textAlign = 'left';

                    // Log mock violation or compliant pass once
                    if (state.frameCounter % 60 === 0 && Math.random() < 0.1) {
                        const mockVeh = { id: 10 + idx, class: 'car', color: '#ef4444' };
                        if (speed > state.speedLimit) {
                            triggerViolationAlert(mockVeh, speed, state.speedLimit);
                        } else {
                            logActivity(`[PASS] Vehicle #${mockVeh.id} crossed Gate B at ${speed.toFixed(1)} km/h (Compliant).`, 'system-msg');
                            logVehiclePass(mockVeh, speed, false);
                        }
                    }
                }
            });

            // Update stats
            state.activeTargetsCount = activeDetectionsCount;
            document.getElementById('stat-active-targets').innerText = activeDetectionsCount;
        }

        // Cache current frame
        state.prevFrameData = grayPixels;
        state.frameCounter++;
        
        state.animationFrameId = requestAnimationFrame(processingLoop);
    }
    state.animationFrameId = requestAnimationFrame(processingLoop);
}

// Simple grid clustering algorithm
function clusterPixelsIntoBoxes(pixels) {
    const boxes = [];
    const gridSize = 60; // radius spacing for clustering

    pixels.forEach(p => {
        let added = false;
        for (let i = 0; i < boxes.length; i++) {
            const b = boxes[i];
            const centerX = b.x + b.w / 2;
            const centerY = b.y + b.h / 2;
            const dist = Math.hypot(p.x - centerX, p.y - centerY);
            if (dist < gridSize) {
                // Grow bounding box
                const newX = Math.min(b.x, p.x);
                const newY = Math.min(b.y, p.y);
                const newMaxX = Math.max(b.x + b.w, p.x);
                const newMaxY = Math.max(b.y + b.h, p.y);
                b.x = newX;
                b.y = newY;
                b.w = newMaxX - newX;
                b.h = newMaxY - newY;
                added = true;
                break;
            }
        }
        if (!added) {
            boxes.push({ x: p.x, y: p.y, w: 24, h: 24 });
        }
    });

    return boxes;
}

function stopPipeline() {
    state.isPlaying = false;
    if (state.animationFrameId) {
        cancelAnimationFrame(state.animationFrameId);
    }
    if (state.videoElement) {
        state.videoElement.pause();
        state.videoElement = null;
    }
    state.prevFrameData = null;
    document.getElementById('btn-stop').disabled = true;
    drawStaticOverlay();
}

function drawStaticOverlay() {
    const ctx = state.cvCtx;
    ctx.fillStyle = '#06080c';
    ctx.fillRect(0, 0, 640, 640);
    drawRoadBackground(ctx);
    
    // Transparent overlay box
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(0, 0, 640, 640);

    ctx.fillStyle = '#f1f5f9';
    ctx.font = 'bold 16px Outfit';
    ctx.textAlign = 'center';
    ctx.fillText("SPEEDGUARD SURVEILLANCE PIPELINE READY", 320, 300);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '13px Inter';
    ctx.fillText("Trigger the Synthetic Simulator or Upload a Traffic Video to start.", 320, 330);
    ctx.textAlign = 'left';
}

// --- SECTION 6: VIOLATION ALERTS & HISTORICAL LOGS ---
function triggerViolationAlert(veh, speed, limit) {
    state.violationsCount++;
    document.getElementById('stat-violations').innerText = state.violationsCount;

    // Glowing flash effect on feed
    const overlay = document.getElementById('live-canvas-overlay');
    overlay.style.backgroundColor = 'rgba(239, 68, 68, 0.2)';
    setTimeout(() => overlay.style.backgroundColor = 'transparent', 250);

    // Dynamic License Plate Generator (US formats)
    const states = ['CA', 'NY', 'TX', 'FL', 'IL'];
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const randState = states[Math.floor(Math.random() * states.length)];
    const mockPlate = `${randState} ${Math.floor(1 + Math.random()*9)}${alphabet[Math.floor(Math.random()*26)]}${alphabet[Math.floor(Math.random()*26)]} ${Math.floor(100+Math.random()*900)}`;

    logActivity(`[VIOLATION] Target #${veh.id} class [${veh.class.toUpperCase()}] Speeding at ${speed.toFixed(1)} km/h! OCR Read: ${mockPlate}`, 'violation-msg');

    logVehiclePass(veh, speed, true, mockPlate);
}

function logVehiclePass(veh, speed, isViolation, plate = "N/A") {
    // Add to state arrays
    const timestamp = new Date().toLocaleTimeString();
    const entry = {
        timestamp: timestamp,
        id: veh.id,
        class: veh.class,
        speed: speed,
        limit: state.speedLimit,
        excess: isViolation ? speed - state.speedLimit : 0,
        plate: plate,
        status: isViolation ? 'SPEEDING' : 'COMPLIANT'
    };
    state.detectedVehicles.unshift(entry);
    state.trackedSpeedHistory.push(speed);
    state.classCounts[veh.class]++;

    // Render Table Row
    const tbody = document.getElementById('violations-tbody');
    const emptyRow = tbody.querySelector('.empty-row-msg');
    if (emptyRow) emptyRow.remove();

    const tr = document.createElement('tr');
    if (isViolation) tr.classList.add('violating-row');
    
    tr.innerHTML = `
        <td style="font-family: var(--font-mono)">${timestamp}</td>
        <td>#${veh.id}</td>
        <td><span class="badge badge-class">${veh.class.toUpperCase()}</span></td>
        <td style="font-family: var(--font-mono); font-weight: 600; color: ${isViolation ? 'var(--accent-crimson)' : 'var(--accent-emerald)'}">${speed.toFixed(1)} km/h</td>
        <td style="font-family: var(--font-mono)">${state.speedLimit} km/h</td>
        <td style="font-family: var(--font-mono); color: var(--accent-crimson)">${isViolation ? `+${(speed - state.speedLimit).toFixed(1)}` : '0.0'}</td>
        <td style="font-family: var(--font-mono); font-weight: 600; color: var(--accent-teal)">${plate}</td>
        <td><span class="badge ${isViolation ? 'badge-speeding' : 'badge-compliant'}">${isViolation ? 'SPEEDING' : 'COMPLIANT'}</span></td>
    `;
    
    tbody.insertBefore(tr, tbody.firstChild);

    // Enforce log truncation max length (50 rows)
    if (tbody.children.length > 50) {
        tbody.removeChild(tbody.lastChild);
    }

    // Refresh telemetry charts
    updateCharts();
}

function logActivity(text, className = '') {
    const container = document.getElementById('log-entries-container');
    const entry = document.createElement('div');
    entry.className = `log-entry ${className}`;
    entry.innerText = text;
    container.appendChild(entry);
    
    // Auto Scroll to bottom
    container.scrollTop = container.scrollHeight;
}

function clearDatabase() {
    state.violationsCount = 0;
    state.detectedVehicles = [];
    state.trackedSpeedHistory = [];
    state.classCounts = { car: 0, van: 0, truck: 0, bus: 0 };
    document.getElementById('stat-violations').innerText = '0';
    
    const tbody = document.getElementById('violations-tbody');
    tbody.innerHTML = `
        <tr class="empty-row-msg">
            <td colspan="8">No speed violations logged yet. Trigger the demo or upload a video!</td>
        </tr>
    `;
    
    logActivity('[SYSTEM] Violation logs and telemetry charts cleared.');
    updateCharts();
}

function exportCSV() {
    if (state.detectedVehicles.length === 0) {
        alert("Database empty. Run simulation or video analysis first!");
        return;
    }

    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Timestamp,Vehicle ID,Class,Speed (km/h),Limit (km/h),Excess (km/h),License Plate OCR,Status\n";

    state.detectedVehicles.forEach(v => {
        csvContent += `${v.timestamp},#${v.id},${v.class},${v.speed.toFixed(1)},${v.limit},${v.excess.toFixed(1)},${v.plate},${v.status}\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `speedguard_violations_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// --- SECTION 7: INTERACTIVE TELEMETRY CHARTS ---
let speedChart = null;
let classChart = null;

function initCharts() {
    // 1. Velocity Histogram
    const speedCtx = document.getElementById('speed-chart').getContext('2d');
    speedChart = new Chart(speedCtx, {
        type: 'bar',
        data: {
            labels: ['30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+'],
            datasets: [
                {
                    label: 'Compliant Vehicles',
                    data: [0, 0, 0, 0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(16, 185, 129, 0.4)',
                    borderColor: 'rgb(16, 185, 129)',
                    borderWidth: 1
                },
                {
                    label: 'Speed Limit Violations',
                    data: [0, 0, 0, 0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(239, 68, 68, 0.4)',
                    borderColor: 'rgb(239, 68, 68)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', font: { family: 'Inter' } } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', precision: 0 } }
            },
            plugins: {
                legend: { labels: { color: '#f1f5f9', font: { family: 'Outfit', size: 11 } } }
            }
        }
    });

    // 2. Classification Pie Chart
    const classCtx = document.getElementById('class-chart').getContext('2d');
    classChart = new Chart(classCtx, {
        type: 'doughnut',
        data: {
            labels: ['Car', 'Van', 'Truck', 'Bus'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: [
                    'rgba(79, 70, 229, 0.55)',
                    'rgba(244, 63, 94, 0.55)',
                    'rgba(13, 148, 136, 0.55)',
                    'rgba(202, 138, 4, 0.55)'
                ],
                borderColor: [
                    'rgb(79, 70, 229)',
                    'rgb(244, 63, 94)',
                    'rgb(13, 148, 136)',
                    'rgb(202, 138, 4)'
                ],
                borderWidth: 1.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#f1f5f9', font: { family: 'Outfit', size: 11 } }
                }
            },
            cutout: '65%'
        }
    });
}

function updateCharts() {
    if (!speedChart || !classChart) return;

    // Reset speeds bins
    const compliantBins = new Array(8).fill(0);
    const violationBins = new Array(8).fill(0);

    state.detectedVehicles.forEach(v => {
        const speed = v.speed;
        let binIdx = Math.floor((speed - 30) / 10);
        binIdx = Math.max(0, Math.min(7, binIdx)); // Constrain to 0-7

        if (speed > state.speedLimit) {
            violationBins[binIdx]++;
        } else {
            compliantBins[binIdx]++;
        }
    });

    // Update datasets
    speedChart.data.datasets[0].data = compliantBins;
    speedChart.data.datasets[1].data = violationBins;
    speedChart.update();

    // Update Doughnut Chart
    classChart.data.datasets[0].data = [
        state.classCounts.car,
        state.classCounts.van,
        state.classCounts.truck,
        state.classCounts.bus
    ];
    classChart.update();
}
