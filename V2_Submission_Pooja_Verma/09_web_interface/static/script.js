/* ═══════════════════════════════════════════════════════════
   Human Keystroke Simulator — Interactive Script
   ═══════════════════════════════════════════════════════════ */

// ─── State ────────────────────────────────────────────────
let isSimulating = false;
let simulationTimeout = null;
let currentKeystrokes = [];
let ddTimings = [];
let selectedSpeed = 'medium';

// ─── DOM Elements ─────────────────────────────────────────
const inputText = document.getElementById('input-text');
const simulateBtn = document.getElementById('simulate-btn');
const stopBtn = document.getElementById('stop-btn');
const clearBtn = document.getElementById('clear-btn');
const typedText = document.getElementById('typed-text');
const cursor = document.getElementById('cursor');
const progressFill = document.getElementById('progress-fill');
const simStatus = document.getElementById('sim-status');
const charCount = document.getElementById('char-count');
const evaluateBtn = document.getElementById('evaluate-btn');
const speedButtons = document.querySelectorAll('.speed-btn');
const timingChartContainer = document.getElementById('timing-chart-container');
const evaluationPanel = document.getElementById('evaluation-panel');
const timingCanvas = document.getElementById('timing-chart');

// ─── Speed Selection ──────────────────────────────────────
speedButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        speedButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedSpeed = btn.dataset.speed;
    });
});

// ─── Character Count ──────────────────────────────────────
inputText.addEventListener('input', () => {
    charCount.textContent = inputText.value.length;
});
charCount.textContent = inputText.value.length;

// ─── Clear Button ─────────────────────────────────────────
clearBtn.addEventListener('click', () => {
    inputText.value = '';
    charCount.textContent = '0';
    typedText.textContent = '';
    progressFill.style.width = '0%';
    timingChartContainer.style.display = 'none';
    evaluationPanel.style.display = 'none';
    updateMetrics(null);
    setStatus('idle', 'Ready');
});

// ─── Simulate Button ─────────────────────────────────────
simulateBtn.addEventListener('click', startSimulation);
stopBtn.addEventListener('click', stopSimulation);

async function startSimulation() {
    const text = inputText.value.trim();
    if (!text) return;
    
    // Reset display
    typedText.textContent = '';
    progressFill.style.width = '0%';
    currentKeystrokes = [];
    ddTimings = [];
    timingChartContainer.style.display = 'none';
    evaluationPanel.style.display = 'none';
    updateMetrics(null);
    
    // Update UI state
    isSimulating = true;
    simulateBtn.disabled = true;
    stopBtn.disabled = false;
    evaluateBtn.disabled = true;
    cursor.classList.add('typing');
    setStatus('running', 'Typing...');
    
    try {
        // Call API to get keystroke data
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, speed: selectedSpeed })
        });
        
        if (!response.ok) throw new Error('API error');
        
        const data = await response.json();
        currentKeystrokes = data.keystrokes;
        
        // Start the typing animation
        await animateTyping(data.keystrokes, data.metrics);
        
    } catch (err) {
        console.error('Simulation error:', err);
        setStatus('idle', 'Error — Is the server running?');
        isSimulating = false;
        simulateBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

async function animateTyping(keystrokes, metrics) {
    const totalChars = keystrokes.length;
    
    for (let i = 0; i < keystrokes.length; i++) {
        if (!isSimulating) break;
        
        const ks = keystrokes[i];
        const delay = i === 0 ? 100 : ks.dd_time * 1000; // Convert to ms
        
        // Wait for the delay
        await new Promise(resolve => {
            simulationTimeout = setTimeout(resolve, delay);
        });
        
        if (!isSimulating) break;
        
        // Display the character
        const charSpan = document.createElement('span');
        charSpan.textContent = ks.char;
        charSpan.classList.add('key-flash');
        typedText.appendChild(charSpan);
        
        // Track DD timings for chart
        if (i > 0) ddTimings.push(ks.dd_time * 1000);
        
        // Update progress
        const progress = ((i + 1) / totalChars) * 100;
        progressFill.style.width = progress + '%';
        
        // Update live metrics every 5 keystrokes
        if (i % 5 === 0 || i === totalChars - 1) {
            updateLiveMetrics(keystrokes, i + 1);
        }
        
        // Update chart every 10 keystrokes
        if (ddTimings.length > 2 && (i % 10 === 0 || i === totalChars - 1)) {
            drawTimingChart();
        }
        
        // Auto-scroll the display
        const display = document.getElementById('typing-display');
        display.scrollTop = display.scrollHeight;
    }
    
    // Simulation complete
    isSimulating = false;
    simulateBtn.disabled = false;
    stopBtn.disabled = true;
    evaluateBtn.disabled = false;
    cursor.classList.remove('typing');
    setStatus('done', 'Complete');
    
    // Final metrics update
    updateMetrics(metrics);
    drawTimingChart();
}

function stopSimulation() {
    isSimulating = false;
    if (simulationTimeout) {
        clearTimeout(simulationTimeout);
    }
    simulateBtn.disabled = false;
    stopBtn.disabled = true;
    evaluateBtn.disabled = currentKeystrokes.length < 5;
    cursor.classList.remove('typing');
    setStatus('idle', 'Stopped');
}

// ─── Status Updates ───────────────────────────────────────
function setStatus(type, text) {
    simStatus.innerHTML = `<span class="status-${type}">${text}</span>`;
}

// ─── Metrics Updates ──────────────────────────────────────
function updateMetrics(metrics) {
    const fields = [
        { id: 'metric-wpm', key: 'wpm', max: 120 },
        { id: 'metric-cpm', key: 'cpm', max: 600 },
        { id: 'metric-avg-dd', key: 'avg_dd_ms', max: 400 },
        { id: 'metric-std-dd', key: 'std_dd_ms', max: 200 },
        { id: 'metric-avg-hold', key: 'avg_hold_ms', max: 200 },
        { id: 'metric-total', key: 'total_time_s', max: 60 },
    ];
    
    fields.forEach(f => {
        const card = document.getElementById(f.id);
        const valueEl = card.querySelector('.metric-value');
        const barFill = card.querySelector('.metric-bar-fill');
        
        if (metrics && metrics[f.key] !== undefined) {
            const val = metrics[f.key];
            valueEl.textContent = typeof val === 'number' ? 
                (val >= 100 ? Math.round(val) : val.toFixed(1)) : val;
            barFill.style.width = Math.min(100, (val / f.max) * 100) + '%';
        } else {
            valueEl.textContent = '—';
            barFill.style.width = '0%';
        }
    });
}

function updateLiveMetrics(keystrokes, currentIdx) {
    const current = keystrokes.slice(0, currentIdx);
    if (current.length < 2) return;
    
    const dd = current.slice(1).map(k => k.dd_time);
    const holds = current.map(k => k.hold_time);
    const elapsed = current[current.length - 1].key_up - current[0].key_down;
    const words = current.filter(k => k.char === ' ').length + 1;
    
    const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const med = arr => {
        const sorted = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };
    const std = arr => {
        const m = avg(arr);
        return Math.sqrt(arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / arr.length);
    };
    
    const metrics = {
        wpm: elapsed > 0 ? (words / (elapsed / 60)).toFixed(1) : 0,
        cpm: elapsed > 0 ? (current.length / (elapsed / 60)).toFixed(1) : 0,
        avg_dd_ms: (avg(dd) * 1000).toFixed(1),
        std_dd_ms: (std(dd) * 1000).toFixed(1),
        avg_hold_ms: (avg(holds) * 1000).toFixed(1),
        total_time_s: elapsed.toFixed(1),
    };
    
    updateMetrics(metrics);
}

// ─── Timing Chart ─────────────────────────────────────────
function drawTimingChart() {
    if (ddTimings.length < 3) return;
    
    timingChartContainer.style.display = 'block';
    
    const canvas = timingCanvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    // Set canvas size
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 200 * dpr;
    canvas.style.height = '200px';
    ctx.scale(dpr, dpr);
    
    const w = rect.width;
    const h = 200;
    const padding = { top: 20, right: 20, bottom: 30, left: 50 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;
    
    // Clear
    ctx.fillStyle = '#0f1420';
    ctx.fillRect(0, 0, w, h);
    
    // Data (last 100 points max)
    const data = ddTimings.slice(-100);
    const maxVal = Math.min(800, Math.max(...data) * 1.2);
    const minVal = 0;
    
    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (plotH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(w - padding.right, y);
        ctx.stroke();
        
        // Labels
        const val = maxVal - (maxVal / 4) * i;
        ctx.fillStyle = '#5a6375';
        ctx.font = '10px "JetBrains Mono"';
        ctx.textAlign = 'right';
        ctx.fillText(Math.round(val) + 'ms', padding.left - 8, y + 3);
    }
    
    // Plot area
    const stepX = plotW / Math.max(data.length - 1, 1);
    
    // Gradient fill
    const gradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
    gradient.addColorStop(0, 'rgba(59,130,246,0.2)');
    gradient.addColorStop(1, 'rgba(59,130,246,0.0)');
    
    ctx.beginPath();
    ctx.moveTo(padding.left, h - padding.bottom);
    
    for (let i = 0; i < data.length; i++) {
        const x = padding.left + i * stepX;
        const y = padding.top + plotH - (data[i] / maxVal) * plotH;
        ctx.lineTo(x, Math.max(padding.top, Math.min(y, h - padding.bottom)));
    }
    
    ctx.lineTo(padding.left + (data.length - 1) * stepX, h - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
    
    // Line
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
        const x = padding.left + i * stepX;
        const y = padding.top + plotH - (data[i] / maxVal) * plotH;
        const clampedY = Math.max(padding.top, Math.min(y, h - padding.bottom));
        
        if (i === 0) ctx.moveTo(x, clampedY);
        else ctx.lineTo(x, clampedY);
    }
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    
    // Label
    ctx.fillStyle = '#8b95a8';
    ctx.font = '11px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Keystroke Index →', w / 2, h - 5);
    
    // Average line
    const avgDD = data.reduce((a, b) => a + b, 0) / data.length;
    const avgY = padding.top + plotH - (avgDD / maxVal) * plotH;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, avgY);
    ctx.lineTo(w - padding.right, avgY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    ctx.fillStyle = '#f59e0b';
    ctx.font = '10px "JetBrains Mono"';
    ctx.textAlign = 'left';
    ctx.fillText(`avg: ${Math.round(avgDD)}ms`, w - padding.right - 80, avgY - 5);
}

// ─── Evaluate Button ──────────────────────────────────────
evaluateBtn.addEventListener('click', async () => {
    if (currentKeystrokes.length < 5) return;
    
    evaluateBtn.disabled = true;
    evaluateBtn.textContent = 'Evaluating...';
    
    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ keystrokes: currentKeystrokes })
        });
        
        if (!response.ok) throw new Error('Evaluation failed');
        
        const data = await response.json();
        displayEvaluation(data);
        
    } catch (err) {
        console.error('Evaluation error:', err);
        alert('Evaluation failed. Make sure the server is running.');
    }
    
    evaluateBtn.disabled = false;
    evaluateBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.5"/>
            <path d="M5 8l2 2 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Evaluate Realism`;
});

function displayEvaluation(data) {
    evaluationPanel.style.display = 'block';
    
    const evalGrid = document.getElementById('eval-grid');
    evalGrid.innerHTML = '';
    
    // Model cards
    const modelEntries = [
        { name: 'Random Forest', data: data.random_forest },
        { name: 'Gradient Boosting', data: data.gradient_boosting },
    ];
    
    modelEntries.forEach(entry => {
        const prob = entry.data.human_probability;
        const isHuman = prob > 0.5;
        
        const card = document.createElement('div');
        card.className = 'eval-card';
        card.innerHTML = `
            <div class="model-name">${entry.name}</div>
            <div class="verdict ${isHuman ? 'human' : 'synthetic'}">${entry.data.verdict}</div>
            <div class="probability">Human probability: ${(prob * 100).toFixed(1)}%</div>
            <div class="prob-bar">
                <div class="prob-bar-fill ${prob > 0.5 ? 'high' : 'low'}" style="width: ${prob * 100}%"></div>
            </div>
        `;
        evalGrid.appendChild(card);
    });
    
    // Overall verdict
    const overall = document.createElement('div');
    overall.className = 'overall-verdict';
    const isHumanLike = data.overall_verdict === 'Human-like';
    overall.innerHTML = `
        <div class="label">Overall Assessment</div>
        <div class="value" style="color: ${isHumanLike ? 'var(--accent-emerald)' : 'var(--accent-red)'}">
            ${data.overall_verdict}
        </div>
        <div class="confidence">Confidence: ${data.confidence}%</div>
    `;
    evalGrid.appendChild(overall);
    
    // Smooth scroll to evaluation
    evaluationPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ─── Window Resize Handler ────────────────────────────────
window.addEventListener('resize', () => {
    if (ddTimings.length > 2) drawTimingChart();
});
