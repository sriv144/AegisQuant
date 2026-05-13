// ── AegisQuant Dashboard — Phase 3 ──────────────────────────────────────────
// WebSocket real-time, Nifty50 benchmark, trade reasoning, JWT auth
// ─────────────────────────────────────────────────────────────────────────────

let pnlChartInstance = null;
let exposureChartInstance = null;
let holdingsBarInstance = null;
let allPositions = [];
let allDecisions = [];
let authToken = localStorage.getItem('aegis_token') || '';
let ws = null;
let wsRetryCount = 0;
const MAX_WS_RETRIES = 10;

const fmt = (v) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v);
const fmtPct = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
const fmtIST = (ts) => {
    try { return new Date(ts).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' }); }
    catch { return ts || '—'; }
};

// ── Auth helpers ─────────────────────────────────────────────────────────────
function authHeaders() {
    return authToken ? { 'Authorization': `Bearer ${authToken}` } : {};
}

async function apiFetch(url) {
    const res = await fetch(url, { headers: authHeaders() });
    if (res.status === 401) {
        showLogin();
        throw new Error('Unauthorized');
    }
    return res.json();
}

async function checkAuth() {
    try {
        const status = await fetch('/api/auth/status').then(r => r.json());
        if (!status.auth_enabled) {
            authToken = 'dev-mode';
            hideLogin();
            return;
        }
        // Auth is enabled — check if we have a valid token
        if (authToken && authToken !== 'dev-mode') {
            try {
                await apiFetch('/api/portfolio');
                hideLogin();
            } catch {
                showLogin();
            }
        } else {
            showLogin();
        }
    } catch {
        hideLogin(); // Can't reach server, show dashboard anyway
    }
}

function showLogin() {
    document.getElementById('login-overlay').style.display = 'flex';
    document.getElementById('app-container').style.filter = 'blur(8px)';
}

function hideLogin() {
    document.getElementById('login-overlay').style.display = 'none';
    document.getElementById('app-container').style.filter = '';
}

async function doLogin() {
    const pw = document.getElementById('login-password').value;
    try {
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password: pw }),
        });
        const data = await res.json();
        if (res.ok && data.token) {
            authToken = data.token;
            localStorage.setItem('aegis_token', authToken);
            hideLogin();
            document.getElementById('login-error').style.display = 'none';
            await fetchAll();
            connectWebSocket();
        } else {
            document.getElementById('login-error').style.display = 'block';
        }
    } catch {
        document.getElementById('login-error').style.display = 'block';
    }
}

// ── Routing ──────────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        const page = link.dataset.page;
        document.querySelectorAll('.nav-item').forEach(l => l.classList.remove('active'));
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        link.classList.add('active');
        document.getElementById(`page-${page}`).classList.add('active');
        if (page === 'positions') renderPositionsPage();
        if (page === 'decisions') renderDecisionsPage();
        if (page === 'trades') renderTradesPage();
    });
});

// ── Bootstrap ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // Auth events
    document.getElementById('login-btn').addEventListener('click', doLogin);
    document.getElementById('login-password').addEventListener('keydown', e => {
        if (e.key === 'Enter') doLogin();
    });

    await checkAuth();
    await fetchAll();
    connectWebSocket();

    // HTTP polling fallback (30s) — WebSocket is primary
    setInterval(fetchAll, 30000);
});

// ── WebSocket ────────────────────────────────────────────────────────────────
function connectWebSocket() {
    if (ws && ws.readyState <= 1) return; // Already connected or connecting

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/live?token=${authToken}`;

    try {
        ws = new WebSocket(url);
    } catch {
        updateWsStatus(false);
        return;
    }

    ws.onopen = () => {
        wsRetryCount = 0;
        updateWsStatus(true);
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'snapshot') {
                updateLiveKPIs(data);
            }
        } catch { /* ignore malformed messages */ }
    };

    ws.onclose = () => {
        updateWsStatus(false);
        // Exponential backoff reconnect
        if (wsRetryCount < MAX_WS_RETRIES) {
            const delay = Math.min(2000 * Math.pow(1.5, wsRetryCount), 30000);
            wsRetryCount++;
            setTimeout(connectWebSocket, delay);
        }
    };

    ws.onerror = () => {
        updateWsStatus(false);
    };
}

function updateWsStatus(connected) {
    const el = document.getElementById('ws-status');
    if (connected) {
        el.textContent = 'WS Connected';
        el.className = 'ws-badge ws-connected';
    } else {
        el.textContent = 'WS Disconnected';
        el.className = 'ws-badge ws-disconnected';
    }
}

function updateLiveKPIs(data) {
    // Real-time updates from WebSocket
    document.getElementById('val-portfolio').textContent = fmt(data.portfolio_value);
    const pnl = data.total_pnl || 0;
    const pnlEl = document.getElementById('val-pnl');
    pnlEl.textContent = `${pnl >= 0 ? '+' : ''}${fmt(pnl)} total P&L`;
    pnlEl.className = 'subtext ' + (pnl >= 0 ? 'text-success' : 'text-danger');

    const totalPnlEl = document.getElementById('val-total-pnl');
    totalPnlEl.textContent = `${pnl >= 0 ? '+' : ''}${fmt(pnl)}`;
    totalPnlEl.className = 'value ' + (pnl >= 0 ? 'text-success' : 'text-danger');

    document.getElementById('val-drawdown').textContent = ((data.drawdown || 0) * 100).toFixed(2) + '%';
    document.getElementById('val-positions').textContent = data.open_positions || '0';
    document.getElementById('last-refresh').textContent = 'Live ' + new Date().toLocaleTimeString('en-IN');
}

// ── Data Fetching ────────────────────────────────────────────────────────────
async function fetchAll() {
    try {
        const [portfolioRes, latestRunRes, decisionsRes, benchmarkRes] = await Promise.all([
            apiFetch('/api/portfolio').catch(() => ({})),
            apiFetch('/api/latest-run').catch(() => ({})),
            apiFetch('/api/decisions').catch(() => []),
            apiFetch('/api/benchmark').catch(() => ({ benchmark: [] })),
        ]);

        allPositions = latestRunRes.positions || [];
        allDecisions = Array.isArray(decisionsRes) ? decisionsRes : [];

        updateKPIs(portfolioRes, latestRunRes);
        renderPnlChart(portfolioRes.history || [], benchmarkRes.benchmark || []);
        renderLatestRunTable(latestRunRes);

        document.getElementById('last-refresh').textContent = 'Updated ' + new Date().toLocaleTimeString('en-IN');
    } catch { /* auth redirect or network error */ }
}

// ── KPI Cards ────────────────────────────────────────────────────────────────
function updateKPIs(portfolio, run) {
    const val = portfolio.current_value || 250000;
    const pnl = portfolio.total_pnl || 0;
    const dd = (portfolio.drawdown || 0) * 100;
    const s = run.summary || {};

    document.getElementById('val-portfolio').textContent = fmt(val);
    const pnlEl = document.getElementById('val-pnl');
    pnlEl.textContent = `${pnl >= 0 ? '+' : ''}${fmt(pnl)} total P&L`;
    pnlEl.className = 'subtext ' + (pnl >= 0 ? 'text-success' : 'text-danger');

    const totalPnlEl = document.getElementById('val-total-pnl');
    totalPnlEl.textContent = `${pnl >= 0 ? '+' : ''}${fmt(pnl)}`;
    totalPnlEl.className = 'value ' + (pnl >= 0 ? 'text-success' : 'text-danger');

    document.getElementById('val-drawdown').textContent = dd.toFixed(2) + '%';

    const posCount = (s.long_count || 0) + (s.short_count || 0);
    document.getElementById('val-positions').textContent = posCount || '0';
    document.getElementById('val-exposure').textContent =
        posCount ? `${s.long_count} long · ${s.short_count} short` : 'No positions yet';

    document.getElementById('val-gross').textContent = s.gross_exposure_pct != null ? s.gross_exposure_pct + '%' : '—';
    document.getElementById('val-net-exp').textContent = s.net_exposure_pct != null ? `Net: ${s.net_exposure_pct}%` : '—';

    if (run.timestamp) {
        document.getElementById('val-last-run').textContent = fmtIST(run.timestamp);
        document.getElementById('val-model').textContent = run.model_version || '—';
    }
}

// ── P&L Chart with Nifty50 Benchmark ─────────────────────────────────────────
function renderPnlChart(history, benchmark) {
    const ctx = document.getElementById('pnlChart').getContext('2d');

    let labels, data;
    if (history.length <= 1) {
        const base = history.length === 1 ? history[0].total_portfolio_value : 250000;
        labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Today'];
        data = [base, base, base, base, base, base];
    } else {
        labels = history.map(r => r.date);
        data = history.map(r => r.total_portfolio_value);
    }

    // Map benchmark data to the same date labels
    const benchMap = {};
    (benchmark || []).forEach(b => { benchMap[b.date] = b.value; });
    const benchData = labels.map(d => benchMap[d] || null);

    const gradient = ctx.createLinearGradient(0, 0, 0, 320);
    gradient.addColorStop(0, 'rgba(0,208,156,0.35)');
    gradient.addColorStop(1, 'rgba(0,208,156,0.0)');

    const datasets = [
        {
            label: 'Portfolio Value',
            data,
            borderColor: '#00d09c',
            backgroundColor: gradient,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 6,
            pointBackgroundColor: '#00d09c',
            fill: true,
            tension: 0.4,
        },
        {
            label: 'Nifty 50',
            data: benchData,
            borderColor: '#5b8def',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [6, 3],
            pointRadius: 0,
            pointHoverRadius: 4,
            pointBackgroundColor: '#5b8def',
            fill: false,
            tension: 0.4,
            spanGaps: true,
        },
    ];

    const cfg = {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index', intersect: false,
                    backgroundColor: 'rgba(26,29,39,0.95)',
                    titleColor: '#fff', bodyColor: '#ccc',
                    borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, padding: 12,
                    callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.parsed.y)}` }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => '₹' + (v / 1000).toFixed(0) + 'k' }
                }
            },
            interaction: { mode: 'nearest', axis: 'x', intersect: false }
        }
    };

    if (pnlChartInstance) {
        pnlChartInstance.data.labels = labels;
        pnlChartInstance.data.datasets = datasets;
        pnlChartInstance.update();
    } else {
        Chart.defaults.color = '#8b92a5';
        Chart.defaults.font.family = 'Outfit';
        pnlChartInstance = new Chart(ctx, cfg);
    }
}

// ── Latest Run table (Dashboard) ─────────────────────────────────────────────
function renderLatestRunTable(data) {
    const tsEl = document.getElementById('run-timestamp');
    const summaryEl = document.getElementById('run-summary');
    const tbody = document.getElementById('latest-run-body');

    if (!data || !data.positions) return;
    if (data.timestamp) tsEl.textContent = '— ' + fmtIST(data.timestamp);

    const s = data.summary || {};
    summaryEl.innerHTML = [
        `<span class="pill">Universe <b>${data.universe_size || 0}</b></span>`,
        `<span class="pill pill-green">Long <b>${s.long_count || 0}</b></span>`,
        `<span class="pill pill-red">Short <b>${s.short_count || 0}</b></span>`,
        `<span class="pill">Gross <b>${s.gross_exposure_pct || 0}%</b></span>`,
        `<span class="pill">Net <b>${s.net_exposure_pct || 0}%</b></span>`,
        `<span class="pill ${data.circuit_breaker === 'OK' ? 'pill-green' : 'pill-red'}">CB <b>${data.circuit_breaker || '—'}</b></span>`,
    ].join('');

    tbody.innerHTML = '';
    if (!data.positions.length) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-state">No positions in last run.</td></tr>`;
        return;
    }
    data.positions.forEach(pos => {
        const isLong = pos.direction === 'LONG';
        const reasoning = pos.reasoning || {};
        const committee = reasoning.committee || {};
        const rationale = committee.rationale || '';
        const shortRationale = rationale.length > 80 ? rationale.substring(0, 80) + '…' : rationale;

        tbody.insertAdjacentHTML('beforeend', `
            <tr class="clickable-row" onclick="showReasoningModal('${pos.ticker}', ${JSON.stringify(reasoning).replace(/'/g, "&#39;").replace(/"/g, '&quot;')})">
                <td class="symbol-cell">${pos.ticker.replace('.NS', '')}</td>
                <td><span class="badge ${isLong ? 'badge-long' : 'badge-short'}">${pos.direction}</span></td>
                <td class="${isLong ? 'text-success' : 'text-danger'}">${fmtPct(pos.weight_pct)}</td>
                <td>${fmt(pos.rupees)}</td>
                <td class="dim reasoning-cell" title="${rationale}">${shortRationale || '—'}</td>
            </tr>`);
    });
}

// ── Reasoning Modal ──────────────────────────────────────────────────────────
window.showReasoningModal = function(ticker, reasoning) {
    if (!reasoning || !Object.keys(reasoning).length) return;

    const signals = reasoning.research_signals || [];
    const committee = reasoning.committee || {};
    const allocation = reasoning.allocation || {};
    const risk = reasoning.risk || {};

    let signalRows = signals.map(s =>
        `<tr><td>${s.agent || '—'}</td><td><span class="badge ${s.action?.includes('LONG') ? 'badge-long' : s.action?.includes('SHORT') ? 'badge-short' : 'badge-hold'}">${s.action || '—'}</span></td><td class="dim">${s.rationale || '—'}</td></tr>`
    ).join('');

    const html = `
        <div class="reasoning-modal-overlay" onclick="this.remove()">
            <div class="reasoning-modal" onclick="event.stopPropagation()">
                <div class="table-header-row">
                    <h2>Trade Reasoning: ${ticker.replace('.NS', '')}</h2>
                    <button class="close-btn" onclick="this.closest('.reasoning-modal-overlay').remove()">&#10005;</button>
                </div>
                <div class="reasoning-section">
                    <h3>Research Signals</h3>
                    <table class="modern-table compact-table">
                        <thead><tr><th>Agent</th><th>Action</th><th>Rationale</th></tr></thead>
                        <tbody>${signalRows || '<tr><td colspan="3" class="empty-state">No signals</td></tr>'}</tbody>
                    </table>
                </div>
                <div class="reasoning-section">
                    <h3>Committee Decision</h3>
                    <div class="detail-meta">
                        <span>Action: <b class="${committee.action === 'PROPOSE' ? 'text-success' : 'text-danger'}">${committee.action || '—'}</b></span>
                        <span>Direction: <b>${committee.direction || '—'}</b></span>
                    </div>
                    <p class="dim">${committee.rationale || 'No rationale provided'}</p>
                </div>
                <div class="reasoning-section">
                    <h3>Portfolio Allocation</h3>
                    <div class="detail-meta">
                        <span>Exposure: <b>${allocation.exposure_pct != null ? (allocation.exposure_pct * 100).toFixed(1) + '%' : '—'}</b></span>
                    </div>
                    <p class="dim">${allocation.rationale || 'No rationale provided'}</p>
                </div>
                <div class="reasoning-section">
                    <h3>Risk Officer</h3>
                    <div class="detail-meta">
                        <span>Verdict: <b class="${risk.action === 'APPROVE' ? 'text-success' : 'text-danger'}">${risk.action || '—'}</b></span>
                    </div>
                    <p class="dim">${risk.rationale || 'No rationale provided'}</p>
                </div>
            </div>
        </div>`;

    document.body.insertAdjacentHTML('beforeend', html);
};

// ── POSITIONS PAGE ───────────────────────────────────────────────────────────
function renderPositionsPage() {
    renderExposureDonut();
    renderHoldingsBar();
    renderFullPositionsTable('all');

    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderFullPositionsTable(btn.dataset.filter);
        });
    });
}

function renderExposureDonut() {
    const longs = allPositions.filter(p => p.direction === 'LONG').reduce((s, p) => s + p.weight_pct, 0);
    const shorts = allPositions.filter(p => p.direction === 'SHORT').reduce((s, p) => s + Math.abs(p.weight_pct), 0);
    const cash = Math.max(0, 100 - longs - shorts);

    const ctx = document.getElementById('exposureChart').getContext('2d');
    const cfg = {
        type: 'doughnut',
        data: {
            labels: ['Long', 'Short', 'Cash'],
            datasets: [{ data: [longs, shorts, cash], backgroundColor: ['#00d09c', '#ff5252', '#2a2d3a'], borderWidth: 0, hoverOffset: 6 }]
        },
        options: {
            responsive: true, maintainAspectRatio: true, cutout: '70%',
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `${c.label}: ${c.parsed.toFixed(1)}%` } } }
        }
    };

    document.getElementById('exposure-legend').innerHTML = `
        <span class="leg-item"><span class="leg-dot" style="background:#00d09c"></span>Long ${longs.toFixed(1)}%</span>
        <span class="leg-item"><span class="leg-dot" style="background:#ff5252"></span>Short ${shorts.toFixed(1)}%</span>
        <span class="leg-item"><span class="leg-dot" style="background:#2a2d3a"></span>Cash ${cash.toFixed(1)}%</span>
    `;

    if (exposureChartInstance) exposureChartInstance.destroy();
    exposureChartInstance = new Chart(ctx, cfg);
}

function renderHoldingsBar() {
    const positions = [...allPositions].sort((a, b) => Math.abs(b.weight_pct) - Math.abs(a.weight_pct)).slice(0, 12);
    const labels = positions.map(p => p.ticker.replace('.NS', ''));
    const data = positions.map(p => p.weight_pct);
    const colors = data.map(v => v >= 0 ? '#00d09c' : '#ff5252');

    const ctx = document.getElementById('holdingsBar').getContext('2d');
    const cfg = {
        type: 'bar',
        data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 4 }] },
        options: {
            indexAxis: 'y', responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => fmtPct(c.parsed.x) } } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { callback: v => v + '%' } },
                y: { grid: { display: false } }
            }
        }
    };

    if (holdingsBarInstance) holdingsBarInstance.destroy();
    holdingsBarInstance = new Chart(ctx, cfg);
}

function renderFullPositionsTable(filter) {
    const tbody = document.getElementById('positions-full-body');
    const rows = filter === 'all' ? allPositions : allPositions.filter(p => p.direction === filter);

    tbody.innerHTML = '';
    if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="7" class="empty-state">No ${filter === 'all' ? '' : filter.toLowerCase() + ' '}positions.</td></tr>`;
        return;
    }
    rows.forEach(pos => {
        const isLong = pos.direction === 'LONG';
        const strategy = pos.reasoning?.trade_type || '';
        tbody.insertAdjacentHTML('beforeend', `
            <tr class="clickable-row" onclick="showReasoningModal('${pos.ticker}', ${JSON.stringify(pos.reasoning || {}).replace(/'/g, "&#39;").replace(/"/g, '&quot;')})">
                <td class="symbol-cell">${pos.ticker.replace('.NS', '')}</td>
                <td><span class="badge ${isLong ? 'badge-long' : 'badge-short'}">${pos.direction}</span></td>
                <td class="${isLong ? 'text-success' : 'text-danger'}">${fmtPct(pos.weight_pct)}</td>
                <td>${fmt(pos.rupees)}</td>
                <td class="dim">${pos.last_price ? '₹' + pos.last_price.toLocaleString('en-IN') : '—'}</td>
                <td class="dim">${pos.est_shares ?? '—'}</td>
                <td class="dim">${strategy}</td>
            </tr>`);
    });
}

// ── TRADES PAGE ──────────────────────────────────────────────────────────────
async function renderTradesPage() {
    const tbody = document.getElementById('trades-body');
    try {
        const trades = await apiFetch('/api/trades/closed');
        tbody.innerHTML = '';

        if (!trades.length) {
            tbody.innerHTML = `<tr><td colspan="10" class="empty-state">No closed trades yet.</td></tr>`;
            return;
        }

        trades.forEach(t => {
            const pnlClass = t.pnl_pct >= 0 ? 'text-success' : 'text-danger';
            const pnlSign = t.pnl_pct >= 0 ? '+' : '';
            const exitBadge = {
                'SL': 'badge-short', 'TP': 'badge-long', 'AGING': 'badge-hold', 'EXIT_SIGNAL': 'badge-hold'
            }[t.exit_reason] || 'badge-hold';

            tbody.insertAdjacentHTML('beforeend', `
                <tr>
                    <td class="symbol-cell">${t.ticker.replace('.NS', '')}</td>
                    <td class="dim">${t.strategy || '—'}</td>
                    <td>₹${t.entry_price.toLocaleString('en-IN')}</td>
                    <td>₹${t.exit_price.toLocaleString('en-IN')}</td>
                    <td>${t.quantity}</td>
                    <td class="${pnlClass}">${pnlSign}${(t.pnl_pct * 100).toFixed(2)}%</td>
                    <td class="${pnlClass}">${pnlSign}${fmt(t.realized_pnl)}</td>
                    <td><span class="badge ${exitBadge}">${t.exit_reason || '—'}</span></td>
                    <td class="dim">${t.entry_date || '—'}</td>
                    <td class="dim">${t.exit_date || '—'}</td>
                </tr>`);
        });
    } catch {
        tbody.innerHTML = `<tr><td colspan="10" class="empty-state">Failed to load trades.</td></tr>`;
    }
}

// ── DECISIONS PAGE ───────────────────────────────────────────────────────────
function renderDecisionsPage() {
    const tbody = document.getElementById('decisions-full-body');
    tbody.innerHTML = '';

    if (!allDecisions.length) {
        tbody.innerHTML = `<tr><td colspan="8" class="empty-state">No decisions logged yet.</td></tr>`;
        return;
    }

    allDecisions.forEach((dec) => {
        const weights = Array.isArray(dec.final_weights) ? dec.final_weights : [];
        const tickers = Array.isArray(dec.ticker_universe) ? dec.ticker_universe : [];
        const longs = weights.filter(w => w > 0.001).length;
        const shorts = weights.filter(w => w < -0.001).length;
        const gross = weights.reduce((s, w) => s + Math.abs(w), 0);
        const cbOk = dec.circuit_breaker_status === 'OK';

        const tr = document.createElement('tr');
        tr.classList.add('clickable-row');
        tr.innerHTML = `
            <td>${fmtIST(dec.timestamp)}</td>
            <td class="text-success" style="font-size:0.82rem">${dec.model_version}</td>
            <td>${tickers.length || weights.length}</td>
            <td class="text-success">${longs}</td>
            <td class="text-danger">${shorts}</td>
            <td>${(gross * 100).toFixed(1)}%</td>
            <td><span class="badge ${cbOk ? 'badge-long' : 'badge-short'}">${dec.circuit_breaker_status}</span></td>
            <td>${(dec.transaction_costs || 0).toFixed(2)}</td>
        `;
        tr.addEventListener('click', () => openDecisionDetail(dec));
        tbody.appendChild(tr);
    });

    document.getElementById('close-detail').addEventListener('click', () => {
        document.getElementById('decision-detail-card').style.display = 'none';
    });
}

function openDecisionDetail(dec) {
    const card = document.getElementById('decision-detail-card');
    const body = document.getElementById('detail-body');
    document.getElementById('detail-title').textContent = `Decision — ${fmtIST(dec.timestamp)}`;

    const weights = Array.isArray(dec.final_weights) ? dec.final_weights : [];
    const tickers = Array.isArray(dec.ticker_universe) ? dec.ticker_universe : [];
    const reasoning = dec.trade_reasoning || {};

    let rows = '';
    tickers.forEach((ticker, i) => {
        const w = weights[i] || 0;
        if (Math.abs(w) < 0.001) return;
        const isLong = w > 0;
        const r = reasoning[ticker] || {};
        const committee = r.committee || {};
        const rationale = committee.rationale || '';
        const shortRationale = rationale.length > 60 ? rationale.substring(0, 60) + '…' : rationale;

        rows += `<tr class="clickable-row" onclick="showReasoningModal('${ticker}', ${JSON.stringify(r).replace(/'/g, "&#39;").replace(/"/g, '&quot;')})">
            <td class="symbol-cell">${ticker.replace('.NS', '')}</td>
            <td><span class="badge ${isLong ? 'badge-long' : 'badge-short'}">${isLong ? 'LONG' : 'SHORT'}</span></td>
            <td class="${isLong ? 'text-success' : 'text-danger'}">${fmtPct(w * 100)}</td>
            <td class="dim">${shortRationale || '—'}</td>
        </tr>`;
    });

    body.innerHTML = `
        <div class="detail-meta">
            <span>Model: <b>${dec.model_version}</b></span>
            <span>Circuit Breaker: <b class="${dec.circuit_breaker_status === 'OK' ? 'text-success' : 'text-danger'}">${dec.circuit_breaker_status}</b></span>
            <span>Slippage: <b>${(dec.transaction_costs || 0).toFixed(2)} bps</b></span>
        </div>
        <table class="modern-table">
            <thead><tr><th>Symbol</th><th>Direction</th><th>Weight</th><th>Reasoning</th></tr></thead>
            <tbody>${rows || '<tr><td colspan="4" class="empty-state">All weights were zero.</td></tr>'}</tbody>
        </table>
    `;

    card.style.display = 'block';
    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
