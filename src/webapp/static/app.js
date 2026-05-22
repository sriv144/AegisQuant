// AegisQuant Terminal — Dashboard App
// 6 functional screens: Overview, AI Trades, Positions, Journal, Circuits, Setup
// WebSocket real-time + HTTP polling fallback, JWT auth

let pnlChartInstance = null;
let allPositions = [];
let allDecisions = [];
let authToken = localStorage.getItem('aegis_token') || '';
let ws = null;
let wsRetryCount = 0;
const MAX_WS_RETRIES = 10;

let MARKET_CFG = {
    market: 'US',
    currency_symbol: '$',
    currency_code: 'USD',
    locale: 'en-US',
    timezone: 'America/New_York',
    benchmark_label: 'S&P 500',
    default_capital: 100000,
};

const fmt = (v) => {
    try {
        return new Intl.NumberFormat(MARKET_CFG.locale, {
            style: 'currency', currency: MARKET_CFG.currency_code, maximumFractionDigits: 0
        }).format(v);
    } catch { return '$' + Math.round(v).toLocaleString(); }
};
const fmtPct = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
const fmtTime = (ts) => {
    try {
        return new Date(ts).toLocaleString(MARKET_CFG.locale, {
            timeZone: MARKET_CFG.timezone, day: '2-digit', month: 'short',
            year: 'numeric', hour: '2-digit', minute: '2-digit'
        });
    } catch { return ts || '--'; }
};
const shortTime = (ts) => {
    try {
        return new Date(ts).toLocaleString(MARKET_CFG.locale, {
            timeZone: MARKET_CFG.timezone, hour: '2-digit', minute: '2-digit'
        });
    } catch { return '--'; }
};

// ---- Auth ----
function authHeaders() {
    return authToken ? { 'Authorization': `Bearer ${authToken}` } : {};
}

async function apiFetch(url) {
    const res = await fetch(url, { headers: authHeaders() });
    if (res.status === 401) { showLogin(); throw new Error('Unauthorized'); }
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
        if (authToken && authToken !== 'dev-mode') {
            try { await apiFetch('/api/portfolio'); hideLogin(); }
            catch { showLogin(); }
        } else { showLogin(); }
    } catch { hideLogin(); }
}

function showLogin() {
    document.getElementById('login-overlay').style.display = 'flex';
    document.getElementById('app').style.filter = 'blur(8px)';
}
function hideLogin() {
    document.getElementById('login-overlay').style.display = 'none';
    document.getElementById('app').style.filter = '';
}

async function doLogin() {
    const pw = document.getElementById('login-password').value;
    try {
        const res = await fetch('/api/auth/login', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
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
    } catch { document.getElementById('login-error').style.display = 'block'; }
}

// ---- Routing ----
function navigateTo(page) {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const btn = document.querySelector(`.nav-btn[data-page="${page}"]`);
    if (btn) btn.classList.add('active');
    const el = document.getElementById(`page-${page}`);
    if (el) el.classList.add('active');

    // Lazy-load page data
    if (page === 'positions') renderPositionsPage();
    if (page === 'trades') renderTradesPage();
    if (page === 'journal') renderJournalPage();
    if (page === 'circuits') renderCircuitsPage();
    if (page === 'setup') renderSetupPage();
}

document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => navigateTo(btn.dataset.page));
});

// Keyboard shortcuts
window.addEventListener('keydown', (e) => {
    if (e.target && /input|textarea/i.test(e.target.tagName)) return;
    const pages = ['overview', 'trades', 'positions', 'journal', 'circuits', 'setup'];
    const idx = parseInt(e.key) - 1;
    if (idx >= 0 && idx < pages.length) navigateTo(pages[idx]);
});

// ---- Clock & Market State ----
function updateClock() {
    const now = new Date();
    const etTime = now.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour12: false });
    document.getElementById('market-time').textContent = etTime + ' ET';
    document.getElementById('status-time').textContent = etTime + ' ET';

    // Market hours: 9:30 AM - 4:00 PM ET, weekdays
    const etHour = parseInt(now.toLocaleString('en-US', { timeZone: 'America/New_York', hour: 'numeric', hour12: false }));
    const etMin = parseInt(now.toLocaleString('en-US', { timeZone: 'America/New_York', minute: 'numeric' }));
    const day = now.toLocaleString('en-US', { timeZone: 'America/New_York', weekday: 'short' });
    const isWeekday = !['Sat', 'Sun'].includes(day);
    const afterOpen = etHour > 9 || (etHour === 9 && etMin >= 30);
    const beforeClose = etHour < 16;
    const isOpen = isWeekday && afterOpen && beforeClose;

    const dot = document.getElementById('market-dot');
    const statusEl = document.getElementById('market-status');
    dot.className = 'dot ' + (isOpen ? 'live' : 'off');
    statusEl.textContent = isOpen ? 'OPEN' : 'CLOSED';
    statusEl.className = 'mono ' + (isOpen ? 'up' : 'dim');
}
setInterval(updateClock, 1000);
updateClock();

// ---- WebSocket ----
function connectWebSocket() {
    if (ws && ws.readyState <= 1) return;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/live?token=${authToken}`;
    try { ws = new WebSocket(url); } catch { updateWsStatus(false); return; }

    ws.onopen = () => { wsRetryCount = 0; updateWsStatus(true); };
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'snapshot') updateLiveKPIs(data);
        } catch {}
    };
    ws.onclose = () => {
        updateWsStatus(false);
        if (wsRetryCount < MAX_WS_RETRIES) {
            const delay = Math.min(2000 * Math.pow(1.5, wsRetryCount), 30000);
            wsRetryCount++;
            setTimeout(connectWebSocket, delay);
        }
    };
    ws.onerror = () => updateWsStatus(false);
}

function updateWsStatus(connected) {
    const badge = document.getElementById('ws-badge');
    const dot = document.getElementById('ws-dot');
    const text = document.getElementById('ws-text');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    if (connected) {
        badge.className = 'pill-action ws-on';
        dot.className = 'dot live';
        text.textContent = 'WS Live';
        statusDot.className = 'dot live';
        statusText.textContent = 'ALL SYSTEMS NOMINAL';
        statusText.className = 'ok';
    } else {
        badge.className = 'pill-action ws-off';
        dot.className = 'dot off';
        text.textContent = 'WS Off';
        statusDot.className = 'dot off';
        statusText.textContent = 'WS DISCONNECTED';
        statusText.className = 'down';
    }
}

function updateLiveKPIs(data) {
    const pv = data.portfolio_value || MARKET_CFG.default_capital;
    const pnl = data.total_pnl || 0;
    const dd = (data.drawdown || 0) * 100;

    document.getElementById('ov-equity').textContent = fmt(pv);
    document.getElementById('topbar-equity').textContent = fmt(pv);
    const pnlEl = document.getElementById('ov-pnl');
    pnlEl.textContent = `${pnl >= 0 ? '+' : ''}${fmt(pnl)} total P&L`;
    pnlEl.className = 'd ' + (pnl >= 0 ? 'up' : 'down');

    document.getElementById('ov-total-pnl').textContent = `${pnl >= 0 ? '+' : ''}${fmt(pnl)}`;
    document.getElementById('ov-total-pnl').className = 'v mono ' + (pnl >= 0 ? 'up' : 'down');
    document.getElementById('ov-drawdown').textContent = dd.toFixed(2) + '%';
    document.getElementById('ov-positions').textContent = data.open_positions || '0';
    document.getElementById('status-dd').textContent = dd.toFixed(1) + '%';
    document.getElementById('status-cb').textContent = data.circuit_breaker || 'OK';
}

// ---- Data Fetching ----
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

        updateOverviewKPIs(portfolioRes, latestRunRes);
        renderPnlChart(portfolioRes.history || [], benchmarkRes.benchmark || []);
        renderOverviewPositions(latestRunRes);
        renderAgentStrip(latestRunRes);
        renderCircuitsMini();
    } catch {}
}

function updateOverviewKPIs(portfolio, run) {
    const val = portfolio.current_value || MARKET_CFG.default_capital;
    const pnl = portfolio.total_pnl || 0;
    const dailyPnl = portfolio.daily_pnl != null ? portfolio.daily_pnl : null;
    const dd = (portfolio.drawdown || 0) * 100;
    const s = run.summary || {};

    document.getElementById('ov-equity').textContent = fmt(val);
    document.getElementById('topbar-equity').textContent = fmt(val);

    // Show today's P&L if available, else cumulative
    const displayPnl = dailyPnl !== null ? dailyPnl : pnl;
    const pnlLabel = dailyPnl !== null ? 'today' : 'total P&L';
    const pnlEl = document.getElementById('ov-pnl');
    pnlEl.textContent = `${displayPnl >= 0 ? '+' : ''}${fmt(displayPnl)} ${pnlLabel}`;
    pnlEl.className = 'd ' + (displayPnl >= 0 ? 'up' : 'down');

    const totalPnlEl = document.getElementById('ov-total-pnl');
    totalPnlEl.textContent = `${pnl >= 0 ? '+' : ''}${fmt(pnl)}`;
    totalPnlEl.className = 'v mono ' + (pnl >= 0 ? 'up' : 'down');

    document.getElementById('ov-drawdown').textContent = dd.toFixed(2) + '%';
    document.getElementById('status-dd').textContent = dd.toFixed(1) + '%';

    const posCount = (s.long_count || 0) + (s.short_count || 0);
    document.getElementById('ov-positions').textContent = posCount || '0';
    document.getElementById('ov-pos-detail').textContent =
        posCount ? `${s.long_count || 0} long | Gross: ${s.gross_exposure_pct || 0}%` : 'No positions yet';

    if (run.timestamp) {
        document.getElementById('run-ts').textContent = fmtTime(run.timestamp);
    }
}

// ---- Equity Chart ----
function renderPnlChart(history, benchmark) {
    const ctx = document.getElementById('pnlChart').getContext('2d');

    let labels, data;
    if (history.length <= 1) {
        labels = ['Day 1', 'Day 2', 'Day 3'];
        data = [MARKET_CFG.default_capital, MARKET_CFG.default_capital, MARKET_CFG.default_capital];
    } else {
        labels = history.map(h => h.date);
        data = history.map(h => h.total_portfolio_value);
    }

    const benchMap = {};
    (benchmark || []).forEach(b => { benchMap[b.date] = b.value; });
    const benchData = labels.map(l => benchMap[l] || null);

    if (pnlChartInstance) pnlChartInstance.destroy();
    pnlChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'AegisQuant',
                    data, fill: false,
                    borderColor: '#5ee0a3', borderWidth: 2,
                    pointRadius: 0, tension: 0.3,
                },
                {
                    label: MARKET_CFG.benchmark_label,
                    data: benchData, fill: false,
                    borderColor: '#6b7080', borderWidth: 1.5,
                    borderDash: [6, 3], pointRadius: 0, tension: 0.3,
                },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: '#6b7080', font: { size: 10 }, maxTicksLimit: 12 },
                    grid: { color: 'rgba(240,241,245,0.04)' },
                },
                y: {
                    ticks: {
                        color: '#6b7080', font: { size: 10, family: 'JetBrains Mono' },
                        callback: v => '$' + (v / 1000).toFixed(0) + 'k',
                    },
                    grid: { color: 'rgba(240,241,245,0.04)' },
                },
            },
            interaction: { intersect: false, mode: 'index' },
        },
    });

    if (history.length > 1) {
        document.getElementById('chart-subtitle').textContent =
            `paper portfolio | ${history.length} sessions`;
    }
}

// ---- Agent Committee Strip ----
function renderAgentStrip(run) {
    const container = document.getElementById('agent-strip');
    if (!run.positions || run.positions.length === 0) {
        container.innerHTML = '<div class="muted" style="padding:16px;font-size:12px">No agent data from latest run</div>';
        return;
    }

    // Aggregate reasoning across positions
    const agents = {};
    run.positions.forEach(p => {
        const r = p.reasoning || {};
        const signals = r.research_signals || [];
        signals.forEach(s => {
            const name = s.agent_name || s.agent || 'Unknown';
            if (!agents[name]) agents[name] = { total: 0, count: 0, actions: [] };
            agents[name].total += (s.confidence || 0);
            agents[name].count++;
            agents[name].actions.push(s.action || 'HOLD');
        });
    });

    // Fallback to known agents if no signals
    const agentNames = Object.keys(agents);
    if (agentNames.length === 0) {
        const defaultAgents = ['Quant', 'Fundamental', 'Macro', 'Sentiment'];
        container.innerHTML = defaultAgents.map(name => `
            <div class="agent-cell">
                <div class="name">${name}</div>
                <div class="conf">--<small>conf</small></div>
                <div class="sub"><span class="dot live"></span> Ready</div>
            </div>
        `).join('');
        document.getElementById('committee-summary').textContent =
            `${run.positions.length} positions analyzed`;
        return;
    }

    container.innerHTML = agentNames.map(name => {
        const a = agents[name];
        const avgConf = a.count > 0 ? (a.total / a.count) : 0;
        const longCount = a.actions.filter(x => x === 'PROPOSE_LONG').length;
        return `
            <div class="agent-cell">
                <div class="name">${name}</div>
                <div class="conf">${avgConf.toFixed(2)}<small>conf</small></div>
                <div class="sub">
                    <span class="dot live"></span>
                    ${longCount} LONG / ${a.count} signals
                </div>
            </div>
        `;
    }).join('');

    document.getElementById('committee-summary').textContent =
        `${agentNames.length} agents | ${run.positions.length} positions scored`;
}

// ---- Overview Positions Table ----
function renderOverviewPositions(run) {
    const body = document.getElementById('ov-positions-body');
    const positions = run.positions || [];

    if (positions.length === 0) {
        body.innerHTML = '<tr><td colspan="6" class="empty">No positions in latest run</td></tr>';
        return;
    }

    body.innerHTML = positions.slice(0, 15).map(p => {
        const r = p.reasoning || {};
        const strategy = r.committee?.strategy || r.strategy || r.strategy_used || '--';
        const reasoning = r.committee?.reasoning || r.reasoning || r.analyst_reasoning || '--';
        const reasonShort = typeof reasoning === 'string' ? reasoning.slice(0, 80) : '--';

        return `<tr>
            <td class="bright" style="font-weight:500">${p.ticker}</td>
            <td><span class="chip ${p.direction === 'LONG' ? 'up' : 'down'}">${p.direction}</span></td>
            <td class="r mono">${p.weight_pct.toFixed(1)}%</td>
            <td class="r mono">${fmt(p.capital)}</td>
            <td class="mono dim">${strategy}</td>
            <td class="muted" style="font-size:11.5px;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${reasonShort}</td>
        </tr>`;
    }).join('');
}

// ---- Circuit Breakers Mini ----
function renderCircuitsMini() {
    const container = document.getElementById('circuit-mini');
    const circuits = [
        { name: 'Long-Only', trip: 'w < 0', desc: 'Zeroes any negative weights' },
        { name: 'Drawdown CB', trip: '-15%', desc: 'Portfolio drawdown hard cutoff' },
        { name: 'Max Position', trip: '10%', desc: 'Per-ticker cap' },
        { name: 'Time Window', trip: '09:35-15:55', desc: 'NYSE trading hours only' },
        { name: 'Position SL', trip: '-8%', desc: 'Per-position stop loss' },
    ];

    container.innerHTML = circuits.map((c, i) => `
        <div class="cb-cell" style="${i < circuits.length - 1 ? 'border-right:1px solid var(--hairline)' : ''}">
            <div style="display:flex;align-items:center;gap:6px">
                <span class="dot live"></span>
                <span class="kicker" style="color:var(--ink-300)">${c.name}</span>
            </div>
            <div class="mono bright" style="font-size:15px;margin-top:6px">Trip @ ${c.trip}</div>
        </div>
    `).join('');

    document.getElementById('status-cb').textContent = `${circuits.length}/${circuits.length} armed`;
}

// ---- AI TRADES PAGE ----
let selectedDecisionId = null;

function renderTradesPage() {
    const list = document.getElementById('trades-list');
    const countEl = document.getElementById('trades-count');

    if (allDecisions.length === 0) {
        list.innerHTML = '<div class="muted" style="padding:20px;font-size:12px">No decisions yet</div>';
        countEl.textContent = '0 decisions';
        return;
    }

    countEl.textContent = `${allDecisions.length} decisions | live feed`;

    list.innerHTML = allDecisions.slice(0, 50).map((d, i) => {
        const weights = d.final_weights || [];
        const tickers = d.ticker_universe || [];
        const longCount = weights.filter(w => w > 0.001).length;
        const gross = weights.reduce((s, w) => s + Math.abs(w), 0);
        const ts = shortTime(d.timestamp);
        const reasoning = d.trade_reasoning || {};
        const firstTicker = tickers.find((t, j) => Math.abs(weights[j]) > 0.001) || '--';

        return `<div class="row ${i === 0 ? 'active' : ''}" data-idx="${i}" onclick="selectDecision(${i})">
            <div style="display:flex;align-items:center;justify-content:space-between">
                <div style="display:flex;align-items:center;gap:8px">
                    <span class="ticker">${firstTicker}${longCount > 1 ? ` +${longCount - 1}` : ''}</span>
                    <span class="chip up">LONG x${longCount}</span>
                </div>
                <span class="meta">${ts}</span>
            </div>
            <div style="display:flex;align-items:center;justify-content:space-between">
                <span class="meta">Gross: ${(gross * 100).toFixed(1)}% | ${d.model_version || '--'}</span>
                <span class="chip">${d.circuit_breaker_status || 'OK'}</span>
            </div>
        </div>`;
    }).join('');

    // Auto-select first
    if (allDecisions.length > 0) selectDecision(0);

    // Wire up filter buttons
    document.querySelectorAll('#page-trades .toggle-group button').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#page-trades .toggle-group button').forEach(b => b.classList.remove('on'));
            btn.classList.add('on');
            // Filter is cosmetic for now — all decisions are shown
        });
    });
}

function selectDecision(idx) {
    const d = allDecisions[idx];
    if (!d) return;

    // Update active state in list
    document.querySelectorAll('.ait .list-rows .row').forEach(r => r.classList.remove('active'));
    const activeRow = document.querySelector(`.ait .list-rows .row[data-idx="${idx}"]`);
    if (activeRow) activeRow.classList.add('active');

    const detail = document.getElementById('trade-detail');
    const weights = d.final_weights || [];
    const tickers = d.ticker_universe || [];
    const reasoning = d.trade_reasoning || {};

    const positions = [];
    for (let i = 0; i < tickers.length; i++) {
        if (Math.abs(weights[i]) >= 0.001) {
            positions.push({
                ticker: tickers[i],
                weight: weights[i],
                reasoning: reasoning[tickers[i]] || {},
            });
        }
    }
    positions.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

    const longCount = positions.filter(p => p.weight > 0).length;
    const shortCount = positions.filter(p => p.weight < 0).length;
    const gross = weights.reduce((s, w) => s + Math.abs(w), 0);

    detail.innerHTML = `
        <div class="detail-h">
            <div>
                <div style="display:flex;align-items:center;gap:10px">
                    <span style="font-size:12px;color:var(--ink-400);letter-spacing:0.08em">Decision #${d.id || idx + 1}</span>
                    <span style="width:1px;height:14px;background:var(--hairline)"></span>
                    <span class="bright" style="font-size:16px;font-weight:500">${fmtTime(d.timestamp)}</span>
                    <span class="chip">${d.model_version || '--'}</span>
                    <span class="chip ${d.circuit_breaker_status === 'OK' ? 'up' : 'down'}">${d.circuit_breaker_status || '--'}</span>
                </div>
                <div style="font-size:11.5px;color:var(--ink-400);margin-top:6px">
                    ${longCount} long | ${shortCount} short | Gross: ${(gross * 100).toFixed(1)}% | Universe: ${tickers.length} tickers
                </div>
            </div>
        </div>

        <div class="panel" style="margin-top:12px">
            <div class="panel-h">
                <div class="title">Position allocations</div>
                <div class="kicker">${positions.length} active</div>
            </div>
            <div class="panel-b flush">
                <table class="table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Direction</th>
                            <th class="r">Weight %</th>
                            <th>Strategy</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${positions.map(p => {
                            const r = p.reasoning;
                            const strategy = r.committee?.strategy || r.strategy || '--';
                            const dir = p.weight > 0 ? 'LONG' : 'SHORT';
                            return `<tr>
                                <td class="bright" style="font-weight:500">${p.ticker}</td>
                                <td><span class="chip ${dir === 'LONG' ? 'up' : 'down'}">${dir}</span></td>
                                <td class="r mono">${(p.weight * 100).toFixed(2)}%</td>
                                <td class="mono dim">${strategy}</td>
                            </tr>`;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        </div>

        ${positions.slice(0, 5).map(p => {
            const r = p.reasoning;
            const committee = r.committee || {};
            const signals = r.research_signals || [];
            const reasonText = committee.reasoning
                || r.reasoning
                || r.analyst_reasoning
                || 'No reasoning available';

            return `
            <div class="reason-panel">
                <div class="label">Bot reasoning: ${p.ticker}</div>
                <div class="text">${typeof reasonText === 'string' ? reasonText : JSON.stringify(reasonText)}</div>
                ${signals.length > 0 ? `
                    <div style="margin-top:10px">
                        <div class="kicker" style="margin-bottom:6px">Agent signals</div>
                        ${signals.map(s => `
                            <div class="gauge-row">
                                <div class="name">${s.agent_name || s.agent || '?'}</div>
                                <div class="gauge${(s.confidence || 0) < 0.3 ? ' neg' : ''}">
                                    <div class="fill" style="width:${Math.min(100, (s.confidence || 0) * 100)}%"></div>
                                </div>
                                <div class="v">${(s.confidence || 0).toFixed(2)}</div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>`;
        }).join('')}
    `;
}
window.selectDecision = selectDecision;

// ---- POSITIONS PAGE ----
function renderPositionsPage() {
    const summary = document.getElementById('pos-summary');
    const body = document.getElementById('positions-body');
    const subtitle = document.getElementById('pos-subtitle');

    if (allPositions.length === 0) {
        summary.innerHTML = '';
        body.innerHTML = '<tr><td colspan="8" class="empty">No positions in latest run</td></tr>';
        subtitle.textContent = 'no positions';
        return;
    }

    const totalCapital = allPositions.reduce((s, p) => s + (p.capital || 0), 0);
    const longCount = allPositions.filter(p => p.direction === 'LONG').length;
    const shortCount = allPositions.filter(p => p.direction === 'SHORT').length;
    const grossWeight = allPositions.reduce((s, p) => s + Math.abs(p.weight_pct || 0), 0);

    summary.innerHTML = `
        <div class="cell">
            <div class="kicker">Total allocated</div>
            <div class="mono bright" style="font-size:20px;margin-top:4px">${fmt(totalCapital)}</div>
            <div class="muted" style="font-size:11px;margin-top:2px">${allPositions.length} positions</div>
        </div>
        <div class="cell">
            <div class="kicker">Long positions</div>
            <div class="mono up" style="font-size:20px;margin-top:4px">${longCount}</div>
            <div class="muted" style="font-size:11px;margin-top:2px">long-only strategy</div>
        </div>
        <div class="cell">
            <div class="kicker">Gross exposure</div>
            <div class="mono bright" style="font-size:20px;margin-top:4px">${grossWeight.toFixed(1)}%</div>
            <div class="muted" style="font-size:11px;margin-top:2px">max 10% per ticker</div>
        </div>
        <div class="cell">
            <div class="kicker">Cash %</div>
            <div class="mono bright" style="font-size:20px;margin-top:4px">${Math.max(0, 100 - grossWeight).toFixed(1)}%</div>
            <div class="muted" style="font-size:11px;margin-top:2px">unallocated</div>
        </div>
    `;

    subtitle.textContent = `${allPositions.length} positions | paper mode`;

    body.innerHTML = allPositions.map(p => {
        const r = p.reasoning || {};
        const strategy = r.committee?.strategy || r.strategy || r.strategy_used || '--';
        const thesis = r.committee?.reasoning || r.reasoning || r.analyst_reasoning || '--';
        const thesisShort = typeof thesis === 'string' ? thesis.slice(0, 60) : '--';

        return `<tr>
            <td class="bright" style="font-weight:500">${p.ticker}</td>
            <td><span class="chip ${p.direction === 'LONG' ? 'up' : 'down'}">${p.direction}</span></td>
            <td class="r mono">${p.weight_pct.toFixed(1)}%</td>
            <td class="r mono">${fmt(p.capital)}</td>
            <td class="r mono">${p.last_price ? '$' + p.last_price.toFixed(2) : '--'}</td>
            <td class="r mono">${p.est_shares || '--'}</td>
            <td class="mono dim">${strategy}</td>
            <td class="muted" style="font-size:11px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${thesisShort}</td>
        </tr>`;
    }).join('');

    // Wire up filters
    document.querySelectorAll('#pos-filters button').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#pos-filters button').forEach(b => b.classList.remove('on'));
            btn.classList.add('on');
            const filter = btn.dataset.filter;
            const filtered = filter === 'all' ? allPositions : allPositions.filter(p => p.direction === filter);
            body.innerHTML = filtered.map(p => {
                const r = p.reasoning || {};
                const strategy = r.committee?.strategy || r.strategy || '--';
                const thesis = r.committee?.reasoning || r.reasoning || '--';
                const thesisShort = typeof thesis === 'string' ? thesis.slice(0, 60) : '--';
                return `<tr>
                    <td class="bright" style="font-weight:500">${p.ticker}</td>
                    <td><span class="chip ${p.direction === 'LONG' ? 'up' : 'down'}">${p.direction}</span></td>
                    <td class="r mono">${p.weight_pct.toFixed(1)}%</td>
                    <td class="r mono">${fmt(p.capital)}</td>
                    <td class="r mono">${p.last_price ? '$' + p.last_price.toFixed(2) : '--'}</td>
                    <td class="r mono">${p.est_shares || '--'}</td>
                    <td class="mono dim">${strategy}</td>
                    <td class="muted" style="font-size:11px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${thesisShort}</td>
                </tr>`;
            }).join('') || '<tr><td colspan="8" class="empty">No matching positions</td></tr>';
        });
    });
}

// ---- ORDER JOURNAL PAGE ----
function renderJournalPage() {
    const body = document.getElementById('journal-body');
    const stats = document.getElementById('journal-stats');

    if (allDecisions.length === 0) {
        body.innerHTML = '<tr><td colspan="7" class="empty">No decisions yet</td></tr>';
        return;
    }

    // Stats
    const okCount = allDecisions.filter(d => d.circuit_breaker_status === 'OK').length;
    const cbTripped = allDecisions.filter(d => d.circuit_breaker_status !== 'OK').length;
    const totalLongs = allDecisions.reduce((s, d) => {
        return s + (d.final_weights || []).filter(w => w > 0.001).length;
    }, 0);

    stats.innerHTML = [
        { l: 'Total runs', v: allDecisions.length, c: 'var(--ink-100)' },
        { l: 'Clean (OK)', v: okCount, c: 'var(--mint)' },
        { l: 'CB tripped', v: cbTripped, c: 'var(--coral)' },
        { l: 'Total longs placed', v: totalLongs, c: 'var(--amber)' },
    ].map((m, i) => `
        <div class="stat" style="${i < 3 ? 'border-right:1px solid var(--hairline)' : ''}">
            <div class="kicker">${m.l}</div>
            <div class="mono" style="font-size:18px;font-weight:600;color:${m.c};margin-top:4px">${m.v}</div>
        </div>
    `).join('');

    body.innerHTML = allDecisions.slice(0, 50).map((d, i) => {
        const weights = d.final_weights || [];
        const tickers = d.ticker_universe || [];
        const longCount = weights.filter(w => w > 0.001).length;
        const gross = weights.reduce((s, w) => s + Math.abs(w), 0);

        return `<tr style="cursor:pointer" onclick="navigateTo('trades');setTimeout(()=>selectDecision(${i}),100)">
            <td class="mono dim" style="font-size:11px">${fmtTime(d.timestamp)}</td>
            <td class="mono">${d.model_version || '--'}</td>
            <td class="mono">${tickers.length}</td>
            <td class="r mono up">${longCount}</td>
            <td class="r mono">${(gross * 100).toFixed(1)}%</td>
            <td><span class="chip ${d.circuit_breaker_status === 'OK' ? 'up' : 'down'}">${d.circuit_breaker_status || '--'}</span></td>
            <td><button class="btn tiny ghost" onclick="event.stopPropagation();navigateTo('trades');setTimeout(()=>selectDecision(${i}),100)">View</button></td>
        </tr>`;
    }).join('');
}
window.navigateTo = navigateTo;

// ---- CIRCUIT BREAKERS PAGE ----
function renderCircuitsPage() {
    const list = document.getElementById('circuits-list');
    const armed = document.getElementById('circuits-armed');

    const circuits = [
        {
            name: 'LongOnlyRule',
            trip: 'w < 0',
            desc: 'Zeroes any negative (short) weights. First rule in the pipeline.',
            detail: 'Enforces long-only constraint across the entire system. Any weight below zero is automatically set to zero before execution. This prevents accidental short positions from any upstream agent or strategy signal.',
        },
        {
            name: 'DrawdownCB',
            trip: '-15% DD',
            desc: 'Hard cutoff if portfolio falls 15% from peak equity.',
            detail: 'Monitors portfolio drawdown in real-time. If current drawdown exceeds the threshold, all new entries are blocked and existing positions may be liquidated to protect capital.',
        },
        {
            name: 'VolatilityCB',
            trip: 'VIX > 35',
            desc: 'Reduces position sizes when VIX spikes above threshold.',
            detail: 'When market fear index (VIX) exceeds 35, position sizes are scaled down proportionally. This reduces exposure during extreme market stress events.',
        },
        {
            name: 'MaxPositionRule',
            trip: '10% cap',
            desc: 'Prevent any single-name from exceeding 10% of equity.',
            detail: 'Caps maximum allocation to any single ticker at 10% of portfolio value. Ensures diversification and prevents concentration risk. Weights are redistributed proportionally if any position exceeds the cap.',
        },
        {
            name: 'TimeWindowRule',
            trip: '09:35-15:55 ET',
            desc: 'Block orders outside regular trading hours.',
            detail: 'Rejects any order placed outside the 09:35 AM to 15:55 PM ET window. Bypassed when SKIP_TIME_CHECK=true is set for testing. Prevents out-of-hours execution issues.',
        },
        {
            name: 'PositionStopLoss',
            trip: '-8% per pos',
            desc: 'Per-position stop loss at -8%.',
            detail: 'Monitors each individual position. If unrealized loss on any position exceeds 8%, an automatic exit order is triggered to limit downside.',
        },
    ];

    armed.textContent = `${circuits.length} / ${circuits.length} armed`;

    list.innerHTML = circuits.map((c, i) => `
        <div class="circ-card ${i === 0 ? 'active' : ''}" data-idx="${i}" onclick="selectCircuit(${i})">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
                <div style="display:flex;align-items:center;gap:8px">
                    <span class="dot live"></span>
                    <span style="font-weight:500;color:var(--ink-100)">${c.name}</span>
                </div>
                <span class="mono dim" style="font-size:11px">Trip @ ${c.trip}</span>
            </div>
            <div style="font-size:11.5px;color:var(--ink-400)">${c.desc}</div>
        </div>
    `).join('');

    // Auto-select first
    selectCircuit(0);

    window._circuits = circuits;
}

function selectCircuit(idx) {
    const circuits = window._circuits || [];
    const c = circuits[idx];
    if (!c) return;

    document.querySelectorAll('.circ-card').forEach(el => el.classList.remove('active'));
    const card = document.querySelector(`.circ-card[data-idx="${idx}"]`);
    if (card) card.classList.add('active');

    document.getElementById('circuit-detail-title').textContent = c.name;
    document.getElementById('circuit-detail').innerHTML = `
        <div style="font-size:13px;color:var(--ink-200);line-height:1.6;margin-bottom:16px">${c.detail}</div>
        <div class="panel" style="background:var(--ink-850)">
            <div style="display:grid;grid-template-columns:repeat(3,1fr)">
                <div style="padding:12px 16px;border-right:1px solid var(--hairline)">
                    <div class="kicker">Rule</div>
                    <div class="mono bright" style="font-size:14px;margin-top:4px">${c.name}</div>
                </div>
                <div style="padding:12px 16px;border-right:1px solid var(--hairline)">
                    <div class="kicker">Trip threshold</div>
                    <div class="mono bright" style="font-size:14px;margin-top:4px">${c.trip}</div>
                </div>
                <div style="padding:12px 16px">
                    <div class="kicker">Status</div>
                    <div style="display:flex;align-items:center;gap:6px;margin-top:4px">
                        <span class="dot live"></span>
                        <span class="bright" style="font-size:14px">Armed</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}
window.selectCircuit = selectCircuit;

// ---- ALPACA SETUP PAGE ----
function renderSetupPage() {
    const statusEl = document.getElementById('setup-status');
    const modeEl = document.getElementById('setup-mode');
    const urlEl = document.getElementById('setup-url');
    const gridEl = document.getElementById('account-grid');
    const syncEl = document.getElementById('setup-last-sync');

    // Check health endpoint
    apiFetch('/health').then(data => {
        statusEl.innerHTML = `
            <span class="dot live"></span>
            <span style="font-size:12px;color:var(--mint)">Connected</span>
        `;
        modeEl.textContent = 'Paper mode';
        urlEl.textContent = 'https://paper-api.alpaca.markets';
    }).catch(() => {
        statusEl.innerHTML = `
            <span class="dot off"></span>
            <span style="font-size:12px;color:var(--coral)">Not connected</span>
        `;
    });

    // Load portfolio data for account snapshot
    apiFetch('/api/portfolio').then(data => {
        const pv = data.current_value || MARKET_CFG.default_capital;
        const dd = (data.drawdown || 0) * 100;
        const pnl = data.total_pnl || 0;
        const histLen = (data.history || []).length;

        syncEl.textContent = 'last sync ' + new Date().toLocaleTimeString('en-US');

        gridEl.innerHTML = [
            { l: 'Portfolio value', v: fmt(pv) },
            { l: 'Total P&L', v: `${pnl >= 0 ? '+' : ''}${fmt(pnl)}` },
            { l: 'Drawdown', v: dd.toFixed(2) + '%' },
            { l: 'Data points', v: histLen + ' sessions' },
        ].map((m, i) => `
            <div class="acct-cell">
                <div class="kicker">${m.l}</div>
                <div class="mono bright" style="font-size:18px;font-weight:600;margin-top:4px">${m.v}</div>
            </div>
        `).join('');
    }).catch(() => {
        gridEl.innerHTML = '<div class="muted" style="padding:16px;font-size:12px">Failed to load account data</div>';
    });

    // Connection test button
    document.getElementById('test-conn-btn').onclick = async () => {
        const btn = document.getElementById('test-conn-btn');
        const result = document.getElementById('conn-test-result');
        btn.textContent = 'Testing...';
        btn.disabled = true;

        const steps = [];
        const t0 = performance.now();

        try {
            // Test health
            let s0 = performance.now();
            await fetch('/health');
            steps.push({ l: 'GET /health', v: `ok | ${Math.round(performance.now() - s0)} ms` });

            // Test portfolio
            s0 = performance.now();
            await apiFetch('/api/portfolio');
            steps.push({ l: 'GET /api/portfolio', v: `ok | ${Math.round(performance.now() - s0)} ms` });

            // Test positions
            s0 = performance.now();
            await apiFetch('/api/positions');
            steps.push({ l: 'GET /api/positions', v: `ok | ${Math.round(performance.now() - s0)} ms` });

            // Test decisions
            s0 = performance.now();
            await apiFetch('/api/decisions');
            steps.push({ l: 'GET /api/decisions', v: `ok | ${Math.round(performance.now() - s0)} ms` });

            const totalMs = Math.round(performance.now() - t0);

            result.innerHTML = `
                <div style="display:flex;align-items:center;gap:8px;font-size:13px;color:var(--mint);margin-bottom:10px">
                    <span class="dot live"></span>
                    <b>All endpoints responded</b>
                    <span class="muted" style="font-weight:400">| round-trip ${totalMs} ms</span>
                </div>
                ${steps.map(s => `
                    <div class="test-step">
                        <span class="label">${s.l}</span>
                        <span class="val">${s.v}</span>
                    </div>
                `).join('')}
            `;
        } catch (e) {
            result.innerHTML = `
                <div style="display:flex;align-items:center;gap:8px;font-size:13px;color:var(--coral)">
                    <span class="dot off"></span>
                    <b>Connection failed</b>
                    <span class="muted" style="font-weight:400">| ${e.message}</span>
                </div>
            `;
        }

        btn.textContent = 'Run test';
        btn.disabled = false;
    };
}

// ---- Bootstrap ----
document.addEventListener('DOMContentLoaded', async () => {
    document.getElementById('login-btn').addEventListener('click', doLogin);
    document.getElementById('login-password').addEventListener('keydown', e => {
        if (e.key === 'Enter') doLogin();
    });

    // Load market config
    try {
        const cfg = await fetch('/api/market-config').then(r => r.json());
        MARKET_CFG = { ...MARKET_CFG, ...cfg };
        document.getElementById('bench-label').textContent = MARKET_CFG.benchmark_label;
        document.title = `AegisQuant Terminal (${MARKET_CFG.market})`;
    } catch {}

    await checkAuth();
    await fetchAll();
    connectWebSocket();

    // HTTP polling fallback (30s)
    setInterval(fetchAll, 30000);
});
