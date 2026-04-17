// ── Globals ──────────────────────────────────────────────────────────────────
let pnlChartInstance = null;
let exposureChartInstance = null;
let holdingsBarInstance = null;
let allPositions = [];
let allDecisions = [];

const fmt = (v) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v);
const fmtPct = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
const fmtIST = (ts) => new Date(ts).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });

// ── Routing ───────────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        const page = link.dataset.page;
        document.querySelectorAll('.nav-item').forEach(l => l.classList.remove('active'));
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        link.classList.add('active');
        document.getElementById(`page-${page}`).classList.add('active');
        document.querySelector('.topbar h1').textContent =
            page === 'dashboard' ? 'Overview' : page.charAt(0).toUpperCase() + page.slice(1);
        if (page === 'positions') renderPositionsPage();
        if (page === 'decisions') renderDecisionsPage();
    });
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    await fetchAll();
    setInterval(fetchAll, 30000);
});

async function fetchAll() {
    const [portfolioRes, latestRunRes, decisionsRes] = await Promise.all([
        fetch('/api/portfolio').then(r => r.json()).catch(() => ({})),
        fetch('/api/latest-run').then(r => r.json()).catch(() => ({})),
        fetch('/api/decisions').then(r => r.json()).catch(() => []),
    ]);

    allPositions = latestRunRes.positions || [];
    allDecisions = Array.isArray(decisionsRes) ? decisionsRes : [];

    updateKPIs(portfolioRes, latestRunRes);
    renderPnlChart(portfolioRes.history || []);
    renderLatestRunTable(latestRunRes);

    document.getElementById('last-refresh').textContent = 'Updated ' + new Date().toLocaleTimeString('en-IN');
}

// ── KPI Cards ─────────────────────────────────────────────────────────────────
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
    document.getElementById('val-positions').textContent = posCount || '—';
    document.getElementById('val-exposure').textContent =
        posCount ? `${s.long_count} long · ${s.short_count} short` : 'No positions yet';

    document.getElementById('val-gross').textContent = s.gross_exposure_pct != null ? s.gross_exposure_pct + '%' : '—';
    document.getElementById('val-net-exp').textContent = s.net_exposure_pct != null ? `Net: ${s.net_exposure_pct}%` : '—';

    if (run.timestamp) {
        document.getElementById('val-last-run').textContent = fmtIST(run.timestamp);
        document.getElementById('val-model').textContent = run.model_version || '—';
    }
}

// ── P&L Chart ─────────────────────────────────────────────────────────────────
function renderPnlChart(history) {
    const ctx = document.getElementById('pnlChart').getContext('2d');

    // Pad single-day data so chart renders visibly
    let labels, data;
    if (history.length <= 1) {
        const base = history.length === 1 ? history[0].total_portfolio_value : 250000;
        labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Today'];
        data = [base, base, base, base, base, base];
    } else {
        labels = history.map(r => r.date);
        data = history.map(r => r.total_portfolio_value);
    }

    const gradient = ctx.createLinearGradient(0, 0, 0, 320);
    gradient.addColorStop(0, 'rgba(0,208,156,0.35)');
    gradient.addColorStop(1, 'rgba(0,208,156,0.0)');

    const cfg = {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Portfolio Value',
                data,
                borderColor: '#00d09c',
                backgroundColor: gradient,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointBackgroundColor: '#00d09c',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index', intersect: false,
                    backgroundColor: 'rgba(26,29,39,0.95)',
                    titleColor: '#fff', bodyColor: '#00d09c',
                    borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, padding: 12,
                    callbacks: { label: ctx => fmt(ctx.parsed.y) }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { maxTicksLimit: 7 } },
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
        pnlChartInstance.data.datasets[0].data = data;
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
        `<span class="pill ${s.circuit_breaker === 'OK' ? 'pill-green' : 'pill-red'}">CB <b>${data.circuit_breaker || '—'}</b></span>`,
    ].join('');

    tbody.innerHTML = '';
    if (!data.positions.length) {
        tbody.innerHTML = `<tr><td colspan="4" class="empty-state">No positions in last run.</td></tr>`;
        return;
    }
    data.positions.forEach(pos => {
        const isLong = pos.direction === 'LONG';
        tbody.insertAdjacentHTML('beforeend', `
            <tr>
                <td class="symbol-cell">${pos.ticker.replace('.NS', '')}</td>
                <td><span class="badge ${isLong ? 'badge-long' : 'badge-short'}">${pos.direction}</span></td>
                <td class="${isLong ? 'text-success' : 'text-danger'}">${fmtPct(pos.weight_pct)}</td>
                <td>${fmt(pos.rupees)}</td>
            </tr>`);
    });
}

// ── POSITIONS PAGE ─────────────────────────────────────────────────────────────
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

    if (exposureChartInstance) { exposureChartInstance.destroy(); }
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
        data: {
            labels,
            datasets: [{ data, backgroundColor: colors, borderRadius: 4 }]
        },
        options: {
            indexAxis: 'y', responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => fmtPct(c.parsed.x) } } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { callback: v => v + '%' } },
                y: { grid: { display: false } }
            }
        }
    };

    if (holdingsBarInstance) { holdingsBarInstance.destroy(); }
    holdingsBarInstance = new Chart(ctx, cfg);
}

function renderFullPositionsTable(filter) {
    const tbody = document.getElementById('positions-full-body');
    const rows = filter === 'all' ? allPositions : allPositions.filter(p => p.direction === filter);

    tbody.innerHTML = '';
    if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-state">No ${filter === 'all' ? '' : filter.toLowerCase() + ' '}positions.</td></tr>`;
        return;
    }
    rows.forEach(pos => {
        const isLong = pos.direction === 'LONG';
        tbody.insertAdjacentHTML('beforeend', `
            <tr>
                <td class="symbol-cell">${pos.ticker.replace('.NS', '')}</td>
                <td><span class="badge ${isLong ? 'badge-long' : 'badge-short'}">${pos.direction}</span></td>
                <td class="${isLong ? 'text-success' : 'text-danger'}">${fmtPct(pos.weight_pct)}</td>
                <td>${fmt(pos.rupees)}</td>
                <td class="dim">${pos.last_price ? '₹' + pos.last_price.toLocaleString('en-IN') : '—'}</td>
                <td class="dim">${pos.est_shares ?? '—'}</td>
            </tr>`);
    });
}

// ── DECISIONS PAGE ────────────────────────────────────────────────────────────
function renderDecisionsPage() {
    const tbody = document.getElementById('decisions-full-body');
    tbody.innerHTML = '';

    if (!allDecisions.length) {
        tbody.innerHTML = `<tr><td colspan="8" class="empty-state">No decisions logged yet.</td></tr>`;
        return;
    }

    allDecisions.forEach((dec, idx) => {
        const weights = Array.isArray(dec.final_weights) ? dec.final_weights : [];
        const longs = weights.filter(w => w > 0.001).length;
        const shorts = weights.filter(w => w < -0.001).length;
        const gross = weights.reduce((s, w) => s + Math.abs(w), 0);
        const cbOk = dec.circuit_breaker_status === 'OK';

        const tr = document.createElement('tr');
        tr.classList.add('clickable-row');
        tr.innerHTML = `
            <td>${fmtIST(dec.timestamp)}</td>
            <td class="text-success" style="font-size:0.82rem">${dec.model_version}</td>
            <td>${weights.length}</td>
            <td class="text-success">${longs}</td>
            <td class="text-danger">${shorts}</td>
            <td>${(gross * 100).toFixed(1)}%</td>
            <td><span class="badge ${cbOk ? 'badge-long' : 'badge-short'}">${dec.circuit_breaker_status}</span></td>
            <td>${(dec.transaction_costs || 0).toFixed(2)}</td>
        `;
        tr.addEventListener('click', () => openDecisionDetail(dec, idx));
        tbody.appendChild(tr);
    });

    document.getElementById('close-detail').addEventListener('click', () => {
        document.getElementById('decision-detail-card').style.display = 'none';
    });
}

function openDecisionDetail(dec, idx) {
    const card = document.getElementById('decision-detail-card');
    const body = document.getElementById('detail-body');
    document.getElementById('detail-title').textContent = `Run #${idx + 1} — ${fmtIST(dec.timestamp)}`;

    const weights = Array.isArray(dec.final_weights) ? dec.final_weights : [];
    // We don't have ticker list in decisions response — parse from latest-run if same run
    const positions = allPositions;

    let rows = '';
    if (positions.length && weights.length === (allPositions.length || weights.length)) {
        // Use live position data if this is the latest decision
        positions.forEach(pos => {
            const isLong = pos.direction === 'LONG';
            rows += `<tr>
                <td class="symbol-cell">${pos.ticker.replace('.NS', '')}</td>
                <td><span class="badge ${isLong ? 'badge-long' : 'badge-short'}">${pos.direction}</span></td>
                <td class="${isLong ? 'text-success' : 'text-danger'}">${fmtPct(pos.weight_pct)}</td>
                <td>${fmt(pos.rupees)}</td>
            </tr>`;
        });
    } else {
        // Fallback: just show non-zero weights
        weights.forEach((w, i) => {
            if (Math.abs(w) < 0.001) return;
            const isLong = w > 0;
            rows += `<tr>
                <td class="symbol-cell">Ticker #${i + 1}</td>
                <td><span class="badge ${isLong ? 'badge-long' : 'badge-short'}">${isLong ? 'LONG' : 'SHORT'}</span></td>
                <td class="${isLong ? 'text-success' : 'text-danger'}">${fmtPct(w * 100)}</td>
                <td>${fmt(Math.abs(w) * 250000)}</td>
            </tr>`;
        });
    }

    body.innerHTML = `
        <div class="detail-meta">
            <span>Model: <b>${dec.model_version}</b></span>
            <span>Circuit Breaker: <b class="${dec.circuit_breaker_status === 'OK' ? 'text-success' : 'text-danger'}">${dec.circuit_breaker_status}</b></span>
            <span>Slippage: <b>${(dec.transaction_costs || 0).toFixed(2)} bps</b></span>
        </div>
        <table class="modern-table">
            <thead><tr><th>Symbol</th><th>Direction</th><th>Weight</th><th>Capital</th></tr></thead>
            <tbody>${rows || '<tr><td colspan="4" class="empty-state">All weights were zero (circuit breaker or no signals).</td></tr>'}</tbody>
        </table>
    `;

    card.style.display = 'block';
    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
