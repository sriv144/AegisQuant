// Utility to format currency
const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        maximumFractionDigits: 2
    }).format(value);
};

// Global chart instance
let pnlChartInstance = null;

// Initialization
document.addEventListener('DOMContentLoaded', async () => {
    await fetchDashboardData();
    setInterval(fetchDashboardData, 30000); // Refresh every 30s
});

async function fetchDashboardData() {
    try {
        const [portfolioRes, positionsRes, decisionsRes, latestRunRes] = await Promise.all([
            fetch('/api/portfolio'),
            fetch('/api/positions'),
            fetch('/api/decisions'),
            fetch('/api/latest-run')
        ]);

        const portfolioData = await portfolioRes.json();
        const positionsData = await positionsRes.json();
        const decisionsData = await decisionsRes.json();
        const latestRunData = await latestRunRes.json();

        updateMetrics(portfolioData, positionsData, latestRunData);
        renderChart(portfolioData.history);
        renderPositions(positionsData);
        renderDecisions(decisionsData);
        renderLatestRun(latestRunData);

    } catch (e) {
        console.error("Failed to fetch dashboard data", e);
    }
}

function updateMetrics(portfolio, _positions, latestRun) {
    document.getElementById('val-portfolio').textContent = formatCurrency(portfolio.current_value || 250000);

    const pnlEl = document.getElementById('val-pnl');
    const pnl = portfolio.total_pnl || 0;
    pnlEl.textContent = `${pnl >= 0 ? '+' : ''}${formatCurrency(pnl)} (Total P&L)`;
    pnlEl.className = pnl >= 0 ? 'subtext text-success' : 'subtext text-danger';

    const dd = (portfolio.drawdown || 0) * 100;
    document.getElementById('val-drawdown').textContent = `${dd.toFixed(2)}%`;

    const summary = latestRun && latestRun.summary ? latestRun.summary : {};
    const posCount = (summary.long_count || 0) + (summary.short_count || 0);
    document.getElementById('val-positions').textContent = posCount;
    document.getElementById('val-exposure').textContent =
        `Gross: ${summary.gross_exposure_pct || 0}%  |  Net: ${summary.net_exposure_pct || 0}%`;

    if (latestRun && latestRun.timestamp) {
        const ts = new Date(latestRun.timestamp);
        document.getElementById('val-last-run').textContent = ts.toLocaleString('en-IN', {timeZone:'Asia/Kolkata'});
        document.getElementById('val-model').textContent = `Model: ${latestRun.model_version || '—'}`;
    }
}

function renderChart(history) {
    if (!history || history.length === 0) return;
    
    const ctx = document.getElementById('pnlChart').getContext('2d');
    
    // Prepare Data
    const labels = history.map(row => row.date);
    const data = history.map(row => row.total_portfolio_value);

    // Create Gradient for the line
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(0, 208, 156, 0.5)'); // Groww green
    gradient.addColorStop(1, 'rgba(0, 208, 156, 0.0)');

    if (pnlChartInstance) {
        pnlChartInstance.data.labels = labels;
        pnlChartInstance.data.datasets[0].data = data;
        pnlChartInstance.update();
        return;
    }

    Chart.defaults.color = '#8b92a5';
    Chart.defaults.font.family = 'Outfit';

    pnlChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Portfolio Value',
                data: data,
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
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(26, 29, 39, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#00d09c',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return formatCurrency(context.parsed.y);
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false, drawBorder: false },
                    ticks: { maxTicksLimit: 7 }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                    ticks: {
                        callback: function(value) {
                            return '₹' + (value / 1000).toFixed(0) + 'k';
                        }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

function renderPositions(positions) {
    const tbody = document.getElementById('positions-body');
    tbody.innerHTML = '';

    if (!positions || positions.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-secondary);">No active positions found.</td></tr>`;
        return;
    }

    positions.forEach(pos => {
        const tr = document.createElement('tr');
        
        const typeClass = pos.trade_type === 'LONG' ? 'badge-long' : 'badge-short';
        const pnlClass = parseFloat(pos.pnl_pct) >= 0 ? 'text-success' : 'text-danger';
        
        tr.innerHTML = `
            <td style="font-weight: 600;">${pos.ticker}</td>
            <td>${pos.quantity}</td>
            <td>${formatCurrency(pos.entry_price)}</td>
            <td><span class="badge ${typeClass}">${pos.trade_type || 'CNC'}</span></td>
            <td class="${pnlClass}">${(pos.pnl_pct * 100).toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderLatestRun(data) {
    const tbody = document.getElementById('latest-run-body');
    const tsEl = document.getElementById('run-timestamp');
    const summaryEl = document.getElementById('run-summary');
    if (!data || !data.positions) return;

    if (data.timestamp) {
        tsEl.textContent = '— ' + new Date(data.timestamp).toLocaleString('en-IN', {timeZone: 'Asia/Kolkata'});
    }

    const s = data.summary || {};
    summaryEl.innerHTML = `
        <span>Universe: <b>${data.universe_size || 0}</b></span>
        <span style="color:#00d09c">Longs: <b>${s.long_count || 0}</b></span>
        <span style="color:#f04b4b">Shorts: <b>${s.short_count || 0}</b></span>
        <span>Gross: <b>${s.gross_exposure_pct || 0}%</b></span>
        <span>Net: <b>${s.net_exposure_pct || 0}%</b></span>
        <span>CB: <b>${data.circuit_breaker || '—'}</b></span>
    `;

    tbody.innerHTML = '';
    if (!data.positions.length) {
        tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;opacity:0.5">No positions in last run.</td></tr>`;
        return;
    }

    data.positions.forEach(pos => {
        const tr = document.createElement('tr');
        const isLong = pos.direction === 'LONG';
        const dirClass = isLong ? 'badge-long' : 'badge-short';
        const weightColor = isLong ? 'var(--accent-primary)' : '#f04b4b';
        tr.innerHTML = `
            <td style="font-weight:600">${pos.ticker.replace('.NS', '')}</td>
            <td><span class="badge ${dirClass}">${pos.direction}</span></td>
            <td style="color:${weightColor}">${pos.weight_pct > 0 ? '+' : ''}${pos.weight_pct.toFixed(2)}%</td>
            <td>₹${pos.rupees.toLocaleString('en-IN')}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderDecisions(decisions) {
    const tbody = document.getElementById('decisions-body');
    tbody.innerHTML = '';

    if (!decisions || decisions.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; color: var(--text-secondary);">No recent decisions found.</td></tr>`;
        return;
    }

    // Only show top 10 decisions
    decisions.slice(0, 10).forEach(dec => {
        const tr = document.createElement('tr');
        
        const timestamp = new Date(dec.timestamp).toLocaleString();
        
        tr.innerHTML = `
            <td>${timestamp}</td>
            <td style="color:var(--accent-primary)">${dec.model_version}</td>
            <td>${dec.circuit_breaker_status}</td>
            <td>${(dec.transaction_costs || 0).toFixed(2)} bps</td>
        `;
        tbody.appendChild(tr);
    });
}
