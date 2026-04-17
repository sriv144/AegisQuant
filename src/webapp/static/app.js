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
        const [portfolioRes, positionsRes, decisionsRes] = await Promise.all([
            fetch('/api/portfolio'),
            fetch('/api/positions'),
            fetch('/api/decisions')
        ]);

        const portfolioData = await portfolioRes.json();
        const positionsData = await positionsRes.json();
        const decisionsData = await decisionsRes.json();

        updateMetrics(portfolioData, positionsData);
        renderChart(portfolioData.history);
        renderPositions(positionsData);
        renderDecisions(decisionsData);

    } catch (e) {
        console.error("Failed to fetch dashboard data", e);
    }
}

function updateMetrics(portfolio, positions) {
    document.getElementById('val-portfolio').textContent = formatCurrency(portfolio.current_value || 0);
    
    // PnL subtext
    const pnlEl = document.getElementById('val-pnl');
    const pnl = portfolio.total_pnl || 0;
    pnlEl.textContent = `${pnl >= 0 ? '+' : ''}${formatCurrency(pnl)} (Total P&L)`;
    pnlEl.className = pnl >= 0 ? 'subtext text-success' : 'subtext text-danger';

    // Drawdown
    const dd = (portfolio.drawdown || 0) * 100;
    document.getElementById('val-drawdown').textContent = `${dd.toFixed(2)}%`;
    
    // Positions Count
    document.getElementById('val-positions').textContent = positions.length || 0;
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
