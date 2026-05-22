# AegisQuant Trading Strategy

Persistent rules the research + committee agents should honour on every run.
Edit freely - the pipeline reads this file at the start of each cycle and
passes it to the LLM committee as `context_memory.strategy`.

## Capital

- Base capital: Rs 2,50,000 (paper mode)
- Split: 80% delivery (CNC, 1-3 month holds) + 20% intraday (MIS, same-day)
- Max gross exposure: 1.5x

## Risk

- Per-CNC position: SL -8%, TP +20%, max hold 90 days
- Per-MIS position: SL -1.5%, TP +2%, must close same day
- Hard drawdown cut-off: 15% from peak equity
- No-trade windows: before 09:15 IST, after 15:25 IST

## Universe

- NSE only - screened weekly for >= Rs 10 crore avg daily traded value
- Skip stocks in circuit or with < 20 trading days of history

## Philosophy

- Long-term fundamental-first, short-term momentum overlay
- Committee confidence floor: 0.4 for CNC, 0.5 for MIS
- Beat NIFTY 50, not the broker - measured vs. ^NSEI
