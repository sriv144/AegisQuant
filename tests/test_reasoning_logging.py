from collections import defaultdict

from src.db.models import AgentReasoning, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def test_agent_reasoning_rows_group_by_ticker(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'reasoning.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    run_id = "run_test_reasoning"
    session = Session()
    try:
        for i in range(5):
            session.add(
                AgentReasoning(
                    run_id=run_id,
                    ticker="AAA.NS" if i < 3 else "BBB.NS",
                    agent_name=f"Agent_{i}",
                    action="PROPOSE_LONG" if i % 2 == 0 else "HOLD",
                    confidence=0.5 + i * 0.05,
                    rationale=f"Reason {i}",
                )
            )
        session.commit()

        rows = session.query(AgentReasoning).filter(AgentReasoning.run_id == run_id).all()
        grouped = defaultdict(list)
        for row in rows:
            grouped[row.ticker].append(row)

        assert len(rows) == 5
        assert len(grouped["AAA.NS"]) == 3
        assert len(grouped["BBB.NS"]) == 2
        assert grouped["AAA.NS"][0].agent_name.startswith("Agent_")
        assert grouped["AAA.NS"][0].rationale
    finally:
        session.close()
