from src.agents.base_agent import BaseAgent
from src.execution.alpaca_executor import AlpacaExecutor


def test_parse_llm_json_extracts_fenced_json():
    agent = BaseAgent(name="TestAgent", role="tester")
    response = """
    Here is the decision.

    ```json
    {
      "agent_name": "TestAgent",
      "action": "HOLD",
      "confidence": 0.6,
      "rationale": "Structured payload is present."
    }
    ```
    """

    parsed = agent._parse_llm_json(response)

    assert parsed["action"] == "HOLD"
    assert parsed["confidence"] == 0.6


def test_alpaca_executor_stays_mock_when_execution_flag_disabled(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ENABLE_BROKER_EXECUTION", "False")

    executor = AlpacaExecutor(["SPY"])

    assert executor.mock_mode is True
