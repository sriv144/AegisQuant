import json
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..state import AgentState

class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Sentiment_Research_Agent",
            role="Alternative Data & Sentiment Analyst tracking news feeds, social media, and retail flow. You capture irrational exuberance or panic before technicals do."
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        print(f"[{self.name}] Analyzing sentiment for {state['current_asset']}...")
        
        # Prepare context
        alt_data = state.get("alternative_data", {})
        
        prompt = f"""
        Analyze the following sentiment aggregation for {state['current_asset']}:
        {json.dumps(alt_data, indent=2)}
        
        Evaluate public sentiment, news momentum, and fear/greed.
        Produce a JSON output matching this schema:
        {{
            "agent_name": "Sentiment_Research_Agent",
            "action": "PROPOSE_LONG" | "PROPOSE_SHORT" | "HOLD",
            "confidence": float (0.0 to 1.0),
            "rationale": "A concise explanation of the sentiment signal."
        }}
        """
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": prompt},
            ])
            decision = self._parse_llm_json(response.content)
        except Exception as exc:
            print(f"[{self.name}] LLM call failed: {exc}")
            decision = {
                "agent_name": self.name,
                "action": "HOLD",
                "confidence": 0.0,
                "rationale": f"LLM unavailable: {exc}",
            }

        return {"research_signals": [decision]}

sentiment_agent = SentimentAgent()
