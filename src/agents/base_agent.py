import os
import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI

from src import config  # noqa: F401

class BaseAgent:
    def __init__(self, name: str, role: str, model_name: str = "gpt-4-turbo-preview"):
        self.name = name
        self.role = role
        self.model_name = model_name
        self.llm = None

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            try:
                self.llm = ChatOpenAI(model=model_name, temperature=0.2)
            except Exception as exc:
                print(f"[{self.name}] Failed to initialize ChatOpenAI: {exc}")
        
    def _create_system_prompt(self, additional_instructions: str = "") -> str:
        base = f"""You are {self.name}, an expert AI hedge fund agent with the following role: {self.role}.
        You must analyze the incoming state, think step-by-step, and output your final decision in a structured JSON format.
        Your JSON output MUST match the requested Pydantic schema exactly.
        """
        if additional_instructions:
            base += f"\n\nSpecial Instructions: {additional_instructions}"
        return base

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract invocation method. 
        Each agent subclass must implement how it processes the state.
        """
        raise NotImplementedError("Subclasses must implement the invoke method.")
        
    def _parse_llm_json(self, response: str) -> Dict[str, Any]:
        """
        Utility to extract JSON from LLM output.
        """
        try:
            # Strip markdown formatting if present
            clean_str = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_str)
        except json.JSONDecodeError:
            print(f"[{self.name}] Error parsing JSON from LLM output: {response}")
            return {"error": "Failed to parse decision."}

    def _invoke_llm_json(self, prompt: str, fallback: Dict[str, Any], additional_instructions: str = "") -> Dict[str, Any]:
        if self.llm is None:
            return fallback

        try:
            response = self.llm.invoke([
                {"role": "system", "content": self._create_system_prompt(additional_instructions)},
                {"role": "user", "content": prompt},
            ])
            return self._parse_llm_json(response.content)
        except Exception as exc:
            print(f"[{self.name}] LLM call failed: {exc}")
            return fallback
