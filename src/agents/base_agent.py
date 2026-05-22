import os
import json
import re
from typing import Dict, Any

from langchain_openai import ChatOpenAI

from src import config  # noqa: F401

class BaseAgent:
    def __init__(self, name: str, role: str, model_name: str | None = None):
        self.name = name
        self.role = role
        self.model_name = model_name
        self.llm = None

        if os.getenv("ENABLE_LLM_AGENTS", "True").lower() != "true":
            return

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            try:
                use_openrouter = api_key.startswith("sk-or-v1")
                resolved_model = (
                    model_name
                    or os.getenv("LLM_MODEL", "").strip()
                    or ("openai/gpt-4o-mini" if use_openrouter else "gpt-4o-mini")
                )
                client_kwargs = {
                    "model": resolved_model,
                    "temperature": 0.2,
                    "api_key": api_key,
                    "max_tokens": 1024,  # Keep responses concise, save credits
                }

                base_url = os.getenv("OPENAI_BASE_URL", "").strip()
                if base_url:
                    client_kwargs["base_url"] = base_url
                elif use_openrouter:
                    client_kwargs["base_url"] = "https://openrouter.ai/api/v1"

                self.model_name = resolved_model
                self.llm = ChatOpenAI(**client_kwargs)
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

    def _format_memory_context(self, state: Dict[str, Any], max_chars: int = 4500) -> str:
        """
        Render concise persistent memory blocks for prompts.

        The memory is markdown maintained by journal.py. Keep it bounded so it
        informs the agents without swallowing the prompt budget.
        """
        memory = state.get("context_memory") or {}
        if not isinstance(memory, dict):
            return ""

        labels = [
            ("strategy", "Persistent strategy rules"),
            ("learnings", "Recent weekly learnings"),
            ("recent_trades", "Recent trade log"),
        ]
        blocks = []
        for key, label in labels:
            value = str(memory.get(key) or "").strip()
            if not value:
                continue
            if len(value) > max_chars:
                value = value[-max_chars:]
            blocks.append(f"{label}:\n{value}")

        if not blocks:
            return ""
        return "\n\nPersistent agent memory:\n" + "\n\n".join(blocks)

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
        candidates = []

        fenced_json = re.findall(r"```json\s*(\{.*?\})\s*```", response, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(fenced_json)

        fenced_any = re.findall(r"```\s*(\{.*?\})\s*```", response, flags=re.DOTALL)
        candidates.extend(fenced_any)

        first_brace = response.find("{")
        last_brace = response.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidates.append(response[first_brace:last_brace + 1])

        try:
            for candidate in candidates or [response]:
                clean_str = candidate.strip()
                try:
                    return json.loads(clean_str)
                except json.JSONDecodeError:
                    continue
            print(f"[{self.name}] Error parsing JSON from LLM output: {response}")
            return {"error": "Failed to parse decision."}
        except Exception:
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
            parsed = self._parse_llm_json(response.content)
            return fallback if parsed.get("error") else parsed
        except Exception as exc:
            print(f"[{self.name}] LLM call failed: {exc}")
            return fallback
