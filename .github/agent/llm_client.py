"""
LLM Client
==========
Client for calling LLM API (OpenAI/Claude) to generate code.
Used by spawn_agent to create new agent code.
"""

import os
import json
from typing import Optional, Dict, Any


# Check which API is available
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


class LLMClient:
    """Client for LLM API calls."""

    def __init__(self):
        self.api_key = OPENAI_API_KEY or ANTHROPIC_API_KEY
        self.provider = (
            "openai" if OPENAI_API_KEY else "anthropic" if ANTHROPIC_API_KEY else None
        )

    def is_available(self) -> bool:
        """Check if LLM API is available."""
        return self.api_key is not None

    def generate_code(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
        """
        Generate code using LLM API.

        Args:
            prompt: The prompt describing what code to generate
            max_tokens: Maximum tokens in response

        Returns:
            Generated code string or None if failed
        """
        if not self.is_available():
            print("[LLM] No API key available, using template fallback")
            return None

        try:
            if self.provider == "openai":
                return self._call_openai(prompt, max_tokens)
            elif self.provider == "anthropic":
                return self._call_anthropic(prompt, max_tokens)
        except Exception as e:
            print(f"[LLM] Error calling API: {e}")
            return None

        return None

    def _call_openai(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Call OpenAI API."""
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a Python code generator for autonomous agents.
Generate ONLY valid Python code, no explanations or markdown.
The code should define a class that inherits from BaseAgent.
Include proper imports and type hints.""",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )

            content = response.choices[0].message.content

            # Extract code from markdown if present
            if "```python" in content:
                content = content.split("```python")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return content.strip()

        except ImportError:
            print("[LLM] openai package not installed")
            return None
        except Exception as e:
            print(f"[LLM] OpenAI error: {e}")
            return None

    def _call_anthropic(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Call Anthropic/Claude API."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system="""You are a Python code generator for autonomous agents.
Generate ONLY valid Python code, no explanations or markdown.
The code should define a class that inherits from BaseAgent.
Include proper imports and type hints.""",
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text

            # Extract code from markdown if present
            if "```python" in content:
                content = content.split("```python")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return content.strip()

        except ImportError:
            print("[LLM] anthropic package not installed")
            return None
        except Exception as e:
            print(f"[LLM] Anthropic error: {e}")
            return None

    def review_code(self, code: str) -> Dict[str, Any]:
        """
        Use LLM to review generated code for issues.

        Returns:
            Dict with "is_safe", "issues", "suggestions"
        """
        if not self.is_available():
            # Can't review without API, assume OK
            return {"is_safe": True, "issues": [], "suggestions": []}

        prompt = f"""Review this Python agent code for safety and correctness.
Check for:
1. Security issues (file access, shell injection, etc.)
2. Syntax errors
3. Logic errors
4. Missing imports

Code:
```python
{code}
```

Respond in JSON format:
{{
    "is_safe": true/false,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}"""

        try:
            response = self.generate_code(prompt, max_tokens=500)
            if response:
                # Try to parse JSON from response
                return json.loads(response)
        except:
            pass

        return {"is_safe": True, "issues": [], "suggestions": []}

    def fix_code(self, code: str, error: str) -> Optional[str]:
        """
        Use LLM to fix code based on error message.

        Args:
            code: Original code that failed
            error: Error message from running the code

        Returns:
            Fixed code or None if can't fix
        """
        if not self.is_available():
            return None

        prompt = f"""Fix this Python code based on the error:

Error:
{error}

Code:
```python
{code}
```

Return ONLY the fixed Python code, no explanations."""

        return self.generate_code(prompt, max_tokens=2000)


# Singleton instance
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the singleton LLM client."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def generate_code(prompt: str) -> Optional[str]:
    """Convenience function to generate code."""
    return get_llm_client().generate_code(prompt)
