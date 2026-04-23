import json
from openai import AzureOpenAI, OpenAI


class Judge:
    """
    Scores answers using one of three providers:

      1. Azure OpenAI  — if AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY are set
      2. Azure Foundry / OpenAI with custom base_url — if OPENAI_BASE_URL is set
      3. OpenAI        — if only OPENAI_API_KEY is set
      4. None          — returns fallback score of 5
    """

    SYSTEM_PROMPT = """You are an expert judge in a Latin America knowledge tournament.
You evaluate answers for factual accuracy, completeness, and precision.
You must respond ONLY with a valid JSON object — no preamble, no markdown."""

    SCORE_PROMPT = """Topic: {topic}
Question: {question}
Answer given: {answer}

Score the answer from 1 to 10:
  10 — Perfect: factually correct, complete, precise
  7-9 — Good: correct with minor gaps or imprecision
  4-6 — Partial: some correct elements but significant gaps
  1-3 — Poor: mostly incorrect or irrelevant

Respond ONLY with this JSON:
{{"score": <integer 1-10>, "reason": "<one concise sentence explaining the score>"}}"""

    def __init__(
        self,
        azure_endpoint:   str = "",
        azure_key:        str = "",
        azure_deployment: str = "gpt-4o",
        openai_key:       str = "",
        openai_model:     str = "gpt-4o-mini",
        openai_base_url:  str = "",
    ):
        if azure_endpoint and azure_key:
            # Option 1: Azure OpenAI (classic)
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                api_version="2024-02-01",
            )
            self.deployment = azure_deployment
            self._provider  = "azure"

        elif openai_key and openai_base_url:
            # Option 2: Azure Foundry or any OpenAI-compatible endpoint
            self.client = OpenAI(
                base_url=openai_base_url,
                api_key=openai_key,
            )
            self.deployment = openai_model
            self._provider  = "azure-foundry"

        elif openai_key:
            # Option 3: Standard OpenAI
            self.client     = OpenAI(api_key=openai_key)
            self.deployment = openai_model
            self._provider  = "openai"

        else:
            # Option 4: No provider configured
            self.client     = None
            self.deployment = None
            self._provider  = "none"

        print(f"[Judge] Provider: {self._provider}"
              + (f" model={self.deployment}" if self.deployment else ""))

    def score(self, topic: str, question: str, answer: str) -> tuple[int, str]:
        """Returns (score 1-10, one-sentence reason)."""
        if self.client is None:
            return 5, "No judge configured — set OPENAI_API_KEY or AZURE_OPENAI_KEY"

        prompt = self.SCORE_PROMPT.format(
            topic=topic, question=question, answer=answer
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            raw  = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            return int(data["score"]), str(data["reason"])
        except Exception as e:
            return 5, f"Judge error: {e}"