import json
from openai import AzureOpenAI


class Judge:
    """
    Scores answers using Azure OpenAI.
    Only CENIA's credentials — participants never touch this.
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

    def __init__(self, endpoint: str, api_key: str, deployment: str):
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-02-01",
        )
        self.deployment = deployment

    def score(self, topic: str, question: str, answer: str) -> tuple[int, str]:
        """Returns (score 1-10, one-sentence reason)."""
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
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            return int(data["score"]), str(data["reason"])
        except Exception as e:
            return 5, f"Judge error: {e}"
