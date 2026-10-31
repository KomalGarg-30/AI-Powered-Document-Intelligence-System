"""Groq vision service for transcribing document images."""

import base64
import logging
from pathlib import Path

from groq import AsyncGroq

logger = logging.getLogger(__name__)


TRANSCRIPTION_PROMPT = """Analyze this document image and provide a detailed description for document retrieval.

Include the visual type, subject, key data or labeled components, relationships, and all legible text.
For tables, preserve the data in Markdown table syntax. For diagrams and charts, explain the structure,
labels, values, and relationships. Do not infer details that are not visible."""


class GroqVisionTranscriber:
    """Transcribe extracted figures and tables with a Groq vision model."""

    def __init__(self, api_key: str, model_name: str) -> None:
        self.client = AsyncGroq(api_key=api_key)
        self.model_name = model_name
        logger.info("Initialized Groq vision transcriber with model: %s", model_name)

    @staticmethod
    def _mime_type(image_path: Path) -> str:
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        return mime_types.get(image_path.suffix.lower(), "image/png")

    async def transcribe(self, image_path: Path) -> str:
        """Return a searchable description of an extracted image."""
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
        image_url = (
            f"data:{self._mime_type(image_path)};base64,{encoded_image}"
        )
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_completion_tokens=2048,
        )
        transcription = response.choices[0].message.content
        if not transcription or not transcription.strip():
            raise RuntimeError(
                f"Groq returned an empty transcription for {image_path.name}"
            )
        return transcription.strip()

    async def generate_summary(self, full_text: str) -> str:
        """Create a local preview summary without an extra API request."""
        preview = full_text[:1500].strip()
        return f"DOCUMENT PREVIEW / EXECUTIVE SUMMARY:\n{preview}"
