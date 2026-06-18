"""Groq LLM client: text chat + vision, with multi-key failover on rate limits."""
import base64

from groq import Groq

from . import config

# These exception classes exist in current Groq SDKs; fall back gracefully if not.
try:
    from groq import RateLimitError, APIStatusError
except ImportError:  # pragma: no cover
    RateLimitError = APIStatusError = Exception


class GroqClient:
    """Wraps the Groq SDK and rotates across multiple API keys on rate limits."""

    def __init__(self, api_keys=None):
        keys = api_keys if api_keys is not None else config.GROQ_API_KEYS
        if not keys:
            raise RuntimeError(
                "No Groq API keys found. Set GROQ_API_KEYS (comma-separated) in your .env."
            )
        self._clients = [Groq(api_key=k) for k in keys]
        self._idx = 0  # index of the key we currently prefer

    def _chat_raw(self, messages, model, temperature, max_tokens):
        last_err = None
        n = len(self._clients)
        for offset in range(n):
            i = (self._idx + offset) % n
            client = self._clients[i]
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._idx = i  # stick with whichever key just worked
                return resp.choices[0].message.content
            except RateLimitError as e:          # this key is throttled -> try next
                last_err = e
                continue
            except APIStatusError as e:
                last_err = e
                if getattr(e, "status_code", None) == 429:
                    continue
                raise
        raise RuntimeError(f"All Groq keys are rate-limited or failing: {last_err}")

    def chat(self, messages, model=None, temperature=0.2, max_tokens=1024):
        """Standard text chat completion. Returns the assistant message string."""
        return self._chat_raw(messages, model or config.TEXT_MODEL, temperature, max_tokens)

    def chat_stream(self, messages, model=None, temperature=0.2, max_tokens=1024):
        """Yield answer tokens as they arrive. Fails over between keys only *before* the
        first token (no mid-stream key switching)."""
        model = model or config.TEXT_MODEL
        last_err = None
        n = len(self._clients)
        for offset in range(n):
            i = (self._idx + offset) % n
            try:
                stream = self._clients[i].chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, stream=True,
                )
                self._idx = i
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return
            except RateLimitError as e:
                last_err = e
                continue
            except APIStatusError as e:
                last_err = e
                if getattr(e, "status_code", None) == 429:
                    continue
                raise
        raise RuntimeError(f"All Groq keys are rate-limited or failing: {last_err}")

    def describe_image(self, image_bytes, mime_type="image/jpeg", prompt=None):
        """OCR + describe an image using Groq's vision model (no Tesseract needed)."""
        prompt = prompt or (
            "Extract ALL text from this image verbatim. Then, on a new line starting "
            "with 'VISUAL:', briefly describe any non-text content (tables, charts, "
            "diagrams, photos). If there is no text, only describe the image."
        )
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                ],
            }
        ]
        return self._chat_raw(messages, config.VISION_MODEL, temperature=0.0, max_tokens=1024)

    def transcribe_audio(self, audio_bytes, filename="audio.mp3", model=None):
        """Transcribe an audio clip with Groq Whisper. Returns the transcript text."""
        model = model or config.AUDIO_MODEL
        last_err = None
        n = len(self._clients)
        for offset in range(n):
            i = (self._idx + offset) % n
            try:
                resp = self._clients[i].audio.transcriptions.create(
                    file=(filename, audio_bytes), model=model,
                )
                self._idx = i
                return resp.text
            except RateLimitError as e:
                last_err = e
                continue
            except APIStatusError as e:
                last_err = e
                if getattr(e, "status_code", None) == 429:
                    continue
                raise
        raise RuntimeError(f"All Groq keys are rate-limited or failing: {last_err}")
