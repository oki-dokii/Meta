"""
multimodal_pipeline.py
======================
Real multimodal analysis for ContentModerationEnv.

Replaces hand-crafted audio_transcript / visual_tags strings in scenario
state dicts with genuine ML-derived signals.

═══════════════════════════════════════════════════════════════
Part 1 — Audio pipeline  (AudioPipeline)
═══════════════════════════════════════════════════════════════
* Accept an audio file (mp3, wav, ogg, m4a) OR a URL.
* Transcribe using Groq Whisper (whisper-large-v3).
* Detect language with langdetect + Hinglish heuristic.
* Classify tone (calm / agitated / aggressive) via heuristics + LLM.

Tone-as-signal-booster contract
--------------------------------
  • Aggressive tone NEVER changes the content label on its own.
  • Transcript already flagged harmful → severity +=1 (capped at 5).
  • Aggressive tone + clean transcript → route_to_human=True, confidence ×0.65.
  • No context (no previous_flags, no platform_policy) → confidence ×0.75.

Return shape (AudioAnalysisResult.to_dict())
----------------------------------------------
{
    "transcript":        str,
    "detected_language": str,    # ISO-639-1 ("hi-Latn" = Hinglish)
    "regional_context":  bool,   # True for hi/ta/te/bn/Hinglish
    "tone_label":        str,    # "calm" | "agitated" | "aggressive"
    "tone_confidence":   float,
    "tone_reasoning":    str,
    "route_to_human":    bool,
    "moderation_input":  dict    # merge directly into env observation
}

═══════════════════════════════════════════════════════════════
Part 2 — Visual pipeline  (VisualPipeline)
═══════════════════════════════════════════════════════════════
* Accept an image file (jpg, png, webp, gif) or a video file (mp4, webm,
  mov) or HTTP(S) URLs to either.
* For images: send directly to a Groq vision-capable model for tag extraction.
* For videos: extract up to MAX_VIDEO_FRAMES evenly-spaced frames (via
  OpenCV if available, else Pillow GIF fallback), then analyse each frame
  and merge tags.
* Tags are returned as a flat list of lowercase strings — exactly the
  format expected by visual_tags in the env Observation model.
* A content_risk_flags list surfaces high-severity findings separately so
  downstream logic can boost severity without collapsing into a single label.

Return shape (VisualAnalysisResult.to_dict())
-----------------------------------------------
{
    "visual_tags":         list[str],  # env-ready; replaces hand-crafted tags
    "content_risk_flags":  list[str],  # high-severity findings (e.g. "weapon")
    "frame_count":         int,        # 1 for images, N for videos
    "model_used":          str,
    "moderation_input":    dict        # merge directly into env observation
}

Quick usage
-----------
    from multimodal_pipeline import AudioPipeline, VisualPipeline

    audio_pipe  = AudioPipeline()   # GROQ_API_KEY from env
    visual_pipe = VisualPipeline()  # same key

    state = env.reset(scenario_id="scen_hard_1")

    # Enrich audio
    state = audio_pipe.enrich_state(state, audio="clip.mp3")

    # Enrich visual
    state = visual_pipe.enrich_state(state, media="thumbnail.jpg")

    # state is now ready with real audio_transcript + visual_tags
    result = env.step(agent_action)
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ── optional import gates ────────────────────────────────────────────────────
try:
    from groq import Groq
except ImportError as _exc:
    raise ImportError(
        "groq>=0.9.0 is required for multimodal_pipeline.  "
        "Run:  pip install 'groq>=0.9.0'"
    ) from _exc

try:
    from langdetect import detect as _langdetect_detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    LangDetectException = Exception  # type: ignore[assignment,misc]


# ── Constants ─────────────────────────────────────────────────────────────────

WHISPER_MODEL = "whisper-large-v3"
TONE_LLM_MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".webm", ".mp4"}

# ── Regional / Hinglish language constants ────────────────────────────────────
# ISO-639-1 codes that trigger regional_context=True
_REGIONAL_LANGUAGE_CODES: frozenset[str] = frozenset({
    "hi",   # Hindi
    "ta",   # Tamil
    "te",   # Telugu
    "bn",   # Bengali
    "mr",   # Marathi (bonus coverage)
    "ur",   # Urdu (shares script overlap with Roman-Urdu/Hinglish)
})

# Common Hinglish indicator words written in Latin script.
# langdetect typically classifies Hinglish as "en" — this list catches it.
_HINGLISH_KEYWORDS: frozenset[str] = frozenset({
    "karo", "karna", "karoge", "karega", "karta", "karti",
    "hai", "hain", "tha", "thi", "the",
    "nahi", "nahin", "mat", "mujhe", "tumhe", "aapko",
    "bhai", "yaar", "dost", "behen", "bhaiya",
    "acha", "accha", "haan", "nahi",
    "kya", "kyon", "kyun", "kaise", "kahan", "kabhi",
    "bahut", "thoda", "zyada", "bilkul",
    "apna", "apni", "unka", "unki", "uska", "uski",
    "phir", "abhi", "wahan", "yahan",
    "dekho", "suno", "bolo", "bol",
    "gaya", "gayi", "aaya", "aayi",
    "sab", "kuch", "sirf", "toh", "aur", "lekin", "magar",
})
# Minimum Hinglish keyword hits to confidently classify as Hinglish
_HINGLISH_MIN_HITS = 2

# Heuristic word-lists for fast pre-LLM tone classification
_AGGRESSIVE_WORDS: frozenset[str] = frozenset({
    "kill", "murder", "die", "destroy", "attack", "threat", "bomb",
    "shoot", "stab", "hurt", "harm", "punish", "beat", "hate",
    "bastard", "screw you", "f**k", "fuck", "idiot", "moron",
    "worthless", "loser", "stupid", "piece of shit",
})
_AGITATED_WORDS: frozenset[str] = frozenset({
    "unbelievable", "ridiculous", "pathetic", "disgusting", "outrageous",
    "ridiculous", "I can't believe", "what the hell", "seriously",
    "how dare", "this is insane", "absolute", "completely wrong",
    "this is a lie", "liar",
})


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class AudioAnalysisResult:
    """Structured result from the audio pipeline."""
    transcript: str
    detected_language: str    # ISO-639-1 code (Whisper + langdetect refined)
    regional_context: bool    # True for hi/ta/te/bn/Hinglish content
    tone_label: str           # "calm" | "agitated" | "aggressive"
    tone_confidence: float    # 0.0 – 1.0
    tone_reasoning: str
    route_to_human: bool
    moderation_input: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Language detection (Step 3) ──────────────────────────────────────────────

def _is_hinglish(text: str) -> bool:
    """
    Heuristic Hinglish detector.

    langdetect often classifies Hinglish (Hindi written in Latin script) as
    English because the training corpus underrepresents code-switching text.
    We override it when >= _HINGLISH_MIN_HITS indicator words appear.
    """
    lower = text.lower()
    # Tokenise on whitespace and strip punctuation for clean word matching
    tokens = set(re.sub(r"[^\w\s]", " ", lower).split())
    hits = len(tokens & _HINGLISH_KEYWORDS)
    return hits >= _HINGLISH_MIN_HITS


def _detect_language(transcript: str, whisper_language: str) -> tuple[str, bool]:
    """
    Refine language detection using langdetect + the Hinglish heuristic.

    Priority order
    --------------
    1. If Whisper already returned a regional language code → trust it.
    2. If _is_hinglish() fires → override to "hi-Latn" (Hinglish tag).
    3. If langdetect is available → use it to verify/correct Whisper's "en".
    4. Fall back to whisper_language.

    Returns
    -------
    (detected_language, regional_context)
      detected_language : ISO-639-1 code (or "hi-Latn" for Hinglish)
      regional_context  : True if the content is in a South Asian language
    """
    # Step 1: Whisper already detected a regional language — trust it
    if whisper_language in _REGIONAL_LANGUAGE_CODES:
        return whisper_language, True

    # Step 2: Hinglish heuristic — fires even when Whisper says "en"
    if _is_hinglish(transcript):
        return "hi-Latn", True  # "hi-Latn" = Hindi written in Latin script

    # Step 3: langdetect for any remaining uncertainty
    if _LANGDETECT_AVAILABLE and transcript.strip():
        try:
            lang = _langdetect_detect(transcript)
            if lang in _REGIONAL_LANGUAGE_CODES:
                return lang, True
            return lang, False
        except LangDetectException:
            pass  # fall through to whisper value

    # Step 4: fallback — use what Whisper gave us
    return whisper_language, False


# ── Heuristic tone pre-classifier ────────────────────────────────────────────

def _heuristic_tone(text: str) -> tuple[str, float]:
    """
    Fast, zero-LLM pre-classification of tone from transcript text.

    Returns (tone_label, confidence) where confidence reflects certainty
    of the heuristic pass alone — the LLM refiner caps final confidence
    if the context is absent.
    """
    lower = text.lower()
    # all-caps word ratio is an agitation signal
    words = text.split()
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)

    agg_hits = sum(1 for w in _AGGRESSIVE_WORDS if w in lower)
    agi_hits = sum(1 for w in _AGITATED_WORDS if w in lower)

    if agg_hits >= 2 or (agg_hits >= 1 and caps_ratio > 0.3):
        return "aggressive", min(0.65 + 0.05 * agg_hits, 0.80)
    if agg_hits == 1 or agi_hits >= 2 or caps_ratio > 0.4:
        return "agitated", 0.55
    if agi_hits == 1 or caps_ratio > 0.25:
        return "agitated", 0.45
    return "calm", 0.70


# ── LLM tone refiner ─────────────────────────────────────────────────────────

_TONE_SYSTEM = """\
You are a tone-analysis assistant for a content moderation system.
You will be given a short transcript and a heuristic pre-classification.
Your task is to refine the tone label and provide a brief reasoning.

Rules:
- Tone labels: "calm", "agitated", "aggressive"
- Aggressive tone NEVER changes the content harmful/safe classification by itself.
- Be conservative: only label as "aggressive" if the speaker uses direct threats
  or extreme dehumanising language.
- Respond ONLY with JSON, no extra text.

Response schema:
{
  "tone_label":      "calm" | "agitated" | "aggressive",
  "tone_confidence": <float 0.0-1.0>,
  "tone_reasoning":  "<one concise sentence>"
}"""


def _llm_tone_refine(
    client: Groq,
    transcript: str,
    heuristic_label: str,
    heuristic_confidence: float,
) -> dict:
    """
    Call the Groq chat LLM to refine tone classification.
    Falls back to heuristic values on any error.
    """
    user_msg = (
        f"Transcript:\n\"\"\"\n{transcript[:2000]}\n\"\"\"\n\n"
        f"Heuristic pre-classification: {heuristic_label} "
        f"(confidence {heuristic_confidence:.2f})\n\n"
        f"Refine the tone label and provide reasoning."
    )
    try:
        response = client.chat.completions.create(
            model=TONE_LLM_MODEL,
            messages=[
                {"role": "system", "content": _TONE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)
        tone_label = str(parsed.get("tone_label", heuristic_label)).strip().lower()
        if tone_label not in {"calm", "agitated", "aggressive"}:
            tone_label = heuristic_label
        confidence = float(parsed.get("tone_confidence", heuristic_confidence))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(parsed.get("tone_reasoning", ""))
        return {
            "tone_label": tone_label,
            "tone_confidence": confidence,
            "tone_reasoning": reasoning,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "tone_label": heuristic_label,
            "tone_confidence": heuristic_confidence,
            "tone_reasoning": f"(LLM refinement failed: {exc}; using heuristic)",
        }


# ── Severity booster logic ────────────────────────────────────────────────────

def _apply_tone_boost(
    tone_label: str,
    tone_confidence: float,
    transcript_is_harmful: bool,
    current_severity: Optional[int],
    has_context: bool,
) -> tuple[float, bool, int | None]:
    """
    Apply the tone-as-signal-booster policy.

    Parameters
    ----------
    tone_label           : classified tone
    tone_confidence      : raw classifier confidence
    transcript_is_harmful: whether Whisper transcript was already flagged
    current_severity     : severity from scenario (1–5) or None
    has_context          : True if previous_flags or platform_policy present

    Returns
    -------
    adjusted_confidence : float   — final confidence after context adjustments
    route_to_human      : bool
    adjusted_severity   : int | None
    """
    route_to_human = False
    adjusted_severity = current_severity

    if tone_label == "aggressive":
        if transcript_is_harmful:
            # Boost severity by +1 (capped at 5)
            if adjusted_severity is not None:
                adjusted_severity = min(5, adjusted_severity + 1)
            # No confidence reduction — aggressive + harmful transcript is clear
        else:
            # Aggressive tone but clean transcript → send to human
            route_to_human = True
            tone_confidence = tone_confidence * 0.65  # reduce confidence

        if not has_context:
            # No context → reduce confidence further, never auto-remove
            tone_confidence = tone_confidence * 0.75
            route_to_human = True

    elif tone_label == "agitated":
        if not has_context:
            tone_confidence = tone_confidence * 0.85

    # Always clamp
    adjusted_confidence = max(0.0, min(1.0, tone_confidence))
    return adjusted_confidence, route_to_human, adjusted_severity


# ── File / URL loading helpers ────────────────────────────────────────────────

def _load_audio_bytes(source: str) -> tuple[bytes, str]:
    """
    Load raw bytes from a local path or HTTP(S) URL.

    Returns (bytes, filename) where filename is used for MIME sniffing.
    """
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        # URL: download to a temp buffer
        with urllib.request.urlopen(source, timeout=30) as resp:
            audio_bytes = resp.read()
        # Infer filename from URL path or default to .ogg
        url_path = parsed.path
        filename = Path(url_path).name or "audio.ogg"
        if not Path(filename).suffix:
            filename += ".ogg"
        return audio_bytes, filename
    else:
        # Local file
        local_path = Path(source)
        if not local_path.exists():
            raise FileNotFoundError(f"Audio file not found: {source!r}")
        ext = local_path.suffix.lower()
        if ext not in SUPPORTED_AUDIO_EXTS:
            raise ValueError(
                f"Unsupported audio format {ext!r}. "
                f"Supported: {sorted(SUPPORTED_AUDIO_EXTS)}"
            )
        return local_path.read_bytes(), local_path.name


# ── Main pipeline class ───────────────────────────────────────────────────────

class AudioPipeline:
    """
    Real audio transcription + tone analysis pipeline.

    Wraps Groq's Whisper (whisper-large-v3) for transcription and an
    LLM call for tone refinement, then applies the tone-as-signal-booster
    policy defined in the module docstring.

    Parameters
    ----------
    groq_api_key : str | None
        Groq API key.  Falls back to the GROQ_API_KEY env variable.
    tone_model : str | None
        Override the chat model used for tone refinement.
        Defaults to MODEL_NAME env var or "llama-3.3-70b-versatile".
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        tone_model: Optional[str] = None,
    ):
        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "No Groq API key found.  Set GROQ_API_KEY environment variable "
                "or pass groq_api_key= to AudioPipeline()."
            )
        self._client = Groq(api_key=api_key)
        self._tone_model = tone_model or TONE_LLM_MODEL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_audio(
        self,
        source: str,
        *,
        previous_flags: int = 0,
        platform_policy: Optional[str] = None,
        current_severity: Optional[int] = None,
        is_flagged_harmful: Optional[bool] = None,
    ) -> AudioAnalysisResult:
        """
        Transcribe audio and classify tone.

        Parameters
        ----------
        source           : local file path (mp3/wav/ogg/m4a) or HTTP(S) URL
        previous_flags   : account's prior violation count — used for context detection
        platform_policy  : "strict" | "moderate" | "lenient" | None
        current_severity : existing severity from scenario (1-5) or None
        is_flagged_harmful: True/False if caller already knows; None = auto-detect
                           from aggressive/explicit keywords in transcript

        Returns
        -------
        AudioAnalysisResult (also has .to_dict() for JSON serialisation)
        """
        # 1. Load bytes
        audio_bytes, filename = _load_audio_bytes(source)

        # 2. Whisper transcription via Groq
        transcript, whisper_language = self._transcribe(audio_bytes, filename)

        # 3. Language detection + Hinglish check
        #    Refines Whisper's language code using langdetect and our Hinglish
        #    heuristic.  regional_context=True signals South Asian content.
        detected_language, regional_context = _detect_language(transcript, whisper_language)

        # 4. Heuristic pre-classify tone
        heuristic_label, heuristic_conf = _heuristic_tone(transcript)

        # 5. LLM refine
        refined = _llm_tone_refine(
            self._client, transcript, heuristic_label, heuristic_conf
        )
        tone_label = refined["tone_label"]
        tone_confidence = refined["tone_confidence"]
        tone_reasoning = refined["tone_reasoning"]

        # 6. Determine if transcript is harmful (auto-detect if not provided)
        if is_flagged_harmful is None:
            is_flagged_harmful = self._detect_harmful(transcript)

        # 7. Context flag
        has_context = bool(previous_flags > 0 or platform_policy)

        # 8. Tone-as-signal-booster policy
        final_confidence, route_to_human, adjusted_severity = _apply_tone_boost(
            tone_label=tone_label,
            tone_confidence=tone_confidence,
            transcript_is_harmful=is_flagged_harmful,
            current_severity=current_severity,
            has_context=has_context,
        )

        # 9. Build moderation_input dict — ready to merge into env observation
        moderation_input: dict = {
            "audio_transcript": transcript,
            "detected_language": detected_language,
            "regional_context": regional_context,
            "tone_label": tone_label,
            "tone_confidence": round(final_confidence, 4),
            "route_to_human": route_to_human,
        }
        if adjusted_severity is not None:
            moderation_input["suggested_severity"] = adjusted_severity

        return AudioAnalysisResult(
            transcript=transcript,
            detected_language=detected_language,
            regional_context=regional_context,
            tone_label=tone_label,
            tone_confidence=round(final_confidence, 4),
            tone_reasoning=tone_reasoning,
            route_to_human=route_to_human,
            moderation_input=moderation_input,
        )

    def enrich_state(
        self,
        state: dict,
        audio: str,
    ) -> dict:
        """
        Enrich an existing env observation dict with real audio analysis.

        Reads previous_flags and platform_policy from the state dict to
        properly calibrate the tone boost policy.  Returns a shallow-copy
        of state with audio_transcript, tone_label, tone_confidence, and
        route_to_human merged in.

        Parameters
        ----------
        state : env observation dict from env.reset() / env.state()
        audio : local path or URL to the audio file

        Returns
        -------
        enriched_state : dict (new object — original is not mutated)
        """
        result = self.analyse_audio(
            source=audio,
            previous_flags=state.get("previous_flags", 0),
            platform_policy=state.get("platform_policy"),
            current_severity=state.get("severity"),
        )
        enriched = dict(state)  # shallow copy
        enriched.update(result.moderation_input)
        return enriched

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transcribe(self, audio_bytes: bytes, filename: str) -> tuple[str, str]:
        """
        Call Groq Whisper and return (transcript, language).

        Groq's audio API accepts a file-like object with a .name attribute.
        We create a named in-memory buffer so no temp file is written to disk.
        """
        # Groq SDK accepts (filename, bytes, content_type) tuples
        ext = Path(filename).suffix.lower().lstrip(".")
        content_type_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "m4a": "audio/mp4",
            "flac": "audio/flac",
            "webm": "audio/webm",
            "mp4": "video/mp4",
        }
        content_type = content_type_map.get(ext, "audio/mpeg")

        transcription = self._client.audio.transcriptions.create(
            file=(filename, audio_bytes, content_type),
            model=WHISPER_MODEL,
            response_format="verbose_json",
            temperature=0.0,
        )

        # verbose_json returns a Transcription object with .text and .language
        transcript: str = (getattr(transcription, "text", "") or "").strip()
        language: str = (getattr(transcription, "language", "") or "unknown").strip()
        return transcript, language

    def _detect_harmful(self, transcript: str) -> bool:
        """
        Fast heuristic: is the transcript content itself likely harmful?

        Used when the caller does not supply is_flagged_harmful.
        Returns True if 2+ aggressive keywords appear in the transcript —
        a conservative threshold to avoid false-positives.
        """
        lower = transcript.lower()
        hits = sum(1 for w in _AGGRESSIVE_WORDS if w in lower)
        return hits >= 2


# ── Convenience function ──────────────────────────────────────────────────────

def analyse_audio(
    source: str,
    *,
    groq_api_key: Optional[str] = None,
    previous_flags: int = 0,
    platform_policy: Optional[str] = None,
    current_severity: Optional[int] = None,
) -> dict:
    """
    Module-level convenience wrapper.

    Creates a one-shot AudioPipeline and returns the result as a plain dict.

    Example
    -------
        from multimodal_pipeline import analyse_audio

        result = analyse_audio("clip.mp3", previous_flags=3, platform_policy="strict")
        print(result["transcript"])
        print(result["tone_label"])          # "calm" | "agitated" | "aggressive"
        print(result["moderation_input"])    # merge into env observation
    """
    pipe = AudioPipeline(groq_api_key=groq_api_key)
    return pipe.analyse_audio(
        source=source,
        previous_flags=previous_flags,
        platform_policy=platform_policy,
        current_severity=current_severity,
    ).to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Image pipeline  (ImagePipeline / VisualPipeline alias)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Visual constants ──────────────────────────────────────────────────────────

# LLaMA 3.2 Vision on Groq
VISUAL_LLM_MODEL = os.getenv("VISUAL_MODEL_NAME", "llama-3.2-11b-vision-preview")

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

# The 7 harm categories the model is specifically asked to check
_HARM_CATEGORIES: list[str] = [
    "weapons",
    "hate_symbols",
    "extremist_imagery",
    "graphic_violence",
    "nudity",
    "drug_paraphernalia",
    "self_harm_imagery",
]

# Keyword sets for heuristic harm matching (fallback when LLM confidence missing)
_HARM_KEYWORDS: dict[str, frozenset] = {
    "weapons":            frozenset({"weapon", "gun", "rifle", "knife", "bomb", "grenade", "firearm", "pistol"}),
    "hate_symbols":       frozenset({"swastika", "hate-symbol", "kkk", "nazi", "white-supremacy", "noose"}),
    "extremist_imagery":  frozenset({"isis", "terror", "extremist", "jihad", "propaganda"}),
    "graphic_violence":   frozenset({"blood", "gore", "decapitation", "mutilation", "graphic-violence", "corpse"}),
    "nudity":             frozenset({"nudity", "explicit", "pornographic", "genitalia", "nsfw"}),
    "drug_paraphernalia": frozenset({"drug", "syringe", "cocaine", "heroin", "bong", "pipe", "powder"}),
    "self_harm_imagery":  frozenset({"self-harm", "cutting", "suicide", "self-injury", "razor", "wrist"}),
}


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — Symbol database (loaded from symbol_database.json at runtime)
# ══════════════════════════════════════════════════════════════════════════════

_SYMBOL_DB_PATH = Path(__file__).with_name("symbol_database.json")


def _load_symbol_database(path: Optional[Path] = None) -> dict:
    """
    Load the symbol database from a JSON file.

    Falls back to an empty dict if the file is missing or malformed so the
    rest of the pipeline always runs.  Override the path with the
    SYMBOL_DB_PATH environment variable for testing.

    Returns a dict keyed by category name, each value a list of entry dicts.
    """
    db_path = Path(os.getenv("SYMBOL_DB_PATH", "") or (path or _SYMBOL_DB_PATH))
    try:
        with open(db_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return raw.get("categories", {})
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, KeyError) as exc:
        import warnings
        warnings.warn(
            f"symbol_database.json could not be parsed: {exc}.  Symbol detection disabled.",
            RuntimeWarning, stacklevel=2,
        )
        return {}


# Load at import time; can be reloaded at runtime via _reload_symbol_db()
_SYMBOL_DB: dict = _load_symbol_database()


def _reload_symbol_db(path: Optional[Path] = None) -> dict:
    """Reload the symbol database from disk.  Useful after editing the JSON."""
    global _SYMBOL_DB
    _SYMBOL_DB = _load_symbol_database(path)
    return _SYMBOL_DB


def _flat_symbol_entries(db: Optional[dict] = None) -> list[dict]:
    """Return every entry from every category as a flat list."""
    db = db if db is not None else _SYMBOL_DB
    entries: list[dict] = []
    for cat_entries in db.values():
        if isinstance(cat_entries, list):
            entries.extend(cat_entries)
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Context analyser (fuzzy keyword cross-reference)
# ══════════════════════════════════════════════════════════════════════════════

_FUZZY_MATCH_RATIO = float(os.getenv("SYMBOL_FUZZY_RATIO", "0.75"))


def _fuzzy_score(needle: str, haystack: str) -> float:
    """
    Return a 0.0–1.0 similarity score between needle and a text haystack.

    Uses difflib.SequenceMatcher (stdlib, no extra deps).
    Also checks for direct substring containment (score=1.0).
    """
    from difflib import SequenceMatcher
    needle_l = needle.lower()
    hay_l    = haystack.lower()
    if needle_l in hay_l:
        return 1.0
    # Check word-boundary partial match on each word in the haystack
    words = re.split(r"[\s,;.!?\-/]+", hay_l)
    for word in words:
        if not word:
            continue
        ratio = SequenceMatcher(None, needle_l, word).ratio()
        if ratio >= _FUZZY_MATCH_RATIO:
            return ratio
    # Whole-string similarity (catches multi-word queries)
    return SequenceMatcher(None, needle_l, hay_l).ratio()


def _CONTEXT_ANALYZER(
    image_description: str,
    db: Optional[dict] = None,
    min_keyword_score: float = _FUZZY_MATCH_RATIO,
) -> list[dict]:
    """
    Cross-reference an image description (from the vision model) against the
    symbol database using fuzzy keyword matching.

    Parameters
    ----------
    image_description : str
        Concatenated text the vision model produced (visual_tags + evidence
        strings).  Can be the raw JSON string or a pre-joined version.
    db : dict | None
        Symbol database dict (category → list of entries).  Defaults to
        the module-level _SYMBOL_DB loaded from symbol_database.json.
    min_keyword_score : float
        Minimum fuzzy score (0.0–1.0) for a keyword to count as a match.

    Returns
    -------
    list of match dicts:
      {
        "symbol_id":          str,
        "symbol_name":        str,
        "category":           str,
        "matched_keyword":    str,
        "match_score":        float,        # 0.0–1.0
        "confidence_required": float,       # from DB entry
        "route_to_human":     bool,         # True if below route_to_human_below
        "auto_flag":          bool,         # True if match_score >= confidence_required
        "severity":           int
      }
    """
    db = db if db is not None else _SYMBOL_DB
    matches: list[dict] = []

    for entry in _flat_symbol_entries(db):
        all_keywords = list(entry.get("keywords", [])) + list(entry.get("aliases", []))
        best_score = 0.0
        best_kw = ""
        for kw in all_keywords:
            score = _fuzzy_score(kw, image_description)
            if score > best_score:
                best_score = score
                best_kw = kw

        if best_score < min_keyword_score:
            continue  # no meaningful match

        conf_req  = float(entry.get("confidence_required", 0.80))
        rth_below = float(entry.get("route_to_human_below",  conf_req))
        auto_flag = best_score >= conf_req
        route_to_human = (best_score < rth_below) or (not auto_flag and best_score >= min_keyword_score)

        matches.append({
            "symbol_id":           entry.get("id", ""),
            "symbol_name":         entry.get("name", ""),
            "category":            entry.get("category", ""),
            "matched_keyword":     best_kw,
            "match_score":         round(best_score, 4),
            "confidence_required": conf_req,
            "route_to_human":      route_to_human,
            "auto_flag":           auto_flag,
            "severity":            int(entry.get("severity", 3)),
        })

    # Sort by severity desc, then match_score desc
    matches.sort(key=lambda m: (-m["severity"], -m["match_score"]))
    return matches


# ══════════════════════════════════════════════════════════════════════════════
# Vision prompt builder — injects symbol DB context dynamically
# ══════════════════════════════════════════════════════════════════════════════

def _build_image_system_prompt(db: Optional[dict] = None) -> str:
    """
    Build the vision model system prompt, injecting the top symbol entries
    from the symbol database so the model knows specifically what to look for.

    The symbol reference list is generated from the live database, so adding
    entries to symbol_database.json automatically enriches the prompt without
    changing any Python code.
    """
    db = db if db is not None else _SYMBOL_DB
    entries = _flat_symbol_entries(db)

    # Build a compact reference block (up to 30 entries to stay within token budget)
    # Priority: highest severity first, then lowest confidence_required (most sensitive)
    sorted_entries = sorted(
        entries,
        key=lambda e: (-e.get("severity", 3), e.get("confidence_required", 0.8)),
    )[:30]

    symbol_lines: list[str] = []
    for e in sorted_entries:
        kws = ", ".join(e.get("keywords", [])[:4])  # top 4 keywords only
        cat = e.get("category", "?").replace("_", " ")
        cr  = e.get("confidence_required", 0.8)
        symbol_lines.append(
            f"  - [{cat}] {e.get('name', '')} → look for: {kws} "
            f"(auto-flag threshold: {cr:.0%})"
        )

    symbol_ref = "\n".join(symbol_lines) if symbol_lines else "  (no entries loaded)"

    return f"""\
You are a visual content safety analyst for a content moderation platform.
Analyse the image thoroughly and classify what you see.

You MUST check specifically for these 7 harm categories:
  1. weapons           — firearms, knives, bombs, any weapon
  2. hate_symbols      — swastikas, KKK imagery, white-supremacist symbols, nooses
  3. extremist_imagery — terrorist flags, propaganda, extremist group insignia
  4. graphic_violence  — blood, gore, dead bodies, severe physical injury
  5. nudity            — explicit sexual content, exposed genitalia
  6. drug_paraphernalia — syringes, pipes, drug powders, paraphernalia
  7. self_harm_imagery — self-cutting, self-injury imagery, suicide depictions

KNOWN SYMBOLS REFERENCE (from live database — look for these specifically):
{symbol_ref}

For EACH symbol from the reference list you detect, mention it explicitly in
your visual_tags list (e.g. "swastika", "88-numeric-code", "roman-salute").

Rules:
- Respond ONLY with valid JSON. No markdown, no explanation.
- All tag strings must be lowercase.
- confidence values: 0.0 (not present) to 1.0 (clearly present).
- For ambiguous symbols (e.g. ok-sign, celtic-cross): set confidence low and
  include an evidence note explaining the context.

Response schema (fill ALL fields):
{{
  "visual_tags": ["tag1", "tag2", ...],
  "image_description": "<one paragraph describing the full image in detail>",
  "harm_assessment": {{
    "weapons":            {{"detected": true|false, "confidence": 0.0, "evidence": ""}},
    "hate_symbols":       {{"detected": true|false, "confidence": 0.0, "evidence": ""}},
    "extremist_imagery":  {{"detected": true|false, "confidence": 0.0, "evidence": ""}},
    "graphic_violence":   {{"detected": true|false, "confidence": 0.0, "evidence": ""}},
    "nudity":             {{"detected": true|false, "confidence": 0.0, "evidence": ""}},
    "drug_paraphernalia": {{"detected": true|false, "confidence": 0.0, "evidence": ""}},
    "self_harm_imagery":  {{"detected": true|false, "confidence": 0.0, "evidence": ""}}
  }}
}}"""


@dataclass
class ImageAnalysisResult:
    """Structured result from the image pipeline."""
    visual_tags: list           # flat list of all detected lowercase tags
    harmful_tags_detected: list # tags that belong to one of the 7 harm categories
    confidence_per_tag: dict    # {tag: confidence_float}
    model_used: str
    moderation_input: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

# Backward-compat alias
VisualAnalysisResult = ImageAnalysisResult


# ── Vision prompt ─────────────────────────────────────────────────────────────

_IMAGE_SYSTEM = """\
You are a visual content safety analyst for a content moderation platform.
Analyse the image thoroughly and classify what you see.

You MUST check specifically for these 7 harm categories:
  1. weapons           — firearms, knives, bombs, any weapon
  2. hate_symbols      — swastikas, KKK imagery, white-supremacist symbols, nooses
  3. extremist_imagery — terrorist flags, propaganda, extremist group insignia
  4. graphic_violence  — blood, gore, dead bodies, severe physical injury
  5. nudity            — explicit sexual content, exposed genitalia
  6. drug_paraphernalia — syringes, pipes, drug powders, paraphernalia
  7. self_harm_imagery — self-cutting, self-injury imagery, suicide depictions

Rules:
- Respond ONLY with valid JSON. No markdown, no explanation.
- All tag strings must be lowercase.
- confidence values: 0.0 (not present) to 1.0 (clearly present).

Response schema (fill ALL fields):
{
  "visual_tags": ["tag1", "tag2", ...],
  "harm_assessment": {
    "weapons":            {"detected": true|false, "confidence": 0.0, "evidence": ""},
    "hate_symbols":       {"detected": true|false, "confidence": 0.0, "evidence": ""},
    "extremist_imagery":  {"detected": true|false, "confidence": 0.0, "evidence": ""},
    "graphic_violence":   {"detected": true|false, "confidence": 0.0, "evidence": ""},
    "nudity":             {"detected": true|false, "confidence": 0.0, "evidence": ""},
    "drug_paraphernalia": {"detected": true|false, "confidence": 0.0, "evidence": ""},
    "self_harm_imagery":  {"detected": true|false, "confidence": 0.0, "evidence": ""}
  }
}"""


# ── Image loading helpers ─────────────────────────────────────────────────────

def _load_image_bytes(source: str) -> tuple[bytes, str]:
    """
    Load image bytes from a local path or HTTP(S) URL.
    Returns (bytes, filename).
    """
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        with urllib.request.urlopen(source, timeout=30) as resp:
            raw = resp.read()
        filename = Path(parsed.path).name or "image.jpg"
        if not Path(filename).suffix:
            filename += ".jpg"
        return raw, filename
    local = Path(source)
    if not local.exists():
        raise FileNotFoundError(f"Image file not found: {source!r}")
    ext = local.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTS:
        raise ValueError(
            f"Unsupported image format {ext!r}. Supported: {sorted(SUPPORTED_IMAGE_EXTS)}"
        )
    return local.read_bytes(), local.name


def _image_bytes_to_base64(raw_bytes: bytes, filename: str) -> str:
    """
    Convert raw image bytes to a base64 JPEG string.
    Non-JPEG formats converted via Pillow; falls back to raw encoding.
    """
    import base64
    ext = Path(filename).suffix.lower()
    if ext not in (".jpg", ".jpeg"):
        try:
            from PIL import Image as _PIL
            buf_in = io.BytesIO(raw_bytes)
            img = _PIL.open(buf_in).convert("RGB")
            buf_out = io.BytesIO()
            img.save(buf_out, format="JPEG", quality=85)
            return base64.b64encode(buf_out.getvalue()).decode()
        except ImportError:
            pass
    return base64.b64encode(raw_bytes).decode()


def _parse_image_response(
    raw_text: str,
) -> tuple[list[str], list[str], dict[str, float]]:
    """
    Parse LLM vision response into (visual_tags, harmful_tags, confidence_per_tag).
    Handles missing/malformed responses gracefully.
    """
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
    raw_text = re.sub(r"\n?```$", "", raw_text)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return [], [], {}

    # --- visual_tags ---
    visual_tags = [
        str(t).strip().lower()
        for t in parsed.get("visual_tags", [])
        if 1 < len(str(t)) <= 60
    ]

    # --- harm_assessment ---
    harm = parsed.get("harm_assessment", {})
    harmful_tags: list[str] = []
    confidence_per_tag: dict[str, float] = {}

    for category in _HARM_CATEGORIES:
        cat_data = harm.get(category, {})
        detected = bool(cat_data.get("detected", False))
        confidence = float(cat_data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        confidence_per_tag[category] = confidence
        if detected and confidence >= 0.3:
            harmful_tags.append(category)
            # Also add the category as a visual tag if not already
            if category.replace("_", "-") not in visual_tags:
                visual_tags.append(category.replace("_", "-"))

    # Add confidence for all visual tags (1.0 default — model listed them)
    for tag in visual_tags:
        if tag not in confidence_per_tag:
            confidence_per_tag[tag] = 1.0

    return visual_tags, harmful_tags, confidence_per_tag


# ── Main image pipeline class ─────────────────────────────────────────────────

class ImagePipeline:
    """
    Image content analysis pipeline using LLaMA 3.2 Vision via Groq.

    Checks specifically for 7 harm categories:
      weapons, hate_symbols, extremist_imagery, graphic_violence,
      nudity, drug_paraphernalia, self_harm_imagery

    Parameters
    ----------
    groq_api_key : str | None   Falls back to GROQ_API_KEY env var.
    visual_model : str | None   Defaults to llama-3.2-11b-vision-preview.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        visual_model: Optional[str] = None,
    ):
        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "No Groq API key found.  Set GROQ_API_KEY or pass groq_api_key=."
            )
        self._client = Groq(api_key=api_key)
        self._model = visual_model or VISUAL_LLM_MODEL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_image(
        self,
        source: str,
        *,
        platform_policy: Optional[str] = None,
        previous_flags: int = 0,
    ) -> ImageAnalysisResult:
        """
        Analyse an image file or URL for harmful content.

        Parameters
        ----------
        source          : local file path (jpg/png/webp/gif/bmp) or HTTP(S) URL
        platform_policy : \"strict\" | \"moderate\" | \"lenient\" | None
        previous_flags  : account prior violation count

        Returns
        -------
        ImageAnalysisResult
          .visual_tags         — all detected content tags
          .harmful_tags_detected — tags from the 7 harm categories
          .confidence_per_tag  — {tag: float 0.0–1.0}
          .moderation_input    — dict ready to merge into env observation
        """
        raw_bytes, filename = _load_image_bytes(source)
        b64 = _image_bytes_to_base64(raw_bytes, filename)
        visual_tags, harmful_tags, confidence_per_tag = self._call_vision(b64)

        moderation_input: dict = {
            "visual_tags": visual_tags,
            "harmful_tags_detected": harmful_tags,
            "confidence_per_tag": confidence_per_tag,
        }

        return ImageAnalysisResult(
            visual_tags=visual_tags,
            harmful_tags_detected=harmful_tags,
            confidence_per_tag=confidence_per_tag,
            model_used=self._model,
            moderation_input=moderation_input,
        )

    def analyse_image_b64(self, b64: str) -> tuple[list, list, dict]:
        """
        Analyse a pre-encoded base64 JPEG frame.
        Returns (visual_tags, harmful_tags, confidence_per_tag).
        Used internally by VideoPipeline.
        """
        return self._call_vision(b64)

    def enrich_state(self, state: dict, image: str) -> dict:
        """
        Enrich an env observation dict with real visual tags.
        Returns a shallow-copy of state with visual_tags merged in.
        """
        result = self.analyse_image(
            source=image,
            platform_policy=state.get("platform_policy"),
            previous_flags=state.get("previous_flags", 0),
        )
        enriched = dict(state)
        enriched.update(result.moderation_input)
        return enriched

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_vision(self, image_b64: str) -> tuple[list, list, dict]:
        """
        Call LLaMA 3.2 Vision, parse structured response, then run Layer 2
        _CONTEXT_ANALYZER on the image description for fuzzy symbol matching.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _build_image_system_prompt()},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Analyse this image for all 7 harm categories AND all "
                                    "known symbols from the reference list. "
                                    "Return the full JSON response including image_description."
                                ),
                            },
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=800,
            )
            raw = (response.choices[0].message.content or "").strip()
            visual_tags, harmful_tags, confidence_per_tag = _parse_image_response(raw)

            # ── Layer 2: run _CONTEXT_ANALYZER on the image description ──────
            # Extract the image_description field from the raw response for
            # richer fuzzy matching (includes evidence notes from the model).
            description_text = ""
            try:
                parsed_raw = json.loads(
                    re.sub(r"^```[a-z]*\n?", "", raw.strip()).rstrip("```")
                )
                description_text = parsed_raw.get("image_description", "")
            except Exception:
                pass
            # Also include visual_tags in matching text so single-word tags hit
            combined_text = " ".join(visual_tags) + " " + description_text

            ctx_matches = _CONTEXT_ANALYZER(combined_text)
            for match in ctx_matches:
                sid = match["symbol_id"].replace("_", "-")
                # Add to visual_tags
                if sid not in visual_tags:
                    visual_tags.append(sid)
                # If auto_flag, treat as a harmful tag (maps to hate_symbols or
                # extremist_imagery depending on category)
                if match["auto_flag"]:
                    cat = match["category"]
                    harm_cat = (
                        "extremist_imagery"
                        if cat in ("extremist_flags", "regional_india")
                        else "hate_symbols"
                    )
                    if harm_cat not in harmful_tags:
                        harmful_tags.append(harm_cat)
                    # Store per-symbol confidence
                    confidence_per_tag[sid] = match["match_score"]
                    confidence_per_tag[harm_cat] = max(
                        confidence_per_tag.get(harm_cat, 0.0),
                        match["match_score"],
                    )
                # If route_to_human, flag it
                if match["route_to_human"]:
                    confidence_per_tag[f"{sid}:route_to_human"] = 1.0

            return visual_tags, harmful_tags, confidence_per_tag

        except Exception:  # noqa: BLE001
            return [], [], {}


# Backward-compat alias — old code that imports VisualPipeline still works
class VisualPipeline(ImagePipeline):
    """Alias for ImagePipeline (backward compatibility)."""

    def analyse_media(self, source: str, **kwargs) -> ImageAnalysisResult:
        """Thin shim: routes image files through analyse_image()."""
        return self.analyse_image(source, **kwargs)


# ── Convenience function (image) ──────────────────────────────────────────────

def analyse_visual(
    source: str,
    *,
    groq_api_key: Optional[str] = None,
    platform_policy: Optional[str] = None,
    previous_flags: int = 0,
) -> dict:
    """
    Module-level convenience wrapper for image analysis.

    Example
    -------
        from multimodal_pipeline import analyse_visual

        result = analyse_visual("post_thumbnail.jpg")
        print(result["visual_tags"])            # ["crowd", "protest", "banner"]
        print(result["harmful_tags_detected"])  # ["weapons"] if detected
        print(result["confidence_per_tag"])     # {"weapons": 0.95, ...}
        print(result["moderation_input"])       # merge into env observation
    """
    pipe = ImagePipeline(groq_api_key=groq_api_key)
    return pipe.analyse_image(
        source=source,
        platform_policy=platform_policy,
        previous_flags=previous_flags,
    ).to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — Video pipeline  (VideoPipeline)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Video constants ───────────────────────────────────────────────────────────

KEYFRAME_INTERVAL_S = float(os.getenv("KEYFRAME_INTERVAL_S", "3"))  # 1 frame per N seconds
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
YTDLP_BIN = os.getenv("YTDLP_BIN", "yt-dlp")


# ── Video data models ─────────────────────────────────────────────────────────

@dataclass
class FrameAnalysis:
    """Visual analysis result for a single keyframe."""
    timestamp_s: float          # seconds from start
    visual_tags: list[str]
    harmful_tags: list[str]
    confidence_per_tag: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoAnalysisResult:
    """
    Full result from the video pipeline.

    Returned by VideoPipeline.analyse_video().
    """
    # Audio fields (from AudioPipeline)
    transcript: str
    detected_language: str
    regional_context: bool
    tone_label: str
    tone_confidence: float
    route_to_human: bool

    # Visual fields (merged across all keyframes)
    visual_tags: list[str]              # deduplicated union of all frame tags
    harmful_detected_at: list[dict]     # [{timestamp_s, tag}] for harm hits

    # Per-frame breakdown
    frame_analysis: list[dict]          # list of FrameAnalysis.to_dict()

    # Env-ready merge dict
    moderation_input: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── ffmpeg / yt-dlp helpers ───────────────────────────────────────────────────

def _check_binary(name: str, bin_path: str) -> bool:
    """Return True if `bin_path` is executable on PATH."""
    import shutil
    return shutil.which(bin_path) is not None


def _get_video_duration(video_path: str) -> float:
    """
    Use ffprobe to get video duration in seconds.
    Falls back to 0.0 if ffprobe unavailable or fails.
    """
    import subprocess
    try:
        result = subprocess.run(
            [
                FFPROBE_BIN, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "format=duration",
                "-of", "json", video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def _extract_keyframes_ffmpeg(
    video_path: str,
    interval_s: float,
    output_dir: str,
) -> list[tuple[float, str]]:
    """
    Use ffmpeg to extract one JPEG keyframe every `interval_s` seconds.

    Returns list of (timestamp_seconds, jpeg_path).
    """
    import subprocess
    output_pattern = os.path.join(output_dir, "frame_%05d.jpg")

    cmd = [
        FFMPEG_BIN, "-y", "-i", video_path,
        "-vf", f"fps=1/{interval_s}",
        "-q:v", "2",             # high quality JPEG
        "-vsync", "vfr",
        output_pattern,
    ]
    try:
        subprocess.run(
            cmd, capture_output=True, timeout=300, check=True
        )
    except Exception as exc:
        raise RuntimeError(f"ffmpeg keyframe extraction failed: {exc}") from exc

    # Collect extracted frames sorted by index
    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    result: list[tuple[float, str]] = []
    for i, frame_path in enumerate(frames):
        ts = i * interval_s
        result.append((ts, str(frame_path)))
    return result


def _extract_audio_ffmpeg(video_path: str, output_path: str) -> None:
    """Extract full audio track from video as mp3 using ffmpeg."""
    import subprocess
    cmd = [
        FFMPEG_BIN, "-y", "-i", video_path,
        "-vn",                   # no video
        "-acodec", "libmp3lame",
        "-q:a", "2",
        output_path,
    ]
    try:
        subprocess.run(
            cmd, capture_output=True, timeout=300, check=True
        )
    except Exception as exc:
        raise RuntimeError(f"ffmpeg audio extraction failed: {exc}") from exc


def _download_youtube(url: str, output_dir: str) -> str:
    """
    Download a YouTube video using yt-dlp.

    Returns the path of the downloaded file (best mp4 quality).
    Raises RuntimeError if yt-dlp is not found or download fails.
    """
    import subprocess
    if not _check_binary("yt-dlp", YTDLP_BIN):
        raise RuntimeError(
            "yt-dlp not found.  Install with:  pip install yt-dlp  "
            "or:  brew install yt-dlp"
        )
    output_template = os.path.join(output_dir, "yt_video.%(ext)s")
    cmd = [
        YTDLP_BIN,
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-playlist",
        "--quiet",
        url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=600, check=True)
    except Exception as exc:
        raise RuntimeError(f"yt-dlp download failed: {exc}") from exc

    # Find the downloaded file
    candidates = list(Path(output_dir).glob("yt_video.*"))
    if not candidates:
        raise RuntimeError("yt-dlp ran but no output file found.")
    return str(sorted(candidates)[0])


def _is_youtube_url(url: str) -> bool:
    """Return True if `url` looks like a YouTube video URL."""
    parsed = urlparse(url)
    return parsed.netloc.lower() in {
        "youtube.com", "www.youtube.com",
        "youtu.be", "www.youtu.be",
        "m.youtube.com",
    }


# ── Main video pipeline class ─────────────────────────────────────────────────

class VideoPipeline:
    """
    Full video content analysis pipeline.

    Steps
    -----
    1. Accept a local video file OR a YouTube URL (via yt-dlp).
    2. Use ffmpeg to extract 1 JPEG keyframe every KEYFRAME_INTERVAL_S seconds.
    3. Run ImagePipeline on each keyframe.
    4. Extract the full audio track with ffmpeg.
    5. Run AudioPipeline on the audio track.
    6. Merge: combine all visual tags, note timestamps, embed full transcript.

    Parameters
    ----------
    groq_api_key      : Falls back to GROQ_API_KEY env var.
    visual_model      : Vision LLM (default: llama-3.2-11b-vision-preview).
    audio_model       : Whisper model (default: whisper-large-v3).
    keyframe_interval : Seconds between extracted frames (default: 3.0).
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        visual_model: Optional[str] = None,
        audio_model: Optional[str] = None,
        keyframe_interval: float = KEYFRAME_INTERVAL_S,
    ):
        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "No Groq API key found.  Set GROQ_API_KEY or pass groq_api_key=."
            )
        self._image_pipe = ImagePipeline(
            groq_api_key=api_key, visual_model=visual_model
        )
        self._audio_pipe = AudioPipeline(groq_api_key=api_key)
        self._interval = max(0.5, keyframe_interval)

        # Verify ffmpeg is available at init time (warn, don't crash)
        if not _check_binary("ffmpeg", FFMPEG_BIN):
            import warnings
            warnings.warn(
                "ffmpeg not found on PATH.  Video analysis will fail at runtime.  "
                "Install: https://ffmpeg.org/download.html",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_video(
        self,
        source: str,
        *,
        previous_flags: int = 0,
        platform_policy: Optional[str] = None,
        current_severity: Optional[int] = None,
    ) -> VideoAnalysisResult:
        """
        Full video analysis: keyframe vision + audio transcription.

        Parameters
        ----------
        source          : local video file (mp4/webm/mov/avi/mkv) OR
                          YouTube URL (https://youtube.com/watch?v=...)
        previous_flags  : account prior violation count
        platform_policy : \"strict\" | \"moderate\" | \"lenient\" | None
        current_severity: existing scenario severity for tone boost

        Returns
        -------
        VideoAnalysisResult
          .transcript           — full audio transcript
          .visual_tags          — union of all frame tags (deduped)
          .frame_analysis       — [{timestamp_s, visual_tags, harmful_tags, ...}]
          .harmful_detected_at  — [{timestamp_s, tag}] for each harm hit
          .moderation_input     — dict ready to merge into env observation
        """
        with tempfile.TemporaryDirectory(prefix="multimodal_video_") as tmpdir:
            # Step 1: Obtain local video path
            if _is_youtube_url(source):
                video_path = _download_youtube(source, tmpdir)
            else:
                local = Path(source)
                if not local.exists():
                    raise FileNotFoundError(f"Video file not found: {source!r}")
                ext = local.suffix.lower()
                if ext not in SUPPORTED_VIDEO_EXTS:
                    raise ValueError(
                        f"Unsupported video format {ext!r}. "
                        f"Supported: {sorted(SUPPORTED_VIDEO_EXTS)}"
                    )
                video_path = str(local)

            # Step 2: Extract keyframes (1 per KEYFRAME_INTERVAL_S seconds)
            frames_dir = os.path.join(tmpdir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            frame_list = _extract_keyframes_ffmpeg(
                video_path, self._interval, frames_dir
            )

            # Step 3: Run ImagePipeline on each keyframe
            frame_analyses: list[FrameAnalysis] = []
            all_visual_tags: list[str] = []
            harmful_at: list[dict] = []

            import base64 as _b64mod
            for ts, frame_path in frame_list:
                raw = Path(frame_path).read_bytes()
                b64 = _b64mod.b64encode(raw).decode()
                vtags, harmful, conf = self._image_pipe.analyse_image_b64(b64)

                fa = FrameAnalysis(
                    timestamp_s=round(ts, 2),
                    visual_tags=vtags,
                    harmful_tags=harmful,
                    confidence_per_tag=conf,
                )
                frame_analyses.append(fa)
                all_visual_tags.extend(vtags)

                for harm_tag in harmful:
                    harmful_at.append({
                        "timestamp_s": round(ts, 2),
                        "tag": harm_tag,
                        "confidence": conf.get(harm_tag, 0.0),
                    })

            # Deduplicate visual tags preserving order
            seen: set[str] = set()
            unique_visual_tags: list[str] = []
            for t in all_visual_tags:
                if t not in seen:
                    seen.add(t)
                    unique_visual_tags.append(t)

            # Step 4: Extract audio track
            audio_path = os.path.join(tmpdir, "audio.mp3")
            audio_result = None
            try:
                _extract_audio_ffmpeg(video_path, audio_path)
                # Step 5: Run AudioPipeline on extracted audio
                audio_result = self._audio_pipe.analyse_audio(
                    source=audio_path,
                    previous_flags=previous_flags,
                    platform_policy=platform_policy,
                    current_severity=current_severity,
                )
            except Exception as _audio_exc:  # noqa: BLE001
                # Audio extraction failed — create a blank audio result
                audio_result = AudioAnalysisResult(
                    transcript=f"[audio extraction failed: {_audio_exc}]",
                    detected_language="unknown",
                    regional_context=False,
                    tone_label="calm",
                    tone_confidence=0.0,
                    tone_reasoning="Audio unavailable.",
                    route_to_human=False,
                    moderation_input={},
                )

        # Step 6: Merge results
        moderation_input: dict = {
            # Audio fields (env Observation compatible)
            "audio_transcript": audio_result.transcript,
            "detected_language": audio_result.detected_language,
            "regional_context": audio_result.regional_context,
            "tone_label": audio_result.tone_label,
            "tone_confidence": audio_result.tone_confidence,
            "route_to_human": audio_result.route_to_human,
            # Visual fields (env Observation compatible)
            "visual_tags": unique_visual_tags,
            # Extended fields for downstream use
            "harmful_detected_at": harmful_at,
            "frame_count": len(frame_analyses),
        }

        return VideoAnalysisResult(
            transcript=audio_result.transcript,
            detected_language=audio_result.detected_language,
            regional_context=audio_result.regional_context,
            tone_label=audio_result.tone_label,
            tone_confidence=audio_result.tone_confidence,
            route_to_human=audio_result.route_to_human,
            visual_tags=unique_visual_tags,
            harmful_detected_at=harmful_at,
            frame_analysis=[fa.to_dict() for fa in frame_analyses],
            moderation_input=moderation_input,
        )

    def enrich_state(self, state: dict, video: str) -> dict:
        """
        Enrich an env observation dict with real video analysis.
        Returns a shallow-copy of state with audio + visual fields merged in.
        """
        result = self.analyse_video(
            source=video,
            previous_flags=state.get("previous_flags", 0),
            platform_policy=state.get("platform_policy"),
        )
        enriched = dict(state)
        enriched.update(result.moderation_input)
        return enriched


# ── Convenience function (video) ──────────────────────────────────────────────

def analyse_video(
    source: str,
    *,
    groq_api_key: Optional[str] = None,
    previous_flags: int = 0,
    platform_policy: Optional[str] = None,
    current_severity: Optional[int] = None,
) -> dict:
    """
    Module-level convenience wrapper for video analysis.

    Example
    -------
        from multimodal_pipeline import analyse_video

        # local file
        result = analyse_video("clip.mp4")

        # YouTube
        result = analyse_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        print(result["transcript"])           # full audio transcript
        print(result["visual_tags"])          # union of all frame tags
        print(result["frame_analysis"])       # [{timestamp_s, tags, ...}, ...]
        print(result["harmful_detected_at"])  # [{timestamp_s, tag}, ...]
        print(result["moderation_input"])     # merge into env observation
    """
    pipe = VideoPipeline(groq_api_key=groq_api_key)
    return pipe.analyse_video(
        source=source,
        previous_flags=previous_flags,
        platform_policy=platform_policy,
        current_severity=current_severity,
    ).to_dict()


# ── CLI smoke-test ────────────────────────────────────────────────────────────

def _cli():
    """
    Quick smoke-test:
        python multimodal_pipeline.py audio  <file_or_url> [--json]
        python multimodal_pipeline.py image  <file_or_url> [--json]
        python multimodal_pipeline.py video  <file_or_url_or_youtube> [--json]
    """
    import argparse, pprint

    parser = argparse.ArgumentParser(
        description="Multimodal pipeline — audio / image / video analysis"
    )
    sub = parser.add_subparsers(dest="mode", required=False)

    def _add_common(p):
        p.add_argument("source")
        p.add_argument("--json", action="store_true")
        p.add_argument("--previous-flags", type=int, default=0)
        p.add_argument("--platform-policy", default=None)

    _add_common(sub.add_parser("audio", help="Transcribe + tone-detect an audio file"))
    _add_common(sub.add_parser("image", help="Extract visual tags from an image"))
    _add_common(sub.add_parser("video", help="Full video analysis (keyframes + audio)"))

    # Legacy positional (no sub-command → defaults to audio)
    parser.add_argument("source", nargs="?")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--previous-flags", type=int, default=0)
    parser.add_argument("--platform-policy", default=None)

    args = parser.parse_args()
    mode = args.mode or "audio"
    source = args.source
    as_json = args.json
    prev_flags = args.previous_flags
    policy = args.platform_policy

    if not source:
        parser.error("source argument required")

    if mode == "image":
        result = analyse_visual(source, platform_policy=policy, previous_flags=prev_flags)
    elif mode == "video":
        result = analyse_video(source, platform_policy=policy, previous_flags=prev_flags)
    else:
        result = analyse_audio(source, previous_flags=prev_flags, platform_policy=policy)

    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        pprint.pprint(result)


if __name__ == "__main__":
    _cli()


# ═══════════════════════════════════════════════════════════════════════════════
# Part 4 — Integration: moderate_multimodal()
# ═══════════════════════════════════════════════════════════════════════════════

import sys as _sys
from pathlib import Path as _Path_p4


def _get_inference_helpers():
    """Import get_llm_action / MODEL_NAME from inference.py without side effects."""
    import importlib.util as _ilu
    _here = _Path_p4(__file__).parent
    spec = _ilu.spec_from_file_location("_inf_mod_p4", _here / "inference.py")
    _mod = _ilu.module_from_spec(spec)
    _sys.modules.setdefault("_inf_mod_p4", _mod)
    try:
        spec.loader.exec_module(_mod)
    except Exception:
        pass
    return _mod


def _get_groq_openai_client(api_key: str):
    """Build an OpenAI-compatible Groq client (same pattern as inference.py)."""
    from openai import OpenAI
    return OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=api_key,
    )


def _build_evidence_entry(modality: str, result=None, present: bool = False) -> dict:
    if modality == "text":
        return {"present": present}
    if modality == "audio":
        if result is None:
            return {"present": False, "transcript": None, "tone": None,
                    "language": None, "regional_context": False, "route_to_human": False}
        d = result if isinstance(result, dict) else result.to_dict()
        return {
            "present": True,
            "transcript": d.get("transcript"),
            "tone": d.get("tone_label"),
            "language": d.get("detected_language"),
            "regional_context": d.get("regional_context", False),
            "route_to_human": d.get("route_to_human", False),
        }
    if modality == "image":
        if result is None:
            return {"present": False, "visual_tags": [], "harmful_tags": [],
                    "confidence_per_tag": {}}
        d = result if isinstance(result, dict) else result.to_dict()
        return {
            "present": True,
            "visual_tags": d.get("visual_tags", []),
            "harmful_tags": d.get("harmful_tags_detected", []),
            "confidence_per_tag": d.get("confidence_per_tag", {}),
        }
    if modality == "video":
        if result is None:
            return {"present": False, "transcript": None, "visual_tags": [],
                    "harmful_detected_at": [], "frame_count": 0}
        d = result if isinstance(result, dict) else result.to_dict()
        return {
            "present": True,
            "transcript": d.get("transcript"),
            "visual_tags": d.get("visual_tags", []),
            "harmful_detected_at": d.get("harmful_detected_at", []),
            "frame_count": d.get("moderation_input", {}).get("frame_count", 0),
        }
    return {"present": False}


def moderate_multimodal(
    text: Optional[str] = None,
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    previous_flags: int = 0,
    platform_policy: str = "moderate",
    *,
    groq_api_key: Optional[str] = None,
    tier: str = "hard",
) -> dict:
    """
    Orchestrate all pipelines that have inputs, merge signals into one
    observation dict, call get_llm_action from inference.py, and return a
    structured verdict with a per-modality evidence trail.

    Parameters
    ----------
    text            : Raw post text to moderate.
    audio_path      : Local audio file or URL (mp3/wav/ogg/m4a).
    image_path      : Local image file or URL (jpg/png/webp/gif).
    video_path      : Local video file or YouTube URL.
    previous_flags  : Account's prior policy violation count.
    platform_policy : "strict" | "moderate" | "lenient"
    groq_api_key    : Falls back to GROQ_API_KEY env var.
    tier            : "easy" | "medium" | "hard"  (affects LLM prompt + scoring).

    Returns
    -------
    {
        "verdict":     {label, action, severity, rationale},
        "observation": dict,           # merged env observation sent to LLM
        "evidence":    {text, audio, image, video},
        "signals":     {route_to_human, harmful_tags_found,
                        tone_boost_applied, regional_content},
        "model_used":  str,
        "error":       str | None
    }
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No Groq API key.  Set GROQ_API_KEY or pass groq_api_key=."
        )

    error: Optional[str] = None

    evidence: dict = {
        "text":  _build_evidence_entry("text", present=text is not None),
        "audio": _build_evidence_entry("audio"),
        "image": _build_evidence_entry("image"),
        "video": _build_evidence_entry("video"),
    }

    observation: dict = {
        "text":                  text or "",
        "previous_flags":        previous_flags,
        "platform_policy":       platform_policy,
        "audio_transcript":      None,
        "visual_tags":           [],
        "tone_label":            None,
        "tone_confidence":       None,
        "regional_context":      False,
        "route_to_human":        False,
        "harmful_tags_detected": [],
        "harmful_detected_at":   [],
        # env fields build_prompt expects
        "campaign_id":           None,
        "campaign_post_index":   None,
        "campaign_total_posts":  None,
        "is_adversarial":        False,
    }

    all_harmful_tags: list = []
    tone_boost_applied = False

    # ── Audio pipeline ────────────────────────────────────────────────────────
    if audio_path:
        try:
            ap = AudioPipeline(groq_api_key=api_key)
            ar = ap.analyse_audio(
                source=audio_path,
                previous_flags=previous_flags,
                platform_policy=platform_policy,
            )
            evidence["audio"] = _build_evidence_entry("audio", ar)
            observation.update(ar.moderation_input)
            if ar.tone_label == "aggressive":
                tone_boost_applied = True
        except Exception as exc:  # noqa: BLE001
            _e = f"audio_error: {exc}"
            error = (_e if not error else error + " | " + _e)
            evidence["audio"]["error"] = str(exc)

    # ── Image pipeline ────────────────────────────────────────────────────────
    if image_path:
        try:
            ip = ImagePipeline(groq_api_key=api_key)
            ir = ip.analyse_image(
                source=image_path,
                platform_policy=platform_policy,
                previous_flags=previous_flags,
            )
            evidence["image"] = _build_evidence_entry("image", ir)
            existing = list(observation.get("visual_tags") or [])
            observation["visual_tags"] = existing + [
                t for t in ir.visual_tags if t not in existing
            ]
            observation["harmful_tags_detected"] = ir.harmful_tags_detected
            all_harmful_tags.extend(ir.harmful_tags_detected)
        except Exception as exc:  # noqa: BLE001
            _e = f"image_error: {exc}"
            error = (_e if not error else error + " | " + _e)
            evidence["image"]["error"] = str(exc)

    # ── Video pipeline ────────────────────────────────────────────────────────
    if video_path:
        try:
            vp = VideoPipeline(groq_api_key=api_key)
            vr = vp.analyse_video(
                source=video_path,
                previous_flags=previous_flags,
                platform_policy=platform_policy,
            )
            evidence["video"] = _build_evidence_entry("video", vr)
            # Video audio overrides standalone audio only if no audio_path given
            if not audio_path:
                observation["audio_transcript"] = vr.transcript
                observation["detected_language"] = vr.detected_language
                observation["regional_context"]  = vr.regional_context
                observation["tone_label"]        = vr.tone_label
                observation["tone_confidence"]   = vr.tone_confidence
                observation["route_to_human"]    = vr.route_to_human
                if vr.tone_label == "aggressive":
                    tone_boost_applied = True
            existing = list(observation.get("visual_tags") or [])
            observation["visual_tags"] = existing + [
                t for t in vr.visual_tags if t not in existing
            ]
            observation["harmful_detected_at"] = vr.harmful_detected_at
            harm_from_video = list({h["tag"] for h in vr.harmful_detected_at})
            all_harmful_tags.extend(t for t in harm_from_video if t not in all_harmful_tags)
        except Exception as exc:  # noqa: BLE001
            _e = f"video_error: {exc}"
            error = (_e if not error else error + " | " + _e)
            evidence["video"]["error"] = str(exc)

    # ── Text population from transcripts (if original text is empty) ──────────
    if not observation.get("text"):
        # Prioritize video transcript then audio transcript
        if evidence["video"].get("transcript"):
            observation["text"] = evidence["video"]["transcript"]
        elif evidence["audio"].get("transcript"):
            observation["text"] = evidence["audio"]["transcript"]

    # ── Signals ───────────────────────────────────────────────────────────────
    route_to_human = bool(
        observation.get("route_to_human")
        or evidence["audio"].get("route_to_human")
    )
    regional_content = bool(
        observation.get("regional_context")
        or evidence["audio"].get("regional_context")
    )
    signals: dict = {
        "route_to_human":     route_to_human,
        "harmful_tags_found": list(dict.fromkeys(all_harmful_tags)),
        "tone_boost_applied": tone_boost_applied,
        "regional_content":   regional_content,
    }

    # ── LLM verdict (get_llm_action from inference.py) ────────────────────────
    verdict: dict = {}
    model_used: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    try:
        inf = _get_inference_helpers()
        client = _get_groq_openai_client(api_key)
        llm_action, llm_err = inf.get_llm_action(client, observation, tier)
        if llm_err:
            error = (error or "") + f" | llm_error: {llm_err}"
        verdict = {
            "label":     llm_action.get("label", "safe"),
            "action":    llm_action.get("action", "allow"),
            "severity":  llm_action.get("severity"),
            "rationale": llm_action.get("rationale"),
        }
        model_used = getattr(inf, "MODEL_NAME", model_used)
    except Exception as exc:  # noqa: BLE001
        _e = f"agent_error: {exc}"
        error = (error or "") + " | " + _e
        verdict = {"label": "safe", "action": "allow", "severity": None, "rationale": None}

    return {
        "verdict":     verdict,
        "observation": observation,
        "evidence":    evidence,
        "signals":     signals,
        "model_used":  model_used,
        "error":       error.strip(" |") if error else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test  (python multimodal_pipeline.py --self-test)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_self_test():
    """
    Creates a 1-second silent WAV, runs the full pipeline (audio + text),
    and prints the verdict + evidence trail.
    Requires GROQ_API_KEY in environment.
    """
    import struct, wave, tempfile, pprint as _pp

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("GROQ_API_KEY not set — cannot run self-test against live API.")
        return

    print("=" * 62)
    print("  multimodal_pipeline — Part 4 self-test")
    print("=" * 62)

    # 1. Minimal valid WAV (1 s, 16kHz, mono, silence)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    sr = 16000
    with wave.open(audio_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<" + "h" * sr, *([0] * sr)))
    print(f"  Synthetic WAV : {audio_path}")

    # 2. Run pipeline
    try:
        result = moderate_multimodal(
            text="Check out my new video — totally family-friendly content!",
            audio_path=audio_path,
            previous_flags=0,
            platform_policy="moderate",
            tier="hard",
        )
    finally:
        _Path_p4(audio_path).unlink(missing_ok=True)

    # 3. Print results
    v = result["verdict"]
    print(f"\n  Label    : {v['label']}")
    print(f"  Action   : {v['action']}")
    print(f"  Severity : {v['severity']}")
    print(f"  Rationale: {v['rationale']}")
    print("\n── Signals ──────────────────────────────────────────────")
    _pp.pprint(result["signals"])
    print("\n── Audio evidence ───────────────────────────────────────")
    ev_a = result["evidence"]["audio"]
    print(f"  Transcript : {ev_a.get('transcript')!r}")
    print(f"  Tone       : {ev_a.get('tone')}")
    print(f"  Language   : {ev_a.get('language')}")
    print(f"  Regional   : {ev_a.get('regional_context')}")
    print(f"  Route→human: {ev_a.get('route_to_human')}")
    print(f"\n  Model      : {result['model_used']}")
    if result["error"]:
        print(f"\n  Errors: {result['error']}")
    print("\n✅  Self-test complete.")
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 — Self-learning (Feedback Loop)
# ═══════════════════════════════════════════════════════════════════════════════

def learn_from_feedback(
    original_result: dict,
    human_verdict: dict,
    suggested_symbol_id: Optional[str] = None,
    suggested_keywords: Optional[list[str]] = None,
) -> dict:
    """
    Self-learning logic: processes human review feedback to update the
    symbol database.  This is Layer 3 of the moderation pipeline.

    If a human review overturns an agent decision and identifies a specific
    coded symbol, this function creates or promotes that symbol in the database.

    Parameters
    ----------
    original_result : dict
        The dict returned by moderate_multimodal().
    human_verdict   : dict
        Ground truth from a human reviewer: {"label": str, "action": str, "rationale": str}
    suggested_symbol_id : str | None
        If the human identifies a specific new symbol (e.g. "new-gang-sign-delta").
    suggested_keywords  : list[str] | None
        Fuzzy keywords the human associates with the new symbol.

    Returns
    -------
    dict : Summary of the learning action taken.
    """
    db = _SYMBOL_DB
    action_taken = "no_action"
    target_sid = suggested_symbol_id or ""

    # 1. Identify if learning is needed
    # Learning trigger: Agent said SAFE, Human said TOXIC/SPAM + provided symbol
    agent_label = original_result.get("verdict", {}).get("label")
    human_label = human_verdict.get("label")

    is_overturn = (agent_label == "safe" and human_label != "safe")
    is_confirmation = (agent_label != "safe" and agent_label == human_label)

    # 2. Update existing symbol if it was matched (confirmation)
    matches = original_result.get("signals", {}).get("harmful_tags_found", [])
    for match_id in matches:
        # Check all categories for this ID
        for cat_name, entries in db.items():
            for entry in entries:
                if entry["id"] == match_id.replace("-", "_"):
                    if is_confirmation:
                        entry["hit_count"] = entry.get("hit_count", 0) + 1
                        # Promotion logic: if trusted, lower the auto-flag threshold
                        if entry["hit_count"] >= 10 and entry["confidence_required"] > 0.8:
                            entry["confidence_required"] = max(0.75, entry["confidence_required"] - 0.05)
                            action_taken = f"promoted_existing_symbol: {match_id}"
                    elif is_overturn:
                        entry["miss_count"] = entry.get("miss_count", 0) + 1
                    break

    # 3. Create NEW symbol if human explicitly provided one during an overturn
    if is_overturn and target_sid:
        target_sid_clean = target_sid.replace("-", "_")
        exists = False
        for entries in db.values():
            if any(e["id"] == target_sid_clean for e in entries):
                exists = True; break

        if not exists:
            new_entry = {
                "id": target_sid_clean,
                "name": f"Human-Defined Symbol: {target_sid}",
                "keywords": suggested_keywords or [target_sid.replace("_", " ")],
                "aliases": [],
                "description": human_verdict.get("rationale", "Self-learned from human feedback overturn."),
                "confidence_required": 1.1,  # Start as Route-to-Human only
                "route_to_human_below": 0.8,
                "category": "coded_gestures",
                "severity": 3,
                "hit_count": 1,
                "miss_count": 0,
                "is_learned": True,
                "metadata": {
                    "sources": ["Self-Learning Feedback Loop"],
                    "verification_status": "Human Overturn Triggered"
                }
            }
            db.setdefault("coded_gestures", []).append(new_entry)
            action_taken = f"created_new_learned_symbol: {target_sid_clean}"

    # 4. Flush to disk
    if action_taken != "no_action":
        try:
            with open(_SYMBOL_DB_PATH, "w", encoding="utf-8") as f:
                json.dump({"_meta": {}, "categories": db}, f, indent=2, ensure_ascii=False)
            _reload_symbol_db()  # Refresh the module-level singleton
        except Exception as exc:
            action_taken = f"error_saving_db: {exc}"

    return {
        "action_taken": action_taken,
        "is_overturn": is_overturn,
        "symbol_id": target_sid_clean if target_sid else None
    }
