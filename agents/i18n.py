"""Localization, language detection and English translation helpers."""

from __future__ import annotations

import os
import re

from langchain_core.prompts import ChatPromptTemplate

from .llm import get_llm


SUPPORTED_LANGS: dict[str, str] = {
    "en": "English",
    "si": "Sinhala",
    "ta": "Tamil",
}


_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "cli_title": "SmartMove CLI (type 'exit' to quit)",
        "you": "You",
        "smartmove": "SmartMove",
        "warn_no_key": "Warning: OPENAI_API_KEY not set. Please set it in your .env file.",
        "greeting": "Welcome to SmartMove. I can help with routes, schedules, and transport options.",
        "fallback": "SmartMove supports transportation queries only. Please ask about routes, stations, or travel options.",
        "follow_up_prefix": "Please provide the following required details",
        "origin": "origin",
        "destination": "destination",
        "departure_time": "departure time",
        "Origin": "Origin",
        "Destination": "Destination",
        "Departure time": "Departure time",
        "fare": "fare / budget",
        "Fare": "Fare (e.g. max LKR 2000, cheapest, or any)",
        "fare_toggle": "Specify fare or budget preference",
        "fare_toggle_need_value": "Enter a fare or budget preference, or turn off the toggle.",
    },
    "si": {
        "cli_title": "SmartMove CLI (ඉවත් වීමට 'exit' ටයිප් කරන්න)",
        "you": "ඔබ",
        "smartmove": "SmartMove",
        "warn_no_key": "අවවාදයයි: OPENAI_API_KEY සකසා නැත. කරුණාකර ඔබගේ .env ගොනුවේ සකසන්න.",
        "greeting": "SmartMove වෙත සාදරයෙන් පිළිගනිමු. මාර්ග, කාලසටහන් සහ ප්‍රවාහන විකල්ප ගැන ඔබට උපකාර කළ හැක.",
        "fallback": "SmartMove ප්‍රවාහන සම්බන්ධ ප්‍රශ්න සඳහා පමණක් සහාය දක්වයි. කරුණාකර මාර්ග/ස්ථාන/ගමන් විකල්ප ගැන අහන්න.",
        "follow_up_prefix": "කරුණාකර අවශ්‍ය විස්තර ලබා දෙන්න",
        "origin": "ආරම්භ ස්ථානය",
        "destination": "ගමනාන්තය",
        "departure_time": "පිටත්වෙන වේලාව",
        "Origin": "ආරම්භ ස්ථානය",
        "Destination": "ගමනාන්තය",
        "Departure time": "පිටත්වෙන වේලාව",
        "fare": "ගාස්තුව / අයවැය",
        "Fare": "ගාස්තුව (උදා: උපරිම LKR 2000, ලාභම, හෝ ඕනෑම)",
        "fare_toggle": "ගාස්තුව හෝ අයවැය මනාපයක් සඳහන් කරන්න",
        "fare_toggle_need_value": "ගාස්තුව ඇතුළත් කරන්න, නැතහොත් ටොගල් ක්‍රියාවිරහිත කරන්න.",
    },
    "ta": {
        "cli_title": "SmartMove CLI ('exit' என টাইப் செய்து வெளியேறலாம்)",
        "you": "நீங்கள்",
        "smartmove": "SmartMove",
        "warn_no_key": "எச்சரிக்கை: OPENAI_API_KEY அமைக்கப்படவில்லை. உங்கள் .env கோப்பில் அமைக்கவும்.",
        "greeting": "SmartMove-க்கு வரவேற்கிறோம். வழிகள், நேர அட்டவணை மற்றும் போக்குவரத்து விருப்பங்களில் உதவலாம்.",
        "fallback": "SmartMove போக்குவரத்து தொடர்பான கேள்விகளுக்கே ஆதரவு தருகிறது. வழிகள்/நிலையங்கள்/பயண விருப்பங்கள் பற்றி கேளுங்கள்.",
        "follow_up_prefix": "தேவையான விவரங்களை வழங்கவும்",
        "origin": "தொடக்க இடம்",
        "destination": "இலக்கு",
        "departure_time": "புறப்படும் நேரம்",
        "Origin": "தொடக்க இடம்",
        "Destination": "இலக்கு",
        "Departure time": "புறப்படும் நேரம்",
        "fare": "கட்டணம் / பட்ஜெட்",
        "Fare": "கட்டணம் (எ.கா: அதிகபட்சம் LKR 2000, மலிவானது, அல்லது எதுவும்)",
        "fare_toggle": "கட்டணம் அல்லது பட்ஜெட் விருப்பத்தைக் குறிப்பிடவும்",
        "fare_toggle_need_value": "கட்டணத்தை உள்ளிடவும், அல்லது டாகிளை அணைக்கவும்.",
    },
}


def _t(lang: str, key: str) -> str:
    """Translate a key to the requested language, falling back to English then the key itself."""
    lang_code = lang if lang in SUPPORTED_LANGS else "en"
    table = _TRANSLATIONS.get(lang_code, _TRANSLATIONS["en"])
    return table.get(key, _TRANSLATIONS["en"].get(key, key))


def detect_language(user_text: str) -> str:
    """Heuristic + LLM language detector returning one of `en`, `si`, `ta`."""
    text = (user_text or "").strip()
    if not text:
        return "en"
    if re.search(r"[\u0D80-\u0DFF]", text):
        return "si"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"

    if os.getenv("OPENAI_API_KEY"):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Detect the user's language. Return only one of: en, si, ta. "
                    "If unsure, return en.",
                ),
                ("human", "{text}"),
            ]
        )
        try:
            code = (prompt | get_llm()).invoke({"text": text}).content.strip().lower()
            if code in SUPPORTED_LANGS:
                return code
        except Exception:
            pass
    return "en"


def to_english(text: str, source_lang: str) -> str:
    """Translate `text` to English so downstream extraction works on a single language.

    Returns the input unchanged when already English or no API key is configured.
    """
    t = (text or "").strip()
    if not t:
        return ""
    if source_lang == "en":
        return t

    if not os.getenv("OPENAI_API_KEY"):
        return t

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Translate the user text to English.\n"
                "- Preserve place names (transliterate if needed).\n"
                "- Preserve times/dates.\n"
                "- Return only the translated text, no quotes.",
            ),
            ("human", "{text}"),
        ]
    )
    try:
        return (prompt | get_llm()).invoke({"text": t}).content.strip()
    except Exception:
        return t
