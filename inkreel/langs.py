"""Language table: which TTS engine can speak which language + which font script it needs."""
import re

# kokoro: lang_code ; chatterbox: language_id ; mms: facebook/mms-tts-<code> ; espeak: -v <code>
LANGS = {
    "en":    dict(name="English (US)",      kokoro="a", chatterbox="en", mms="eng", espeak="en",    script="latin"),
    "en-gb": dict(name="English (UK)",      kokoro="b", chatterbox="en", mms="eng", espeak="en-gb", script="latin"),
    "hi":    dict(name="Hindi",             kokoro="h", chatterbox="hi", mms="hin", espeak="hi",    script="devanagari"),
    "ml":    dict(name="Malayalam",         kokoro=None, chatterbox=None, mms="mal", espeak="ml",   script="malayalam"),
    "ta":    dict(name="Tamil",             kokoro=None, chatterbox=None, mms="tam", espeak="ta",   script="tamil"),
    "te":    dict(name="Telugu",            kokoro=None, chatterbox=None, mms="tel", espeak="te",   script="telugu"),
    "kn":    dict(name="Kannada",           kokoro=None, chatterbox=None, mms="kan", espeak="kn",   script="kannada"),
    "bn":    dict(name="Bengali",           kokoro=None, chatterbox=None, mms="ben", espeak="bn",   script="bengali"),
    "mr":    dict(name="Marathi",           kokoro=None, chatterbox=None, mms="mar", espeak="mr",   script="devanagari"),
    "gu":    dict(name="Gujarati",          kokoro=None, chatterbox=None, mms="guj", espeak="gu",   script="gujarati"),
    "pa":    dict(name="Punjabi",           kokoro=None, chatterbox=None, mms="pan", espeak="pa",   script="gurmukhi"),
    "ur":    dict(name="Urdu",              kokoro=None, chatterbox=None, mms="urd-script_arabic", espeak="ur", script="arabic"),
    "es":    dict(name="Spanish",           kokoro="e", chatterbox="es", mms="spa", espeak="es",    script="latin"),
    "fr":    dict(name="French",            kokoro="f", chatterbox="fr", mms="fra", espeak="fr",    script="latin"),
    "de":    dict(name="German",            kokoro=None, chatterbox="de", mms="deu", espeak="de",   script="latin"),
    "it":    dict(name="Italian",           kokoro="i", chatterbox="it", mms=None, espeak="it",     script="latin"),
    "pt":    dict(name="Portuguese",        kokoro="p", chatterbox="pt", mms="por", espeak="pt",    script="latin"),
    "ru":    dict(name="Russian",           kokoro=None, chatterbox="ru", mms="rus", espeak="ru",   script="cyrillic"),
    "ar":    dict(name="Arabic",            kokoro=None, chatterbox="ar", mms="ara", espeak="ar",   script="arabic"),
    "zh":    dict(name="Chinese (Mandarin)", kokoro="z", chatterbox="zh", mms=None, espeak="cmn",   script="cjk_sc"),
    "ja":    dict(name="Japanese",          kokoro="j", chatterbox="ja", mms=None, espeak="ja",     script="cjk_jp"),
    "ko":    dict(name="Korean",            kokoro=None, chatterbox="ko", mms="kor", espeak="ko",   script="cjk_kr"),
    "tr":    dict(name="Turkish",           kokoro=None, chatterbox="tr", mms="tur", espeak="tr",   script="latin"),
    "id":    dict(name="Indonesian",        kokoro=None, chatterbox=None, mms="ind", espeak="id",   script="latin"),
    "vi":    dict(name="Vietnamese",        kokoro=None, chatterbox=None, mms="vie", espeak="vi",   script="latin"),
    "th":    dict(name="Thai",              kokoro=None, chatterbox=None, mms="tha", espeak="th",   script="thai"),
    "nl":    dict(name="Dutch",             kokoro=None, chatterbox="nl", mms=None, espeak="nl",    script="latin"),
    "pl":    dict(name="Polish",            kokoro=None, chatterbox="pl", mms=None, espeak="pl",    script="latin"),
    "sv":    dict(name="Swedish",           kokoro=None, chatterbox="sv", mms=None, espeak="sv",    script="latin"),
    "da":    dict(name="Danish",            kokoro=None, chatterbox="da", mms=None, espeak="da",    script="latin"),
    "fi":    dict(name="Finnish",           kokoro=None, chatterbox="fi", mms=None, espeak="fi",    script="latin"),
    "no":    dict(name="Norwegian",         kokoro=None, chatterbox="no", mms=None, espeak="nb",    script="latin"),
    "el":    dict(name="Greek",             kokoro=None, chatterbox="el", mms=None, espeak="el",    script="greek"),
    "he":    dict(name="Hebrew",            kokoro=None, chatterbox="he", mms=None, espeak="he",    script="hebrew"),
    "sw":    dict(name="Swahili",           kokoro=None, chatterbox="sw", mms="swh", espeak="sw",   script="latin"),
}

LANG_CHOICES = [("Auto-detect", "auto")] + [(v["name"], k) for k, v in LANGS.items()]
NAME_TO_CODE = {v["name"]: k for k, v in LANGS.items()}

_RANGES = [
    (0x0D00, 0x0D7F, "ml"), (0x0B80, 0x0BFF, "ta"), (0x0C00, 0x0C7F, "te"), (0x0C80, 0x0CFF, "kn"),
    (0x0980, 0x09FF, "bn"), (0x0A80, 0x0AFF, "gu"), (0x0A00, 0x0A7F, "pa"), (0x0900, 0x097F, "hi"),
    (0x0600, 0x06FF, "ar"), (0x0400, 0x04FF, "ru"), (0x0E00, 0x0E7F, "th"), (0xAC00, 0xD7AF, "ko"),
    (0x3040, 0x30FF, "ja"), (0x4E00, 0x9FFF, "zh"), (0x0370, 0x03FF, "el"), (0x0590, 0x05FF, "he"),
]


def detect_lang(text: str) -> str:
    """Cheap script-based detection (Latin text -> English; pick another language manually)."""
    counts = {}
    for ch in text:
        o = ord(ch)
        for lo, hi, code in _RANGES:
            if lo <= o <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if counts.get("ja"):          # kana present -> Japanese even if kanji dominate
        return "ja"
    if counts:
        return max(counts, key=counts.get)
    return "en"


def is_cjk_like(lang: str) -> bool:
    return lang in ("zh", "ja", "ko", "th")
