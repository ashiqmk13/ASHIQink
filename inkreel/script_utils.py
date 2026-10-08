import re
from .langs import is_cjk_like

_SENT_SPLIT = re.compile(r"(?<=[.!?…।॥])\s+|(?<=[。！？])|\n+")


def split_sentences(text: str):
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return parts


def est_seconds(text: str, lang: str = "en", wps: float = 2.6) -> float:
    if is_cjk_like(lang) or " " not in text.strip():
        return max(0.5, len(re.sub(r"\s+", "", text)) / 4.5)
    return max(0.5, len(text.split()) / wps)


def split_scenes(script: str, lang: str = "en", target_sec: float = 9.0, paragraph_scenes: bool = True):
    """Split a script into narration chunks. The user's words are never rewritten here."""
    script = script.replace("\r\n", "\n").strip()
    if not script:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", script) if p.strip()]
    units = paras if (paragraph_scenes and len(paras) > 1) else [script]
    scenes = []
    for unit in units:
        if est_seconds(unit, lang) <= target_sec * 1.6 and len(units) > 1:
            scenes.append(re.sub(r"\s*\n\s*", " ", unit))
            continue
        cur, cur_t = [], 0.0
        for s in split_sentences(unit):
            t = est_seconds(s, lang)
            if cur and cur_t + t > target_sec * 1.25:
                scenes.append(" ".join(cur)); cur, cur_t = [], 0.0
            cur.append(s); cur_t += t
            if cur_t >= target_sec:
                scenes.append(" ".join(cur)); cur, cur_t = [], 0.0
        if cur:
            if scenes and cur_t < target_sec * 0.35 and len(units) == 1:
                scenes[-1] += " " + " ".join(cur)      # avoid a tiny last scene
            else:
                scenes.append(" ".join(cur))
    return scenes


_STOP = set("""a an the and or but if then so of to in on at by for with from as is are was were be been being it its this that these those
i you he she we they me him her us them my your his our their not no do does did have has had will would can could should may might
about into over under again very just also than too there here what which who whom when where why how all any each more most other some such only own same""".split())


def keywords(text: str, n: int = 6):
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
    seen, out = set(), []
    for w in words:
        lw = w.lower()
        if lw in _STOP or lw in seen:
            continue
        seen.add(lw); out.append(w)
        if len(out) >= n:
            break
    return out


def short_label(text: str, max_words: int = 5) -> str:
    first = split_sentences(text)[0] if text.strip() else ""
    if " " not in first.strip():
        return first[:14]
    return " ".join(first.split()[:max_words]).strip(" ,;:.-")
