"""Visual style presets (prompt suffix, default negative prompt, board theme, colour fill)."""
import colorsys

_NEG_BASE = "text, letters, words, watermark, signature, logo, caption, blurry, low quality, deformed, cropped, frame, border"

STYLES = {
    "Whiteboard marker - black line art": dict(
        pos="simple black marker line drawing on pure white background, bold clean outlines, minimal doodle, whiteboard explainer illustration, no shading, lots of white space",
        neg="color, colour, shading, gradient, photo, realistic, 3d, background clutter, " + _NEG_BASE,
        theme="whiteboard", color=False),
    "Whiteboard doodle - colour fill": dict(
        pos="hand drawn doodle illustration, bold black outlines, flat bright marker colors, pure white background, whiteboard animation explainer style, simple, clean, centered",
        neg="photo, realistic, 3d render, gradient, complex background, shadows, " + _NEG_BASE,
        theme="whiteboard", color=True),
    "Cartoon explainer": dict(
        pos="flat cartoon explainer illustration, thick outlines, friendly characters, bright colors, plain white background, vector style",
        neg="photo, realistic, dark, gloomy, complex background, " + _NEG_BASE,
        theme="whiteboard", color=True),
    "Pencil sketch": dict(
        pos="pencil sketch drawing, graphite lines, hand drawn, white paper background, simple composition",
        neg="color, colour, photo, painting, 3d, " + _NEG_BASE,
        theme="paper", color=False),
    "Kids storybook": dict(
        pos="cute children's storybook illustration, soft crayon marker colors, rounded shapes, simple, plain light background",
        neg="scary, dark, realistic, photo, complex background, " + _NEG_BASE,
        theme="paper", color=True),
    "Flat infographic": dict(
        pos="flat vector infographic icon illustration, simple geometric shapes, limited color palette, plain white background",
        neg="photo, realistic, gradient, 3d, clutter, " + _NEG_BASE,
        theme="whiteboard", color=True),
    "Chalkboard": dict(
        pos="white chalk line drawing, simple doodle, bold lines, on black background",
        neg="color, colour, photo, realistic, shading, " + _NEG_BASE,
        theme="blackboard", color=False),
}
STYLE_NAMES = list(STYLES)

_NAMED = {
    "red": (220, 40, 40), "orange": (240, 140, 30), "yellow": (250, 220, 50), "green": (60, 170, 70), "teal": (30, 160, 160),
    "blue": (50, 100, 220), "navy": (30, 40, 120), "purple": (130, 70, 190), "pink": (240, 130, 190), "brown": (130, 85, 50),
    "beige": (225, 205, 170), "gray": (140, 140, 140), "black": (25, 25, 25), "white": (250, 250, 250),
}


def palette_words(rgb_triplets, k=3):
    """Describe dominant colours in words, e.g. 'teal, orange and navy color palette'."""
    names = []
    for r, g, b in rgb_triplets:
        n = min(_NAMED, key=lambda n: sum((a - c) ** 2 for a, c in zip(_NAMED[n], (r, g, b))))
        if n not in names and n not in ("white",):
            names.append(n)
    names = names[:k]
    if not names:
        return ""
    return (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]) + " color palette"


def build_prompts(scene_prompt: str, style_name: str, style_extra="", user_negative="", character="", palette=""):
    st = STYLES.get(style_name) or STYLES[STYLE_NAMES[1]]
    parts = [scene_prompt.strip().rstrip(".")]
    if character:
        parts.append(character)
    parts.append(st["pos"])
    if style_extra:
        parts.append(style_extra.strip())
    if palette:
        parts.append(palette)
    neg = st["neg"] + (", " + user_negative.strip() if user_negative.strip() else "")
    return ", ".join(p for p in parts if p), neg
