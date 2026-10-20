"""Helper for the bundled fonts in this folder (list: FONTS.txt). Names are <Family>-<Style>.

Use it from run_code like this (the sandbox runs Python in isolated mode, so add the path first):

    import os, sys
    sys.path.insert(0, os.environ["FONTS_DIR"])
    import pdf_fonts as pf

reportlab
    body = pf.register("Lora")                    # all Lora-* files + the <b>/<i> mapping; returns "Lora"
    head = pf.register("Poppins")
    ParagraphStyle("b", fontName=body)            # <b> -> Lora-Bold, <i> -> Lora-Italic
    canvas.setFont("Poppins-SemiBold", 14)        # any single weight by its full name
    pf.register("Poppins-SemiBold")               # (registering one weight alone also works)

fpdf2
    fam = pf.add_to_fpdf(pdf, "Lora")             # then pdf.set_font(fam, "B", 12) / "I" / "BI" / ""
    pf.add_to_fpdf(pdf, "Poppins-SemiBold")       # other weights: use the full name as the family, style ""

PyMuPDF (import fitz)
    page.insert_text((72, 72), "Hi", fontname="PoppinsBold", fontfile=pf.path("Poppins-Bold"))

Pillow / matplotlib
    ImageFont.truetype(pf.path("Lora-Bold"), 24)
    font_manager.fontManager.addfont(pf.path("Lora-Regular"))

Missing characters (IMPORTANT)
    Most fonts lack some of: Naira, Yoruba dot-below letters, subscript digits, arrows, Greek (Lora and
    Poppins lack all of them). Wrap Paragraph text:  Paragraph(pf.markup(text, "Lora"), style)  and the
    missing characters are drawn from a Noto fallback automatically. pf.missing("Lora", text) lists them.
    FONTS.txt shows what each family lacks.

Introspection:  pf.families()   pf.styles("Lora")   pf.has("Lora-Bold")   pf.path("Lora")  (= Lora-Regular)

Note: layout tables (kerning/ligatures) are stripped to keep the files small, which is invisible in
reportlab/PyMuPDF output.
"""
import os

FONTS_DIR = os.environ.get("FONTS_DIR") or os.path.dirname(os.path.abspath(__file__))

_INDEX = None  # {family: {style: path}}


def _index():
    global _INDEX
    if _INDEX is None:
        idx = {}
        for fn in os.listdir(FONTS_DIR):
            if fn.lower().endswith(".ttf") and "-" in fn:
                fam, _, style = fn[:-4].partition("-")
                idx.setdefault(fam, {})[style] = os.path.join(FONTS_DIR, fn)
        _INDEX = idx
    return _INDEX


def families():
    return sorted(_index())


def styles(family):
    return sorted(_index().get(_family(family), {}))


def _family(name):
    """Case-insensitive family lookup ('poppins' -> 'Poppins'); raises KeyError with suggestions."""
    idx = _index()
    if name in idx:
        return name
    low = {f.lower().replace(" ", ""): f for f in idx}
    key = name.lower().replace(" ", "").replace("-", "")
    if key in low:
        return low[key]
    near = [f for f in idx if key[:4] and key[:4] in f.lower()][:8]
    raise KeyError(f"font family {name!r} not found. Close matches: {near or 'none'}. All: see FONTS.txt")


def _split(name):
    """'Poppins-SemiBold' -> ('Poppins', 'SemiBold'); 'Lora' -> ('Lora', 'Regular')."""
    fam, _, style = name.partition("-")
    return _family(fam), (style or "Regular")


def has(name):
    try:
        fam, style = _split(name)
    except KeyError:
        return False
    return style in _index()[fam]


def path(name):
    """Full path of a font file. 'Lora' means Lora-Regular."""
    fam, style = _split(name)
    avail = _index()[fam]
    if style not in avail:
        raise KeyError(f"{fam} has no style {style!r}. Available: {sorted(avail)}")
    return avail[style]


def _pick(avail, *wanted):
    for w in wanted:
        if w in avail:
            return w
    return None


def register(name):
    """Register a font family (or one weight) with reportlab. Returns the name to use as fontName."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    done = set(pdfmetrics.getRegisteredFontNames())

    def _reg(face, file):
        if face not in done:
            pdfmetrics.registerFont(TTFont(face, file))
            done.add(face)

    if "-" in name:                                   # a single weight, e.g. Poppins-SemiBold
        fam, style = _split(name)
        face = f"{fam}-{style}"
        _reg(face, path(face))
        return face

    fam = _family(name)
    avail = _index()[fam]
    for style, file in avail.items():
        _reg(f"{fam}-{style}", file)
    normal = _pick(avail, "Regular", "Medium", "Light") or sorted(avail)[0]
    _reg(fam, avail[normal])                          # alias so fontName="Lora" works
    bold = _pick(avail, "Bold", "SemiBold") or normal
    ital = _pick(avail, "Italic") or normal
    bital = _pick(avail, "BoldItalic", "SemiBoldItalic") or ital
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily(fam, normal=fam, bold=f"{fam}-{bold}", italic=f"{fam}-{ital}", boldItalic=f"{fam}-{bital}")
    return fam


# Tried in this order for characters the main font lacks (together they cover Naira, Yoruba letters,
# sub/superscripts, Greek, arrows, equilibrium arrows and the common maths symbols).
FALLBACKS = ["NotoSans-Regular", "NotoSansMath-Regular", "NotoSansSymbols-Regular",
             "NotoSansSymbols2-Regular", "STIXTwoText-Regular"]


def _glyphs(face_name):
    from reportlab.pdfbase import pdfmetrics
    if face_name not in pdfmetrics.getRegisteredFontNames():
        register(face_name)
    return pdfmetrics.getFont(face_name).face.charToGlyph


def missing(font, text):
    """Characters of `text` that the font cannot draw (they would show as blanks/boxes)."""
    glyphs = _glyphs(font)
    return sorted({c for c in text if ord(c) > 32 and ord(c) not in glyphs and ord(c) != 0xA0})


def markup(text, font, fallbacks=None):
    """For reportlab Paragraph text: wrap each character `font` lacks in <font name=...> using the first
    bundled fallback font that has it. Tags and &entities in `text` are left alone, so <b>, <i>, <sub>,
    <br/> keep working. Use it on every Paragraph that may contain Naira, Yoruba letters, subscripts,
    Greek, arrows or maths symbols:   Paragraph(pf.markup("H₂O ⇌ ₦500", "Lora"), style)
    (canvas.drawString has no markup, so use a Paragraph, or pick a font with full coverage.)"""
    primary = _glyphs(font)
    chain = [f for f in (fallbacks or FALLBACKS) if has(f)]
    maps = {f: _glyphs(f) for f in chain}
    uncovered = set()

    def face_for(ch):
        if ord(ch) <= 32 or ord(ch) in primary or ord(ch) == 0xA0:
            return None
        for f in chain:
            if ord(ch) in maps[f]:
                return f
        uncovered.add(ch)
        return None

    import re
    out = []
    for piece in re.split(r"(<[^>]*>|&#?\w+;)", text):
        if not piece or piece[0] in "<&" and re.fullmatch(r"<[^>]*>|&#?\w+;", piece):
            out.append(piece)
            continue
        run, cur = [], None
        for ch in piece:
            f = face_for(ch)
            if f != cur and run:
                out.append(f'<font name="{cur}">{"".join(run)}</font>' if cur else "".join(run))
                run = []
            cur = f
            run.append(ch)
        if run:
            out.append(f'<font name="{cur}">{"".join(run)}</font>' if cur else "".join(run))
    if uncovered:
        print("[pdf_fonts] no bundled font has: " + " ".join(f"{c} (U+{ord(c):04X})" for c in sorted(uncovered)))
    return "".join(out)


def add_to_fpdf(pdf, name):
    """Register with an fpdf2 FPDF object. Returns the family name to pass to pdf.set_font()."""
    if "-" in name:                                   # a single weight: its own family, no style flag
        fam, style = _split(name)
        face = f"{fam}-{style}"
        pdf.add_font(family=face, style="", fname=path(face))
        return face
    fam = _family(name)
    avail = _index()[fam]
    for flag, style in (("", "Regular"), ("B", "Bold"), ("I", "Italic"), ("BI", "BoldItalic")):
        if style in avail:
            pdf.add_font(family=fam, style=flag, fname=avail[style])
    return fam
