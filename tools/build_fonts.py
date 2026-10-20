#!/usr/bin/env python3
"""Turn raw Google Fonts files into the repo's fonts/ folder.

Why this exists
  * reportlab / fpdf2 / PyMuPDF cannot use the named weights inside a *variable*
    font (e.g. Lora[wght].ttf), so bold/semibold would silently fall back to
    regular. This script cuts each weight out as its own static file.
  * Each file is trimmed to the characters a school document needs (Latin,
    Yoruba tone marks, the Naira sign, Greek, sub/superscripts, arrows, maths
    and science symbols), which keeps the whole library small.

Usage
    python tools/build_fonts.py SRC_DIR [--out fonts] [--weights 300,400,500,600,700]
                                        [--keep-layout] [--all-weights]

    SRC_DIR is any folder tree that contains .ttf files: a checkout of
    github.com/google/fonts, or fonts unzipped from fonts.google.com.
    Static fonts keep Light/Regular/Medium/SemiBold/Bold (+ italics). Variable fonts are split
    into those same weights (+ italics when an italic file exists).

Needs:  pip install fonttools
To add a font later: unzip it anywhere, run this script on that folder, commit fonts/.
"""
import argparse
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

# Characters kept in every font (everything else is dropped to save space).
UNICODE_RANGES = [
    (0x0020, 0x007E),   # Basic Latin
    (0x00A0, 0x00FF),   # Latin-1 (degree, plus-minus, multiply, divide ...)
    (0x0100, 0x024F),   # Latin Extended-A/B
    (0x02B0, 0x02FF),   # spacing modifiers
    (0x0300, 0x036F),   # combining marks
    (0x0370, 0x03FF),   # Greek (alpha, beta, delta, omega ...)
    (0x1E00, 0x1EFF),   # Latin Extended Additional (Yoruba: e, o with dot below, etc.)
    (0x2000, 0x206F),   # general punctuation
    (0x2070, 0x209F),   # superscripts and subscripts (H2O, x2)
    (0x20A0, 0x20CF),   # currency (Naira U+20A6, Euro, ...)
    (0x2100, 0x214F),   # letterlike (degree C, Angstrom ...)
    (0x2150, 0x218F),   # number forms (fractions)
    (0x2190, 0x21FF),   # arrows (including equilibrium arrows)
    (0x2200, 0x22FF),   # mathematical operators
    (0x2300, 0x23FF),   # misc technical
    (0x25A0, 0x25FF),   # geometric shapes
    (0x2600, 0x266F),   # misc symbols
    (0x2713, 0x2714),   # check marks
    (0xFB00, 0xFB06),   # ligatures
    (0xFEFF, 0xFEFF),
]

WEIGHT_NAMES = {100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular",
                500: "Medium", 600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black"}

# Rough grouping, written into fonts/FONTS.txt so a model can pick sensibly.
CATEGORY = {
    "sans": "Poppins Roboto OpenSans Montserrat Inter Nunito NunitoSans Raleway SourceSans3 PTSans FiraSans "
            "Ubuntu NotoSans DMSans WorkSans Quicksand Lato JosefinSans Karla Rubik Mulish Barlow Heebo Cabin "
            "TitilliumWeb Archivo Assistant Arimo LibreFranklin AlegreyaSans IBMPlexSans Manrope Outfit Urbanist "
            "Lexend Sora SpaceGrotesk PlusJakartaSans Figtree PublicSans Overpass Hind Catamaran RobotoCondensed".split(),
    "serif": "Lora Merriweather PlayfairDisplay SourceSerif4 PTSerif NotoSerif CrimsonText CrimsonPro "
             "LibreBaskerville EBGaramond CormorantGaramond DMSerifDisplay Tinos Spectral Cardo Alegreya Vollkorn "
             "STIXTwoText IBMPlexSerif Bitter ZillaSlab Slabo27px Arvo Domine Gelasio Newsreader Literata RobotoSlab".split(),
    "mono": "RobotoMono FiraCode JetBrainsMono SourceCodePro IBMPlexMono Cousine".split(),
    "display/handwriting": "Oswald BebasNeue Anton DancingScript Pacifico Exo2 Comfortaa Lobster Caveat "
                           "IndieFlower AmaticSC".split(),
    "symbols (fallback for maths/science glyphs)": "NotoSansMath NotoSansSymbols NotoSansSymbols2".split(),
}


def _unicodes():
    out = []
    for lo, hi in UNICODE_RANGES:
        out.extend(range(lo, hi + 1))
    return out


def _split_name(path: Path):
    """'Lora-Italic[wght].ttf' -> ('Lora', 'Italic', is_variable)"""
    stem = path.stem
    is_var = "[" in stem
    stem = re.sub(r"\[.*\]", "", stem)
    stem = stem.replace("_", "").replace("-Web", "")
    family, _, style = stem.partition("-")
    return family, (style or "Regular"), is_var


def _style_name(weight, italic):
    base = WEIGHT_NAMES.get(weight, str(weight))
    if italic:
        return "Italic" if weight == 400 else base + "Italic"
    return base


KEEP_LAYOUT = ["kern", "mark", "mkmk", "ccmp", "locl", "liga"]
# Static weights kept by default (Thin/ExtraLight/ExtraBold/Black are skipped to save space).
STATIC_WEIGHTS_KEPT = {"Light", "Regular", "Medium", "SemiBold", "Bold"}


def _subset(path: Path, keep_layout: bool = False):
    opts = subset.Options()
    # reportlab, fpdf2 (without HarfBuzz) and PyMuPDF never read GPOS/GSUB, and those tables are
    # 70-90% of a font's size. Dropping them is invisible in the PDF; --keep-layout restores them.
    opts.layout_features = KEEP_LAYOUT if keep_layout else []
    opts.name_IDs = ["*"]
    opts.notdef_outline = True
    opts.glyph_names = False
    opts.hinting = False
    opts.legacy_kern = keep_layout
    font = subset.load_font(str(path), opts)
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=_unicodes())
    sub.subset(font)
    return font


def _set_names(font, family, style):
    """Give a cut-out weight its own internal names. Without this every instance keeps the variable
    font's default name ('Lora-Regular'), so a PDF embeds Bold, Regular, ... under one identical name and
    some viewers/tools then show the wrong weight."""
    italic = style.endswith("Italic")
    base = style[: -len("Italic")] if italic else style
    base = base or "Regular"
    ribbi = base in ("Regular", "Bold")
    n1 = family if ribbi else f"{family} {base}"
    n2 = " ".join(p for p in ("Bold" if base == "Bold" else "", "Italic" if italic else "") if p) or "Regular"
    full = f"{family} Italic" if (italic and base == "Regular") else f"{family} {base}" + (" Italic" if italic else "")
    ps = f"{family}-{style}"
    names = font["name"]
    for nid in (16, 17, 21, 22, 25):
        names.removeNames(nameID=nid)
    for nid, val in ((1, n1), (2, n2), (3, f"{ps};static"), (4, full), (6, ps)):
        names.setName(val, nid, 3, 1, 0x409)
        names.setName(val, nid, 1, 0, 0)
    bold = base == "Bold"
    font["head"].macStyle = (1 if bold else 0) | (2 if italic else 0)
    if "OS/2" in font:
        sel = font["OS/2"].fsSelection & ~((1 << 0) | (1 << 5) | (1 << 6))
        sel |= (1 << 0) if italic else 0
        sel |= (1 << 5) if bold else 0
        sel |= (1 << 6) if (not bold and not italic) else 0
        font["OS/2"].fsSelection = sel


def process(args):
    """Convert one source .ttf into one or more static, trimmed fonts. Returns (written, skipped_msgs)."""
    path, out_dir, weights, keep_layout, all_weights = args
    path, out_dir = Path(path), Path(out_dir)
    family, style, is_var = _split_name(path)
    written, notes = [], []
    try:
        if not is_var and not all_weights and style.replace("Italic", "") not in (STATIC_WEIGHTS_KEPT | {""}):
            return [], notes                      # e.g. Poppins-Thin / -Black: skipped by default
        font = _subset(path, keep_layout)
        if "fvar" not in font:
            name = f"{family}-{style}.ttf"
            font.save(out_dir / name)
            return [name], notes
        axes = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in font["fvar"].axes}
        italic = "Italic" in style
        targets = [None] if "wght" not in axes else [
            w for w in weights if axes["wght"][0] <= w <= axes["wght"][2]
        ] or [int(axes["wght"][1])]
        for w in targets:
            loc = {tag: d for tag, (_, d, _) in axes.items()}   # pin every axis to its default...
            if w is not None:
                loc["wght"] = w                                   # ...except weight
            inst = instancer.instantiateVariableFont(font, loc, inplace=False)
            sname = _style_name(w if w is not None else 400, italic)
            _set_names(inst, family, sname)
            name = f"{family}-{sname}.ttf"
            inst.save(out_dir / name)
            written.append(name)
    except Exception as exc:  # one bad file must not stop the whole build
        notes.append(f"SKIPPED {path.name}: {exc}")
    return written, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "fonts"))
    ap.add_argument("--weights", default="300,400,500,600,700", help="weights cut from variable fonts")
    ap.add_argument("--keep-layout", action="store_true", help="keep kerning/ligature tables (3-10x larger files)")
    ap.add_argument("--all-weights", action="store_true", help="also keep Thin/ExtraLight/ExtraBold/Black static fonts")
    ns = ap.parse_args()

    src, out = Path(ns.src), Path(ns.out)
    weights = [int(w) for w in ns.weights.split(",")]
    out.mkdir(parents=True, exist_ok=True)
    (out / "licenses").mkdir(exist_ok=True)

    files = sorted(p for p in src.rglob("*.ttf") if ".git" not in p.parts)
    if not files:
        sys.exit(f"No .ttf files under {src}")
    print(f"{len(files)} source files -> {out}")

    # Keep each family's licence next to the fonts (the OFL requires it to travel with them).
    for lic in list(src.rglob("OFL.txt")) + list(src.rglob("UFL.txt")) + list(src.rglob("LICENSE.txt")):
        if ".git" in lic.parts:
            continue
        fam = lic.parent.name
        shutil.copyfile(lic, out / "licenses" / f"{fam}-{lic.name}")

    jobs = [(str(p), str(out), weights, ns.keep_layout, ns.all_weights) for p in files]
    written, notes = [], []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 2) as pool:
        for w, n in pool.map(process, jobs, chunksize=1):
            written += w
            notes += n
    for n in notes:
        print(n)
    write_index(out)
    total = sum(f.stat().st_size for f in out.glob("*.ttf"))
    print(f"{len(list(out.glob('*.ttf')))} fonts, {total / 1e6:.1f} MB total")


GLYPH_GROUPS = {
    "naira": [0x20A6],
    "yoruba-letters": [0x1EB9, 0x1ECD, 0x1E63, 0x1EB8, 0x1ECC, 0x1E62],
    "sub/superscripts": list(range(0x2080, 0x208A)) + [0x00B2, 0x00B3, 0x207A, 0x207B],
    "arrows": [0x2192, 0x21CC],
    "greek": [0x0394, 0x03A9, 0x03C0, 0x03BB, 0x03BC, 0x03B1, 0x03B2],
    "maths": [0x00B0, 0x00B1, 0x00D7, 0x00F7, 0x221A, 0x2248, 0x2260, 0x2264, 0x2265, 0x221E],
}


def _lacks(out: Path, fam: str):
    """Glyph groups a family cannot draw (checked on its Regular file)."""
    p = out / f"{fam}-Regular.ttf"
    if not p.exists():
        p = sorted(out.glob(f"{fam}-*.ttf"))[0]
    cmap = TTFont(p).getBestCmap()
    return [g for g, cps in GLYPH_GROUPS.items() if not all(c in cmap for c in cps)]


def write_index(out: Path):
    """fonts/FONTS.txt: one line per family listing its styles, grouped by category."""
    fams = {}
    for f in out.glob("*.ttf"):
        fam, _, style = f.stem.partition("-")
        fams.setdefault(fam, []).append(style)
    lacks = {fam: _lacks(out, fam) for fam in fams}
    order = list(WEIGHT_NAMES.values())

    def key(s):
        base = s.replace("Italic", "") or "Regular"
        return (base in order and order.index(base) or 99, "Italic" in s)

    def line(fam):
        tail = f"   || lacks: {', '.join(lacks[fam])}" if lacks[fam] else "   || full symbol coverage"
        return f"{fam}: {', '.join(sorted(fams[fam], key=key))}{tail}"

    lines = ["# Installed fonts. Use the name exactly as written: <Family>-<Style>, e.g. Poppins-SemiBold.",
             "# Styles: Regular/Italic/Bold/BoldItalic exist for most families; other weights where listed.",
             "# 'lacks' = characters the family cannot draw. Wrap Paragraph text in pf.markup(text, font) and the",
             "# missing ones come from a Noto fallback automatically (see pdf_fonts.py).", ""]
    seen = set()
    for cat, names in CATEGORY.items():
        lines.append(f"[{cat}]")
        for fam in names:
            if fam in fams:
                lines.append(line(fam))
                seen.add(fam)
        lines.append("")
    rest = sorted(set(fams) - seen)
    if rest:
        lines.append("[other]")
        for fam in rest:
            lines.append(line(fam))
    (out / "FONTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
