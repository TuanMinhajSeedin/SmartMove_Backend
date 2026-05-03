"""Sri Lanka place-name normalization.

Many user queries name a place in Sinhala script (මහනුවර / හලාවත), in romanized
Sinhala (mahanuwara / halawatha), or as a common synonym (e.g. "Nuwara" for Kandy,
"Colpetty" for Kollupitiya). The Neo4j graph stores English Title-Case names
(Kandy, Chilaw, Kollupitiya). This module maps any of those variants to a
canonical lowercase English key (e.g. "kandy") that downstream cypher / response
agents can rely on.

Design:
- `_RAW_ALIASES` is a curated table of (canonical, [aliases...]) entries that
  covers the major Sri Lankan cities and the corridor stops in the dataset.
- At import time we build a flat lowercase index from every alias (and the
  canonical itself) -> canonical lowercase. Sinhala-script aliases are also
  indexed verbatim because they pass through `_clean_place` unchanged.
- `trails/normalized_places.xlsx` (when present) is merged in on top, so any
  data-team additions automatically extend the index.
- `canonical_place(value)` is the public API; it returns the canonical lowercase
  form, or None if the value didn't look like a known place. We do an exact
  lookup first, then try a couple of light fuzzy passes (suffix strip,
  whitespace squash) so small spelling drift still resolves.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def _optional_normalized_places_path() -> Path | None:
    """Locate ``trails/normalized_places.json`` regardless of repo nesting depth.

    Railway/Docker often uses a flat ``/app`` tree (only three parents from root),
    so a fixed ``parents[3]`` path raises ``IndexError``. Resolution order:

    1. ``NORMALIZED_PLACES_JSON`` — absolute or relative path to the JSON file.
    2. Walk upward from this package; use the first existing
       ``<dir>/trails/normalized_places.json`` (covers monorepo and flat layouts).
    """
    env = (os.environ.get("NORMALIZED_PLACES_JSON") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_file() else None

    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        candidate = ancestor / "trails" / "normalized_places.json"
        if candidate.is_file():
            return candidate
    return None

# (canonical English, aliases). Aliases include Sinhala script, common romanized
# Sinhala forms, Tamil names, and well-known English synonyms. Keep the
# canonical on the LEFT exactly as it appears in the Neo4j graph.
_RAW_ALIASES: list[tuple[str, list[str]]] = [
    # --- Western Province ---
    ("Colombo", [
        "කොළඹ", "කොලඹ", "kolamba", "kolomba", "kolomboe",
        "கொழும்பு", "kozhumbu",
        "colombo city", "colombo fort", "fort colombo",
    ]),
    ("Kollupitiya", ["කොල්ලුපිටිය", "colpetty", "kolpetty"]),
    ("Bambalapitiya", ["බම්බලපිටිය", "bambalapitiya", "bambalapitya"]),
    ("Wellawatte", ["වැල්ලවත්ත", "wellawatta", "wellawattha"]),
    ("Dehiwala", ["දෙහිවල", "dehiwela"]),
    ("Mount Lavinia", ["ගල්කිස්ස", "galkissa", "mt lavinia", "mount lavinia"]),
    ("Ratmalana", ["රත්මලාන", "rathmalana"]),
    ("Moratuwa", ["මොරටුව", "moratuwa", "moratuva"]),
    ("Panadura", ["පානදුර"]),
    ("Kalutara", ["කළුතර", "kalutara", "kaluthara"]),
    ("Beruwela", ["බේරුවල", "beruwela", "beruwala"]),
    ("Aluthgama", ["අළුත්ගම", "aluthgama"]),
    ("Negombo", ["මීගමුව", "meegamuwa", "negombo"]),
    ("Gampaha", ["ගම්පහ"]),
    ("Ja-Ela", ["ජා-ඇල", "ja ela", "ja-ela", "jaela"]),

    # --- Central / Hill country ---
    ("Kandy", [
        "මහනුවර", "මහ නුවර", "නුවර",
        "mahanuwara", "maha nuwara", "nuwara",
        "கண்டி", "kandi", "kandey",
        "kandy city",
    ]),
    ("Matale", ["මාතලේ", "matale"]),
    ("Nuwara Eliya", [
        "නුවරඑළිය", "නුවර එළිය",
        "nuwara eliya", "nuwaraeliya", "nuwareliya",
    ]),
    ("Hatton", ["හැටන්", "hatton"]),
    ("Bandarawela", ["බණ්ඩාරවෙල", "bandarawela"]),
    ("Badulla", ["බදුල්ල", "badulla"]),
    ("Ella", ["ඇල්ල", "ella"]),
    ("Haputale", ["හපුතලේ", "haputale"]),
    ("Mahiyangana", ["මහියංගණය", "mahiyangana", "mahiyanganaya"]),
    ("Kegalle", ["කෑගල්ල", "kegalle", "kegalla"]),
    ("Kurunegala", ["කුරුණෑගල", "kurunegala"]),
    ("Dambulla", ["දඹුල්ල", "dambulla"]),
    ("Sigiriya", ["සීගිරිය", "sigiriya"]),

    # --- Southern Province ---
    ("Galle", ["ගාල්ල", "ගල්ල", "galla", "காலி", "kaali", "galle fort"]),
    ("Hikkaduwa", ["හික්කඩුව", "hikkaduwa"]),
    ("Ambalangoda", ["අම්බලන්ගොඩ", "ambalangoda"]),
    ("Unawatuna", ["උණාවටුන", "unawatuna"]),
    ("Habaraduwa", ["හබරාදූව", "habaraduwa"]),
    ("Koggala", ["කොග්ගල", "koggala"]),
    ("Ahangama", ["අහංගම", "ahangama"]),
    ("Weligama", ["වැලිගම", "weligama"]),
    ("Mirissa", ["මිරිස්ස", "mirissa"]),
    ("Matara", ["මාතර", "matara", "maathara"]),
    ("Tangalle", ["තංගල්ල", "tangalle", "thangalla", "tangalla"]),
    ("Hambantota", ["හම්බන්තොට", "hambantota"]),
    ("Tissamaharama", ["තිස්සමහාරාම", "tissamaharama", "tissa"]),
    ("Kataragama", ["කතරගම", "kataragama"]),
    ("Mirigama", ["මිරිගම", "mirigama"]),
    ("Rathgama", ["රත්ගම", "rathgama", "ratgama"]),
    ("Kalugama", ["කල්ගම", "kalugama"]),

    # --- North-Western & North-Central ---
    ("Chilaw", ["හලාවත", "halawatha", "halawata", "சிலாபம்", "chilapam"]),
    ("Puttalam", ["පුත්තලම", "puttalam", "puthalam"]),
    ("Anuradhapura", ["අනුරාධපුර", "anuradhapura", "anuradapura"]),
    ("Polonnaruwa", ["පොලොන්නරුව", "polonnaruwa"]),
    ("Ratnapura", ["රත්නපුර", "ratnapura", "rathnapura"]),
    ("Avissawella", ["අවිස්සාවේල්ල", "avissawella", "awissawella"]),
    ("Embilipitiya", ["ඇඹිලිපිටිය", "embilipitiya"]),

    # --- Eastern & Northern ---
    ("Trincomalee", [
        "ත්‍රිකුණාමලය", "trikunamalaya", "thirikunamalaya",
        "திருகோணமலை", "trinco",
    ]),
    ("Batticaloa", [
        "මඩකලපුව", "madakalapuwa",
        "மட்டக்களப்பு", "mattakkalappu",
    ]),
    ("Ampara", ["අම්පාර", "ampara"]),
    ("Kalmunai", ["කල්මුණේ", "kalmunai"]),
    ("Jaffna", [
        "යාපනය", "yapanaya",
        "யாழ்ப்பாணம்", "yarlpanam", "yaazhpaanam",
    ]),
    ("Vavuniya", ["වව්නියාව", "vavuniya"]),
    ("Mannar", ["මන්නාරම", "mannar", "mannaram"]),
    ("Kilinochchi", ["කිලිනොච්චිය", "kilinochchi"]),
    ("Mullaitivu", ["මුලතිව්", "mullaitivu", "mullaithivu"]),
]


# Suffix tokens that some folks tack onto place names but which the graph
# never stores ("Kandy town", "Galle city"). Stripped before lookup.
_TRAILING_NOISE = (
    "city", "town", "district", "area", "stop", "station", "bus stop",
    "bus station", "fort",
)


def _strip_trailing_noise(t: str) -> str:
    s = t
    changed = True
    while changed:
        changed = False
        for w in _TRAILING_NOISE:
            if s.endswith(" " + w):
                s = s[: -(len(w) + 1)].strip()
                changed = True
                break
    return s


def _build_index() -> dict[str, str]:
    idx: dict[str, str] = {}

    def add(alias: str, canonical: str) -> None:
        a = re.sub(r"\s+", " ", (alias or "").strip())
        if not a:
            return
        idx.setdefault(a.lower(), canonical.lower())

    for canonical, aliases in _RAW_ALIASES:
        add(canonical, canonical)
        for al in aliases:
            add(al, canonical)

    excel_json = _optional_normalized_places_path()
    if excel_json is not None:
        try:
            data = json.loads(excel_json.read_text(encoding="utf-8"))
            for row in data:
                eng = (row.get("english") or "").strip()
                if not eng:
                    continue
                add(eng, eng)
                for col in ("original", "corrected_sinhala"):
                    val = (row.get(col) or "").strip()
                    if val:
                        add(val, eng)
                aliases_field = row.get("aliases")
                if isinstance(aliases_field, str) and aliases_field.strip():
                    inner = aliases_field.strip()
                    if inner.startswith("[") and inner.endswith("]"):
                        inner = inner[1:-1]
                    for tok in inner.split(","):
                        tok = tok.strip().strip("'").strip('"')
                        if tok:
                            add(tok, eng)
        except Exception:
            pass

    return idx


_ALIAS_INDEX: dict[str, str] = _build_index()


def canonical_place(value: str | None) -> str | None:
    """Return the canonical lowercase English place name, or None.

    Lookup order:
      1. Exact lowercase match against the alias index.
      2. Lookup with trailing noise tokens removed ("kandy city" -> "kandy").
      3. Whitespace-collapsed lookup.
    """
    if not value:
        return None
    raw = re.sub(r"\s+", " ", str(value).strip())
    if not raw:
        return None

    key = raw.lower()
    hit = _ALIAS_INDEX.get(key)
    if hit:
        return hit

    stripped = _strip_trailing_noise(key)
    if stripped and stripped != key:
        hit = _ALIAS_INDEX.get(stripped)
        if hit:
            return hit

    squashed = re.sub(r"[\s\-]+", "", key)
    for alias_lc, canon in _ALIAS_INDEX.items():
        if re.sub(r"[\s\-]+", "", alias_lc) == squashed:
            return canon

    return None


def known_canonicals() -> list[str]:
    """Sorted list of canonical lowercase names; useful for prompts / debugging."""
    return sorted({v for v in _ALIAS_INDEX.values()})
