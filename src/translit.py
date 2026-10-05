"""Own transliteration to lower-case ASCII (replaces the GPL `unidecode` package).

Latin:  NFKD decomposition, combining accents dropped, a few special letters mapped.
Indic:  the nine Brahmi-derived Unicode blocks (Devanagari 0x900 ... Malayalam 0xD00) share the
        same layout: the same letter sits at the same offset inside each 0x80-wide block. One table
        indexed by offset therefore covers Hindi, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
        Telugu, Kannada and Malayalam. Consonants carry an inherent 'a' unless followed by a
        virama or a vowel sign; the word-final inherent 'a' is dropped (Hindi schwa deletion),
        so "कंसल्टिंग" -> "kansalting".
Other characters are dropped. Pure standard-library code, general knowledge only.
"""
import re
import unicodedata

_LATIN_SPECIAL = {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "đ": "d", "ł": "l", "þ": "th", "ð": "d",
                  "ı": "i", "ŋ": "ng", "ſ": "s", "ƒ": "f", "‘": "'", "’": "'", "“": '"', "”": '"',
                  "–": "-", "—": "-", "´": "'", "`": "'"}

_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
         0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
         0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
         0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
         0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
         0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_VOWEL = {0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
          0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
          0x60: "ri", 0x61: "li"}
_SIGN = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
         0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li",
         0x63: "li", 0x55: "", 0x56: "", 0x57: "u"}
_MARK = {0x01: "n", 0x02: "n", 0x03: "h"}  # chandrabindu, anusvara, visarga
_VIRAMA, _NUKTA = 0x4D, 0x3C
_MALAYALAM_CHILLU = {0xD7A: "n", 0xD7B: "n", 0xD7C: "r", 0xD7D: "l", 0xD7E: "l", 0xD7F: "k", 0xD54: "m",
                     0xD55: "y", 0xD56: "l"}
_SKIP = {0x200B, 0x200C, 0x200D, 0xFEFF}  # zero-width characters
_EXTRA_CONS = {0xB71: "v"}  # Oriya wa
# Malayalam doubled rra (റ്റ) is pronounced "tt"; rewrite it to the tta cluster first.
_PRE = {"റ്റ": "ട്ട"}
_ANUSVARA = "\x01"  # placeholder, resolved to m (before p/b/m or at word end) or n
_ANU_M = re.compile("\x01(?=[pbm]|[^a-z]|$)")


def _indic(cp):
    return 0x900 <= cp < 0xD80


def _latin(ch: str) -> str:
    if ch in _LATIN_SPECIAL:
        return _LATIN_SPECIAL[ch]
    d = unicodedata.normalize("NFKD", ch)
    out = "".join(c for c in d if not unicodedata.combining(c))
    return out if out.isascii() else ""


def to_ascii(s: str) -> str:
    """Lower-case ASCII transliteration of any string (Latin with accents, Indic scripts)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    for a, b in _PRE.items():
        if a in s:
            s = s.replace(a, b)
    out = []
    pending = False  # last emitted char is a consonant still carrying its inherent 'a'
    for ch in s:
        cp = ord(ch)
        if cp in _SKIP:
            continue
        if _indic(cp):
            if cp in _MALAYALAM_CHILLU:
                if pending:
                    out.append("a")
                out.append(_MALAYALAM_CHILLU[cp]); pending = False
                continue
            off = cp & 0x7F
            if cp in _EXTRA_CONS:
                if pending:
                    out.append("a")
                out.append(_EXTRA_CONS[cp]); pending = True
            elif off in _CONS:
                if pending:
                    out.append("a")
                out.append(_CONS[off]); pending = True
            elif off in _SIGN:
                out.append(_SIGN[off]); pending = False
            elif off == _VIRAMA:
                pending = False
            elif off == _NUKTA:
                continue
            elif off in _VOWEL:  # a consonant right before a vowel letter has no inherent 'a' (एलएलपी = elelpi)
                out.append(_VOWEL[off]); pending = False
            elif off in _MARK:
                if pending:
                    out.append("a")
                out.append(_ANUSVARA if off == 0x02 else _MARK[off]); pending = False
            elif 0x66 <= off <= 0x6F:
                if pending:
                    out.append("a")
                out.append(str(off - 0x66)); pending = False
            else:  # danda and other punctuation
                pending = False
                out.append(" ")
            continue
        # non-Indic character: a pending consonant ends a word here -> final schwa dropped
        pending = False
        if cp < 128:
            out.append(ch)
        else:
            out.append(_latin(ch))
    s = "".join(out).lower()
    if _ANUSVARA in s:
        s = _ANU_M.sub("m", s).replace(_ANUSVARA, "n")
    return s


_PH_MAP = str.maketrans({"c": "k", "q": "k", "g": "k", "b": "p", "d": "t", "w": "v", "z": "j", "x": "k", "f": "p"})


def phonetic(word: str) -> str:
    """Coarse sound key: aspirations removed, voiced/unvoiced merged, vowels dropped after the
    first letter, repeats collapsed. 'consulting' and 'kansalting' both give 'knsltnk'."""
    w = "".join(c for c in word if c.isalpha())
    if not w:
        return ""
    w = w.replace("ph", "f").replace("sh", "s").replace("ch", "s").replace("th", "t").replace("kh", "k")
    w = w.replace("gh", "g").replace("dh", "d").replace("bh", "b").replace("jh", "j").replace("ck", "k")
    w = w.replace("x", "ks")
    w = w.translate(_PH_MAP)
    head, tail = w[0], w[1:]
    tail = "".join(c for c in tail if c not in "aeiouyh")
    out = []
    for c in head + tail:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)
