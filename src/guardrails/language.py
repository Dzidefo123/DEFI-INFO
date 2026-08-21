"""Deterministic language detection, ahead of the pattern gate.

The guardrails in `rules` are English regexes. That is fine as a matching
strategy and fatal as a coverage strategy: a compromise report written in
Portuguese does not match `COMPROMISE`, so it does not escalate — it falls
through to the router and reaches a model. The layer exists precisely so that the
highest-cost inputs never reach a model, and outside English it silently stops
doing that.

Translating thirty-eight patterns per language does not fix this. It moves the
same failure to language thirty-nine, and every gap looks identical from the
inside: a clean pass. The property is restored instead by deciding language
first, and refusing anything the patterns cannot honestly be said to cover.

Detection is deterministic and dependency-free, which matters more here than
accuracy in the tail. A statistical detector that is right 98% of the time puts a
2% probabilistic hole in a layer whose entire justification is that it is not
probabilistic.

Three passes, cheapest first:

  1. **Script.** Text written mostly in Cyrillic, Greek, Han, Kana, Hangul,
     Arabic, Hebrew, Devanagari or Thai is not English. This is exact.
  2. **Diacritics.** Characters like ã, ç, ñ, ß, è are vanishingly rare in
     English and common in the Latin-script languages nearest this audience.
  3. **Function words.** Short, high-frequency words carry language identity far
     more reliably than content words, which travel between languages — a
     Portuguese question about DeFi is full of English nouns and almost none of
     its function words.

The asymmetry that keeps this safe: refusal requires positive evidence of another
language, never merely the absence of English. "gm", a bare contract address, and
"HYPE funding?" carry no evidence either way and must pass through, because
refusing them would break the agent for its most common inputs while protecting
nobody.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Scripts that settle the question on sight. Value is the language label used in
# the refusal; `None` means "some language, not identified".
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0400, 0x04FF, "cyrillic"),
    (0x0370, 0x03FF, "greek"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x0E00, 0x0E7F, "thai"),
    (0x3040, 0x30FF, "japanese"),
    (0x3400, 0x9FFF, "chinese"),
    (0xAC00, 0xD7AF, "korean"),
)

# Latin letters carrying marks. English uses these only in loanwords ("café"),
# which is why a single one is suggestive rather than decisive.
_DIACRITIC_HINTS: dict[str, tuple[str, ...]] = {
    "portuguese": ("ã", "õ", "ç", "á", "ê", "ô", "í", "ú", "à"),
    "spanish": ("ñ", "¿", "¡", "á", "í", "ó", "ú"),
    "french": ("ç", "è", "é", "ê", "à", "û", "ô", "œ"),
    "german": ("ß", "ä", "ö", "ü"),
}

# Function words that are frequent in one language and rare or absent in English.
# Deliberately excludes words English shares ("no", "a", "in", "son"), which
# would fire on ordinary English input.
_MARKERS: dict[str, frozenset[str]] = {
    "portuguese": frozenset("""
        não nao como qual quais porque por que minha meu seu sua está esta estão
        foi fui tenho tem preciso quero posso pode fazer sobre quando onde
        muito mais uma dos das nas nos pelo pela isso este essa aqui então
        carteira dinheiro saque taxa conta ajuda perdi roubaram invadiram
    """.split()),
    "spanish": frozenset("""
        cómo como cuál cuáles porque por qué mi mis tu su está esta están
        fue tengo tiene necesito quiero puedo puede hacer sobre cuándo dónde
        muy más una los las del pero esto este esa aquí entonces
        cartera billetera dinero retiro comisión cuenta ayuda perdí robaron
    """.split()),
    "french": frozenset("""
        comment quel quelle pourquoi parce mon ma mes votre vos est sont était
        j'ai ai besoin veux peux peut faire sur quand où très plus une les des
        mais ceci cette ici alors portefeuille argent retrait frais compte aide
        perdu volé piraté
    """.split()),
    "german": frozenset("""
        wie welche warum weil mein meine ihr ihre ist sind war habe hat brauche
        will kann machen über wann wo sehr mehr eine der die das den dem aber
        dies diese hier dann geldbörse geld auszahlung gebühr konto hilfe
        verloren gestohlen gehackt
    """.split()),
    "italian": frozenset("""
        come quale perché perche mio mia tuo sua è sono era ho bisogno voglio
        posso può fare quando dove molto più una gli dei ma questo questa qui
        allora portafoglio soldi prelievo commissione conto aiuto perso rubato
    """.split()),
}

# Words naming the two failures this layer exists to catch, in the languages most
# likely to reach it. One of these is decisive on its own, where an ordinary
# function word needs a second.
#
# The asymmetry is the same one that justifies the English patterns being
# over-broad. A terse report — "fui roubado", "wurde gehackt" — carries too few
# function words to clear the general bar, and it is precisely the message that
# must not reach a model. Excludes anything English also uses ("seed", "phishing").
_INCIDENT_MARKERS: dict[str, frozenset[str]] = {
    "portuguese": frozenset("""
        roubado roubada roubaram hackeado hackeada hackearam invadiram invadida
        golpe fraude perdi sumiu sumiram carteira semente chave frase senha
        devolver reembolso estorno
    """.split()),
    "spanish": frozenset("""
        robado robada robaron hackeado hackeada estafa estafaron fraude perdí
        perdi desapareció desaparecio cartera billetera semilla clave frase
        contraseña devolver reembolso
    """.split()),
    "french": frozenset("""
        volé vole volée piraté pirate arnaque fraude perdu disparu portefeuille
        graine clé cle phrase secrète mot rembourser remboursement
    """.split()),
    "german": frozenset("""
        gestohlen gehackt betrug betrogen verloren verschwunden geldbörse
        geldborse brieftasche schlüssel schlussel passwort erstattung
        zurückbuchen
    """.split()),
    "italian": frozenset("""
        rubato rubata hackerato truffa truffato frode perso sparito portafoglio
        seme chiave frase password rimborso
    """.split()),
}

# Weight of one incident word. Set to MIN_MARKERS so a single hit decides.
INCIDENT_WEIGHT = 2


# English vocabulary, used for two jobs that must not diverge: weighing English
# evidence against a rival language, and — at import — stripping any marker that
# is also an English word.
#
# That filter is not tidying. Before it existed, "seed phrase help" was detected
# as French, because `phrase` and `pirate` had been written into the French
# incident list; the input was a textbook seed-phrase solicitation and it would
# have received a language refusal instead of the seed-phrase warning. German
# contributed `die`, `war`, `hat`; Italian `come`, `fare`, `dove`. Every one of
# those turns an ordinary English question into a refusal.
#
# `tests/test_language_gate.py` asserts the intersection stays empty, so the
# filter is a safety net rather than the guarantee.
ENGLISH_VOCAB = frozenset("""
    the is are was were be been being a an and or but if then than that this
    these those i you he she it we they me my your his her its our their
    how what when where why which who whom whose can could do does did done
    have has had not no yes to of in for with on at by from into over under
    about after before between during without within against through
    please help need want should would will shall may might must
    come came go went get got give gave take took make made fare dove die dies
    war hat hats hate era eras pirate pirates phrase phrases password passwords
    seed seeds key keys wallet wallets account accounts fee fees refund refunds
    lost lose stolen steal hack hacked hacker scam scammed fraud phishing
    money fund funds transfer transaction withdraw withdrawal deposit balance
    price rate funding stake staking swap bridge chain block token coin
    now today here there very more most less least much many some any all
    one two three first last next new old good bad high low up down
    time day days week month year hour minute second
    so as at out off on over just only also even still back way thing
    sure work works working use used using try trying know knows think
    see saw look looking find found tell told say said ask asked
    open close start stop send sent receive buy sell trade
    problem issue error wrong right wait waiting sorry thanks thank
    mot son sur ere are
""".split())

# English function words, used only to weigh against a rival — never to require
# a minimum. Their absence proves nothing.
_ENGLISH = frozenset("""
    the is are was were a an and or but my your his her its our their how what
    when where why which who can could do does did have has had not this that
    these those i you it to of in for with on at be been being if then than
    about from into over under please help need want should would will
""".split())

def _without_english(groups: dict[str, frozenset[str]]) -> dict[str, frozenset[str]]:
    """Drop markers that are also English words.

    A marker shared with English is worse than a missing one: it makes English
    input score for a foreign language, and the refusal it produces replaces a
    more specific guardrail that should have fired instead.
    """
    return {lang: frozenset(w for w in words if w not in ENGLISH_VOCAB)
            for lang, words in groups.items()}


_MARKERS = _without_english(_MARKERS)
_INCIDENT_MARKERS = _without_english(_INCIDENT_MARKERS)


# Stripped before scoring: language-neutral noise that would otherwise dilute
# every count. Addresses and tickers are most of a crypto question by volume.
_NOISE = re.compile(
    r"0x[0-9a-fA-F]+|https?://\S+|\b[A-Z]{2,6}\b|[\d.,]+%?|\S+@\S+",
)

# How much evidence is enough. Two independent markers, and strictly more than
# English scores, so a single shared or misspelled word cannot refuse a question.
MIN_MARKERS = 2
# Above this share of non-Latin letters the script test decides on its own.
NON_LATIN_SHARE = 0.30


@dataclass(frozen=True)
class Verdict:
    """What the detector concluded, and on what basis."""

    language: str | None      # None when undetermined
    is_english: bool
    evidence: str

    @property
    def covered(self) -> bool:
        """Whether the English pattern gate can honestly be said to apply."""
        return self.is_english or self.language is None


def _script_of(text: str) -> tuple[str | None, float]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None, 0.0
    counts: dict[str, int] = {}
    for char in letters:
        point = ord(char)
        for low, high, name in _SCRIPT_RANGES:
            if low <= point <= high:
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return None, 0.0
    name = max(counts, key=lambda k: counts[k])
    return name, counts[name] / len(letters)


def _normalise(text: str) -> list[str]:
    cleaned = _NOISE.sub(" ", text.lower())
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", cleaned, re.UNICODE)


def detect(text: str) -> Verdict:
    """Classify `text`, erring towards "undetermined" rather than a wrong guess."""
    if not text or not text.strip():
        return Verdict(None, True, "empty input")

    script, share = _script_of(text)
    if script and share >= NON_LATIN_SHARE:
        return Verdict(script, False, f"{share:.0%} of letters are {script} script")

    words = _normalise(text)
    if not words:
        return Verdict(None, True, "no words to classify after removing addresses and tickers")

    english = sum(1 for w in words if w in _ENGLISH or w in ENGLISH_VOCAB)
    scores = {
        lang: sum(1 for w in words if w in markers) for lang, markers in _MARKERS.items()
    }
    for lang, markers in _INCIDENT_MARKERS.items():
        hits = sum(1 for w in words if w in markers)
        if hits:
            scores[lang] = scores.get(lang, 0) + hits * INCIDENT_WEIGHT
    # A diacritic is worth one marker: strong evidence, but "café" should not
    # decide a sentence on its own.
    lowered = text.lower()
    for lang, hints in _DIACRITIC_HINTS.items():
        if any(h in lowered for h in hints):
            scores[lang] = scores.get(lang, 0) + 1

    best = max(scores, key=lambda k: scores[k])
    if scores[best] >= MIN_MARKERS and scores[best] > english:
        return Verdict(
            best,
            False,
            f"{scores[best]} {best} markers against {english} English",
        )
    if english:
        return Verdict("english", True, f"{english} English markers")
    return Verdict(None, True, "no decisive markers in either direction")
