"""
Deterministic expected phoneme lookup and normalization.

Provides get_expected_phonemes(word) -> list[str]
which returns a normalized list of ARPAbet phoneme tokens (e.g. ['TH', 'R', 'IY']).

Dictionary source:
  - Primary: CMU Pronouncing Dictionary (deterministic reference table for target vocab)
  - Fallback: Deterministic phonetic transcription rules for unknown words

Normalization rules:
  - Input words are stripped of punctuation and converted to lowercase.
  - ARPAbet stress numbers (0, 1, 2) are stripped so all phonemes are clean tokens (e.g., 'IY1' -> 'IY').

Known limitations:
  - Homographs (e.g., 'read' present vs past) only map to the most common pronunciation.
  - Regional variations (e.g., General American vs British RP) default to General American.
"""

import re
from typing import List, Optional

# Deterministic reference vocabulary based on CMUdict (stress numbers stripped)
CMU_REFERENCE_DICT = {
    "three": ["TH", "R", "IY"],
    "think": ["TH", "IH", "NG", "K"],
    "this": ["DH", "IH", "S"],
    "that": ["DH", "AE", "T"],
    "ship": ["SH", "IH", "P"],
    "sheep": ["SH", "IY", "P"],
    "rice": ["R", "AY", "S"],
    "light": ["L", "AY", "T"],
    "right": ["R", "AY", "T"],
    "sound": ["S", "AW", "N", "D"],
    "voice": ["V", "OY", "S"],
    "word": ["W", "ER", "D"],
    "slow": ["S", "L", "OW"],
    "stop": ["S", "T", "AA", "P"],
}

# Rule-based fallback phoneme mappings for grapheme clusters
_RULES = [
    (r"th", ["TH"]),
    (r"sh", ["SH"]),
    (r"ch", ["CH"]),
    (r"ng", ["NG"]),
    (r"ee|ea", ["IY"]),
    (r"oo", ["UW"]),
    (r"igh|ight", ["AY", "T"]),
    (r"ice|ise", ["AY", "S"]),
    (r"a", ["AE"]),
    (r"e", ["EH"]),
    (r"i", ["IH"]),
    (r"o", ["AA"]),
    (r"u", ["AH"]),
    (r"b", ["B"]),
    (r"c|k", ["K"]),
    (r"d", ["D"]),
    (r"f", ["F"]),
    (r"g", ["G"]),
    (r"h", ["HH"]),
    (r"j", ["JH"]),
    (r"l", ["L"]),
    (r"m", ["M"]),
    (r"n", ["N"]),
    (r"p", ["P"]),
    (r"r", ["R"]),
    (r"s", ["S"]),
    (r"t", ["T"]),
    (r"v", ["V"]),
    (r"w", ["W"]),
    (r"y", ["Y"]),
    (r"z", ["Z"]),
]


def normalize_word(word: str) -> str:
    """Strip punctuation and whitespace, lowercase."""
    return re.sub(r"[^a-zA-Z]", "", word).lower()


def normalize_phoneme(phoneme: str) -> str:
    """Normalize a phoneme symbol by stripping stress digits and capitalizing."""
    return re.sub(r"\d+", "", phoneme).strip().upper()


def has_pronunciation(word: str) -> bool:
    """
    Check if a word has reliable pronunciation data (in CMUdict or reference dict).
    Returns False for words that would only get rule-based approximations.
    """
    clean_word = normalize_word(word)
    if not clean_word:
        return False

    if clean_word in CMU_REFERENCE_DICT:
        return True

    try:
        import cmudict
        d = cmudict.dict()
        if clean_word in d:
            return True
    except (ImportError, Exception):
        pass

    return False


def safe_lookup(word: str) -> Optional[List[str]]:
    """
    Safe phoneme lookup that returns None when no reliable pronunciation
    data exists. Used by the drill manager to detect dictionary failures
    and provide user-friendly fallback messages.

    Returns:
        List of phonemes if found, None if no reliable data.
    """
    clean_word = normalize_word(word)
    if not clean_word:
        return None

    if clean_word in CMU_REFERENCE_DICT:
        return [normalize_phoneme(p) for p in CMU_REFERENCE_DICT[clean_word]]

    try:
        import cmudict
        d = cmudict.dict()
        if clean_word in d:
            cmu_phonemes = d[clean_word][0]
            return [normalize_phoneme(p) for p in cmu_phonemes]
    except (ImportError, Exception):
        pass

    return None


def get_expected_phonemes(word: str) -> List[str]:
    """
    Look up the expected phoneme sequence for a given word.

    Returns a list of normalized ARPAbet phoneme strings.
    Never raises an exception; falls back to deterministic rule-based mapping if not in CMUdict.
    """
    clean_word = normalize_word(word)
    if not clean_word:
        return []

    # 1. Fast path: reference dictionary
    if clean_word in CMU_REFERENCE_DICT:
        return [normalize_phoneme(p) for p in CMU_REFERENCE_DICT[clean_word]]

    # 2. CMUDict lookup if nltk/cmudict package is installed
    try:
        import cmudict
        d = cmudict.dict()
        if clean_word in d:
            cmu_phonemes = d[clean_word][0]
            return [normalize_phoneme(p) for p in cmu_phonemes]
    except (ImportError, Exception):
        pass

    # 3. Deterministic rule-based fallback
    phonemes = []
    remaining = clean_word
    while remaining:
        matched = False
        for pattern, mapped in _RULES:
            match = re.match(r"^(" + pattern + ")", remaining)
            if match:
                phonemes.extend(mapped)
                remaining = remaining[len(match.group(0)):]
                matched = True
                break
        if not matched:
            remaining = remaining[1:]  # skip unhandled char

    return [normalize_phoneme(p) for p in phonemes]


# =============================================================================
# ARPAbet to IPA Mapping Table
# =============================================================================

ARPABET_TO_IPA = {
    # Consonants
    "P": "p",
    "B": "b",
    "T": "t",
    "D": "d",
    "K": "k",
    "G": "ɡ",
    "CH": "tʃ",
    "JH": "dʒ",
    "F": "f",
    "V": "v",
    "TH": "θ",
    "DH": "ð",
    "S": "s",
    "Z": "z",
    "SH": "ʃ",
    "ZH": "ʒ",
    "HH": "h",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "L": "l",
    "R": "ɹ",
    "W": "w",
    "Y": "j",
    # Vowels & Diphthongs
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "IH": "ɪ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
}


# =============================================================================
# Physical Articulation Guide (Speech Pathology & Phonetics)
# =============================================================================

ARTICULATION_GUIDE = {
    "TH": {
        "name": "Voiceless dental fricative",
        "ipa": "θ",
        "type": "consonant",
        "tongue": "Place the tip of your tongue gently between your upper and lower front teeth. Do not clamp down.",
        "lips": "Relaxed and slightly parted; keep your jaw loose.",
        "airflow": "Blow a soft, continuous stream of unvoiced air over the tongue tip.",
        "voicing": "Unvoiced — vocal cords stay still (whisper airflow only).",
        "drill_tip": "Look in a mirror to ensure the tongue tip visibly peeks out between your incisors.",
        "drift_trap": "Do not pull the tongue behind your teeth (which produces 'S') or stop the air (which sounds like 'T').",
    },
    "DH": {
        "name": "Voiced dental fricative",
        "ipa": "ð",
        "type": "consonant",
        "tongue": "Tongue tip gently between teeth, in the identical position as /θ/.",
        "lips": "Parted, relaxed jaw.",
        "airflow": "Continuous air stream over tongue tip with throat resonance.",
        "voicing": "Voiced — vocal cords vibrate; feel a distinct hum in your throat.",
        "drift_trap": "Avoid pressing hard against the hard palate (which turns it into a 'D').",
    },
    "R": {
        "name": "Alveolar / Postalveolar approximant",
        "ipa": "ɹ",
        "type": "consonant",
        "tongue": "Curl the tip of your tongue backward (retroflex) or bunch the tongue body high without touching the roof.",
        "lips": "Slightly flared and rounded forward.",
        "airflow": "Smooth continuous voiced airflow passing through the center channel.",
        "voicing": "Voiced — vocal cords vibrate continuously.",
        "drill_tip": "Make sure your tongue floats freely without making contact with the upper gum ridge.",
        "drift_trap": "If the tongue tip touches the ridge, it sounds like an 'L' or a tapped 'D'.",
    },
    "L": {
        "name": "Alveolar lateral approximant",
        "ipa": "l",
        "type": "consonant",
        "tongue": "Press the tip of your tongue firmly against the bumpy gum ridge (alveolar ridge) right behind upper front teeth.",
        "lips": "Neutral to slightly smiling.",
        "airflow": "Air flows around both sides (laterally) of the tongue while the center is blocked.",
        "voicing": "Voiced — vocal cords active.",
        "drill_tip": "Hold the tongue tip firmly against the gum ridge while continuing to voice.",
        "drift_trap": "Do not round lips or curl tongue backward, which drifts into 'R' or 'W'.",
    },
    "SH": {
        "name": "Voiceless postalveolar fricative",
        "ipa": "ʃ",
        "type": "consonant",
        "tongue": "Arch the blade of your tongue high toward the hard palate just behind the gum ridge.",
        "lips": "Pucker and flare your lips forward into a soft trumpet/oval shape.",
        "airflow": "Blow a wide, turbulent stream of unvoiced air (the universal 'shush' sound).",
        "voicing": "Unvoiced — purely acoustic white noise.",
        "drill_tip": "Keep your lips distinctly rounded forward to create the deeper resonant cavity.",
        "drift_trap": "Pulling lips back flat turns 'SH' into a sharp 'S'.",
    },
    "S": {
        "name": "Voiceless alveolar fricative",
        "ipa": "s",
        "type": "consonant",
        "tongue": "Tongue tip placed directly behind upper front teeth on the alveolar ridge, creating a tiny narrow slit.",
        "lips": "Lips spread slightly, teeth very close together.",
        "airflow": "Direct a sharp, high-velocity jet of unvoiced air straight against the upper incisors.",
        "voicing": "Unvoiced — high frequency hiss.",
        "drill_tip": "Keep the teeth almost touching and the tongue groove razor-thin.",
        "drift_trap": "Pushing the tongue between teeth produces a lisp ('TH'). Keep it tucked right behind.",
    },
    "CH": {
        "name": "Voiceless postalveolar affricate",
        "ipa": "tʃ",
        "type": "consonant",
        "tongue": "Press tongue firmly against alveolar ridge to stop airflow like 'T', then instantly explode into 'SH'.",
        "lips": "Rounded and projected forward.",
        "airflow": "Sudden stop followed immediately by turbulent friction.",
        "voicing": "Unvoiced.",
        "drift_trap": "Ensure there is a complete air seal first so it does not sound like a weak 'SH'.",
    },
    "JH": {
        "name": "Voiced postalveolar affricate",
        "ipa": "dʒ",
        "type": "consonant",
        "tongue": "Stop air on the ridge like 'D', then release into a voiced buzzing 'ZH'.",
        "lips": "Rounded forward.",
        "airflow": "Voiced burst of friction.",
        "voicing": "Voiced — throat vibrates throughout the release.",
        "drift_trap": "Don't unvoice it into 'CH'.",
    },
    "V": {
        "name": "Voiced labiodental fricative",
        "ipa": "v",
        "type": "consonant",
        "tongue": "Relaxed and resting flat inside the lower jaw.",
        "lips": "Rest top front incisors lightly on the wet inside margin of the lower lip.",
        "airflow": "Continuous air forced through the teeth-lip contact.",
        "voicing": "Voiced — you should feel a vigorous buzzing tickle on your bottom lip.",
        "drift_trap": "Don't press both lips together (sounds like 'B') or round both lips (sounds like 'W').",
    },
    "F": {
        "name": "Voiceless labiodental fricative",
        "ipa": "f",
        "type": "consonant",
        "tongue": "Relaxed flat on floor of mouth.",
        "lips": "Top incisors resting lightly on lower lip (identical posture to /v/).",
        "airflow": "Smooth unvoiced friction.",
        "voicing": "Unvoiced — whisper breath only.",
        "drift_trap": "Keep teeth on the lip; do not purse both lips together into 'P'.",
    },
    "W": {
        "name": "Voiced labio-velar approximant",
        "ipa": "w",
        "type": "consonant",
        "tongue": "Back of the tongue rises high toward the soft palate (velum).",
        "lips": "Pucker lips tightly into a tiny circle (like whistling), then glide outwards.",
        "airflow": "Smooth voiced vocalic glide.",
        "voicing": "Voiced.",
        "drift_trap": "Never touch your top teeth to your lower lip (that produces 'V'). Keep lips purely circular.",
    },
    "T": {
        "name": "Voiceless alveolar plosive",
        "ipa": "t",
        "type": "consonant",
        "tongue": "Tap tongue tip firmly against alveolar ridge behind top teeth, building acoustic pressure.",
        "lips": "Parted, relaxed.",
        "airflow": "Sudden crisp release of unvoiced air.",
        "voicing": "Unvoiced.",
        "drift_trap": "Don't protrude the tongue between teeth (which sounds like 'TH'). Tap the gum ridge.",
    },
    "D": {
        "name": "Voiced alveolar plosive",
        "ipa": "d",
        "type": "consonant",
        "tongue": "Tongue tip seals firmly against alveolar ridge.",
        "lips": "Parted, relaxed.",
        "airflow": "Voiced explosive burst upon dropping tongue.",
        "voicing": "Voiced.",
        "drift_trap": "Ensure throat cords vibrate right as the seal breaks.",
    },
    "K": {
        "name": "Voiceless velar plosive",
        "ipa": "k",
        "type": "consonant",
        "tongue": "Back of tongue arches up to firmly seal against the soft palate.",
        "lips": "Parted in position of upcoming vowel.",
        "airflow": "Sudden unvoiced burst when back of tongue drops.",
        "voicing": "Unvoiced.",
        "drift_trap": "Don't let the air leak prematurely; make the pop clean.",
    },
    "G": {
        "name": "Voiced velar plosive",
        "ipa": "ɡ",
        "type": "consonant",
        "tongue": "Back of tongue seals against soft palate, identical to /k/.",
        "lips": "Parted.",
        "airflow": "Voiced burst from the throat.",
        "voicing": "Voiced.",
        "drift_trap": "Voicing starts before release.",
    },
    "P": {
        "name": "Voiceless bilabial plosive",
        "ipa": "p",
        "type": "consonant",
        "tongue": "Relaxed.",
        "lips": "Press upper and lower lips together firmly, building air pressure.",
        "airflow": "Crisp puff of unvoiced air upon releasing lips.",
        "voicing": "Unvoiced.",
        "drift_trap": "Ensure lips seal completely before the burst.",
    },
    "B": {
        "name": "Voiced bilabial plosive",
        "ipa": "b",
        "type": "consonant",
        "tongue": "Relaxed.",
        "lips": "Press both lips together firmly, identical to /p/.",
        "airflow": "Voiced pop when lips part.",
        "voicing": "Voiced.",
        "drift_trap": "Engage vocal cords as lips open.",
    },
    "M": {
        "name": "Voiced bilabial nasal",
        "ipa": "m",
        "type": "consonant",
        "tongue": "Relaxed.",
        "lips": "Lips closed lightly together.",
        "airflow": "All voiced sound resonates entirely through nasal cavity.",
        "voicing": "Voiced.",
        "drift_trap": "Keep lips sealed until transitioning to next sound.",
    },
    "N": {
        "name": "Voiced alveolar nasal",
        "ipa": "n",
        "type": "consonant",
        "tongue": "Tongue tip seals against the alveolar ridge.",
        "lips": "Parted.",
        "airflow": "Voiced airflow escapes through the nose.",
        "voicing": "Voiced.",
        "drift_trap": "Do not let air escape through the mouth until tongue drops.",
    },
    "NG": {
        "name": "Voiced velar nasal",
        "ipa": "ŋ",
        "type": "consonant",
        "tongue": "Back of tongue seals against soft palate, blocking the mouth completely.",
        "lips": "Parted comfortably.",
        "airflow": "Air flows exclusively through the nasal passage.",
        "voicing": "Voiced.",
        "drift_trap": "Avoid adding a hard 'G' release (/ŋɡ/) unless required.",
    },
    "Z": {
        "name": "Voiced alveolar fricative",
        "ipa": "z",
        "type": "consonant",
        "tongue": "Tongue tip behind upper incisors on alveolar ridge, same as /s/.",
        "lips": "Slightly spread, teeth close.",
        "airflow": "Voiced high-frequency friction hiss.",
        "voicing": "Voiced — throat must buzz like a bee.",
        "drift_trap": "Don't let the voicing drop into an unvoiced 'S'.",
    },
    "ZH": {
        "name": "Voiced postalveolar fricative",
        "ipa": "ʒ",
        "type": "consonant",
        "tongue": "Blade of tongue arched toward hard palate, identical to /ʃ/.",
        "lips": "Rounded forward.",
        "airflow": "Voiced continuous friction (as in 'measure', 'vision').",
        "voicing": "Voiced.",
        "drift_trap": "Keep vocal cords vibrating throughout.",
    },
    "HH": {
        "name": "Voiceless glottal fricative",
        "ipa": "h",
        "type": "consonant",
        "tongue": "Takes shape of the following vowel.",
        "lips": "Takes shape of following vowel.",
        "airflow": "Unvoiced whisper of breath through open vocal cords.",
        "voicing": "Unvoiced.",
        "drift_trap": "Don't constrict throat too tightly; keep it light.",
    },
    "Y": {
        "name": "Voiced palatal approximant",
        "ipa": "j",
        "type": "consonant",
        "tongue": "Arch front of tongue high toward hard palate, then glide into vowel.",
        "lips": "Neutral to spread.",
        "airflow": "Smooth voiced glide.",
        "voicing": "Voiced.",
        "drift_trap": "Do not create friction (which sounds like 'JH').",
    },
    # Vowels
    "IY": {
        "name": "Close front unrounded vowel ('ee')",
        "ipa": "i",
        "type": "vowel",
        "tongue": "Front of tongue arched very high and forward near hard palate.",
        "lips": "Spread wide into a firm smile.",
        "airflow": "Tense, clear voiced tone.",
        "voicing": "Voiced.",
        "drift_trap": "Keep the smile tight and tense; don't relax into /ɪ/ ('ship').",
    },
    "IH": {
        "name": "Near-close front unrounded vowel ('ih')",
        "ipa": "ɪ",
        "type": "vowel",
        "tongue": "Tongue slightly lower and further back than /i/.",
        "lips": "Relaxed and neutral (not smiling wide).",
        "airflow": "Short, relaxed, lax voiced tone.",
        "voicing": "Voiced.",
        "drift_trap": "Do not tense your jaw into 'ee'. Keep it short and effortless.",
    },
    "ER": {
        "name": "R-colored vowel (rhotic)",
        "ipa": "ɝ",
        "type": "vowel",
        "tongue": "Sides of tongue pressed against top back molars, tip curled back slightly.",
        "lips": "Slightly flared and rounded forward.",
        "airflow": "Resonant vocalized rhotic acoustic tone.",
        "voicing": "Voiced.",
        "drift_trap": "Keep the tongue body tight and bunched; do not let it drop flat into British /ɜː/.",
    },
    "UW": {
        "name": "Close back rounded vowel ('oo')",
        "ipa": "u",
        "type": "vowel",
        "tongue": "Back of tongue raised high toward soft palate.",
        "lips": "Puckered tightly forward into a small circle.",
        "airflow": "Deep, resonant voiced tone.",
        "voicing": "Voiced.",
        "drift_trap": "Ensure lips are tightly rounded.",
    },
    "UH": {
        "name": "Near-close near-back vowel ('uh' in book)",
        "ipa": "ʊ",
        "type": "vowel",
        "tongue": "Back of tongue moderately high, slightly relaxed.",
        "lips": "Loosely rounded, relaxed jaw.",
        "airflow": "Short lax vocal sound.",
        "voicing": "Voiced.",
        "drift_trap": "Do not over-pucker into /u/ ('too').",
    },
    "AE": {
        "name": "Near-open front unrounded vowel ('cat')",
        "ipa": "æ",
        "type": "vowel",
        "tongue": "Low and forward in the mouth with sides slightly touching lower molars.",
        "lips": "Open wide, corners pulled back.",
        "airflow": "Open, bright voiced vowel.",
        "voicing": "Voiced.",
        "drift_trap": "Drop the jaw lower than for 'EH'.",
    },
    "EH": {
        "name": "Open-mid front unrounded vowel ('bed')",
        "ipa": "ɛ",
        "type": "vowel",
        "tongue": "Mid-height in the front of mouth.",
        "lips": "Parted moderately.",
        "airflow": "Short voiced vowel.",
        "voicing": "Voiced.",
        "drift_trap": "Do not lower jaw too far into 'AE' or raise it to 'IH'.",
    },
    "AH": {
        "name": "Open-mid back unrounded vowel ('cup')",
        "ipa": "ʌ",
        "type": "vowel",
        "tongue": "Resting comfortably low in the center of the mouth.",
        "lips": "Neutral, relaxed.",
        "airflow": "Short neutral vocalization.",
        "voicing": "Voiced.",
        "drift_trap": "Keep jaw completely relaxed.",
    },
    "AA": {
        "name": "Open back unrounded vowel ('father')",
        "ipa": "ɑ",
        "type": "vowel",
        "tongue": "Flattened all the way down in the bottom of the mouth.",
        "lips": "Wide open, unrounded.",
        "airflow": "Deep, resonant open chest tone.",
        "voicing": "Voiced.",
        "drift_trap": "Drop your jaw completely as if at the doctor saying 'ah'.",
    },
    "AO": {
        "name": "Open-mid back rounded vowel ('law')",
        "ipa": "ɔ",
        "type": "vowel",
        "tongue": "Back of tongue raised moderately low.",
        "lips": "Rounded into an open oval.",
        "airflow": "Full resonant tone.",
        "voicing": "Voiced.",
        "drift_trap": "Round your lips slightly more than for 'AA'.",
    },
    "AY": {
        "name": "Diphthong /aɪ/ ('rice', 'light')",
        "ipa": "aɪ",
        "type": "vowel",
        "tongue": "Start low and open on /ɑ/, then smoothly glide up toward /ɪ/.",
        "lips": "Start open, then spread toward smile.",
        "airflow": "Two-stage smooth vocal glide.",
        "voicing": "Voiced.",
        "drift_trap": "Make sure to finish the glide all the way to the high front position.",
    },
    "AW": {
        "name": "Diphthong /aʊ/ ('sound', 'cow')",
        "ipa": "aʊ",
        "type": "vowel",
        "tongue": "Start open on /ɑ/, then glide back and up toward /ʊ/.",
        "lips": "Start wide open, then pucker into a tight circle.",
        "airflow": "Smooth closing vocal glide.",
        "voicing": "Voiced.",
        "drift_trap": "Round your lips firmly at the end of the sound.",
    },
    "EY": {
        "name": "Diphthong /eɪ/ ('day', 'say')",
        "ipa": "eɪ",
        "type": "vowel",
        "tongue": "Start at mid-front /e/, glide upward into /ɪ/.",
        "lips": "Neutral to spread.",
        "airflow": "Upward vocal glide.",
        "voicing": "Voiced.",
        "drift_trap": "Keep it a diphthong; don't truncate to a flat monophthong.",
    },
    "OW": {
        "name": "Diphthong /oʊ/ ('slow', 'go')",
        "ipa": "oʊ",
        "type": "vowel",
        "tongue": "Start mid-back /o/, glide up toward /ʊ/.",
        "lips": "Start medium round, tighten circle at the end.",
        "airflow": "Continuous closing glide.",
        "voicing": "Voiced.",
        "drift_trap": "Pucker lips as you finish the syllable.",
    },
    "OY": {
        "name": "Diphthong /ɔɪ/ ('voice', 'boy')",
        "ipa": "ɔɪ",
        "type": "vowel",
        "tongue": "Start back at /ɔ/, glide forward and high to /ɪ/.",
        "lips": "Start rounded, glide to spread smile.",
        "airflow": "Dramatic back-to-front acoustic shift.",
        "voicing": "Voiced.",
        "drift_trap": "Make the transition distinct between rounded start and smiling finish.",
    },
}


# =============================================================================
# Categorized Practice Tracks (Minimal Pairs)
# =============================================================================

MINIMAL_PAIRS_TRACKS = [
    {
        "id": "th-vs-t",
        "name": "TH vs T Contrast",
        "description": "Master the dental fricative /θ/ vs alveolar stop /t/",
        "focus_phonemes": ["TH", "DH", "T"],
        "words": ["three", "think", "that", "this", "tree", "tank"],
    },
    {
        "id": "r-vs-l",
        "name": "R vs L Contrast",
        "description": "Differentiate curling approximant /ɹ/ from tongue-ridge /l/",
        "focus_phonemes": ["R", "L"],
        "words": ["rice", "light", "right", "lake", "rake", "lead"],
    },
    {
        "id": "sh-vs-s",
        "name": "SH vs S Contrast",
        "description": "Hissing incisor stream /s/ vs rounded postalveolar /ʃ/",
        "focus_phonemes": ["SH", "S"],
        "words": ["ship", "sheep", "sip", "seat", "sea", "shine"],
    },
    {
        "id": "v-vs-w",
        "name": "V vs W Contrast",
        "description": "Teeth-to-lip buzz /v/ vs circular lip glide /w/",
        "focus_phonemes": ["V", "W", "F"],
        "words": ["voice", "wave", "vine", "wine", "vest", "west"],
    },
]


def get_ipa_transcription(phonemes: List[str]) -> str:
    """
    Convert a list of ARPAbet phoneme tokens into clean IPA notation.
    Example: ['TH', 'R', 'IY'] -> '/θɹi/'
    """
    if not phonemes:
        return ""
    ipa_tokens = [ARPABET_TO_IPA.get(normalize_phoneme(p), p.lower()) for p in phonemes]
    return f"/{''.join(ipa_tokens)}/"


def get_phoneme_articulation(phoneme: str) -> Dict[str, str]:
    """
    Retrieve speech pathology articulation guidance for an ARPAbet phoneme token.
    Falls back to safe default guidance if token is unknown.
    """
    norm = normalize_phoneme(phoneme)
    if norm in ARTICULATION_GUIDE:
        return ARTICULATION_GUIDE[norm]

    # Default fallback guidance
    ipa_char = ARPABET_TO_IPA.get(norm, norm.lower())
    is_vowel = norm in ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW")
    return {
        "name": f"Phoneme /{ipa_char}/",
        "ipa": ipa_char,
        "type": "vowel" if is_vowel else "consonant",
        "tongue": "Keep your tongue relaxed and centered.",
        "lips": "Position lips naturally for the sound.",
        "airflow": "Breathe smoothly through the vocal tract.",
        "voicing": "Voiced" if is_vowel else "Natural voicing",
        "drill_tip": f"Listen closely to the model and mimic the exact mouth shape.",
        "drift_trap": "Listen to the Rime slow model to isolate the target sound.",
    }


def get_word_phoneme_breakdown(word: str) -> Dict[str, Any]:
    """
    Generate a full diagnostic breakdown for any English word:
      - Clean word
      - Expected ARPAbet phonemes
      - Clean IPA transcription
      - Per-phoneme articulation and IPA details
      - Dictionary lookup status
    """
    clean = normalize_word(word)
    phonemes = get_expected_phonemes(clean)
    ipa = get_ipa_transcription(phonemes)
    in_dict = has_pronunciation(clean)

    breakdown = []
    for p in phonemes:
        guide = get_phoneme_articulation(p)
        breakdown.append({
            "phoneme": p,
            "ipa": guide.get("ipa", ARPABET_TO_IPA.get(p, p.lower())),
            "name": guide.get("name", p),
            "type": guide.get("type", "consonant"),
            "articulation": guide,
        })

    return {
        "word": clean,
        "phonemes": phonemes,
        "ipa": ipa,
        "in_dict": in_dict,
        "breakdown": breakdown,
    }

