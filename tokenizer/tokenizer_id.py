import re

# ====== MORFEM INDONESIA ======

PROCLITICS = ["ku"]
ENCLITICS = ["ku", "mu", "nya"]
PARTICLES = ["lah", "kah", "pun"]

COMBINATIONS = [
    ("nyapun", ["nya", "pun"]),
    ("nyalah", ["nya", "lah"]),
    ("nyakah", ["nya", "kah"]),
]

# ====== LOAD LEXICON ======

def load_lexicon(path):
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            word, pos = line.strip().split()
            lex[word.lower()] = pos.lower()
    return lex


# ====== STAGE 1: ORTHOGRAPHIC TOKENIZER (TreeTagger-like) ======

def ortho_tokenize(text):
    """
    Rough equivalent of utf8-tokenize.perl
    """
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


# ====== STAGE 2: MORPHOLOGICAL TOKENIZER ======

def tokenize_word(word, lexicon):
    original = word
    lower = word.lower()

    proclitic = None
    enclitics = []

    # --- proclitic (ku-) ---
    if lower.startswith("ku") and len(lower) > 2:
        proclitic = "ku"
        lower = lower[2:]

    # --- suffix combinations (nyapun, etc.) ---
    for suf, parts in COMBINATIONS:
        if lower.endswith(suf) and len(lower) > len(suf):
            enclitics = parts[:]
            lower = lower[:-len(suf)]
            break
    else:
        # --- single suffix ---
        for suf in ENCLITICS + PARTICLES:
            if lower.endswith(suf) and len(lower) > len(suf):
                enclitics = [suf]
                lower = lower[:-len(suf)]
                break

    host = lower
    pos = lexicon.get(host)

    # --- linguistic decision ---
    if pos == "v":
        tokens = []
        if proclitic:
            tokens.append(proclitic)
        tokens.append(host)
        tokens.extend(enclitics)
        return tokens

    # rollback
    return [original]


# ====== FULL TOKENIZER PIPELINE ======

def tokenize(text, lexicon):
    tokens = ortho_tokenize(text)
    final_tokens = []
    for t in tokens:
        final_tokens.extend(tokenize_word(t, lexicon))
    return final_tokens


# ====== TREE TAGGER INPUT FORMAT ======

def treetagger_format(tokens):
    return "\n".join(tokens) + "\n"
