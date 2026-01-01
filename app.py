import streamlit as st
import pandas as pd
import os
import re
import sys
import subprocess
import xml.etree.ElementTree as ET
from io import BytesIO
import zipfile

# =====================
# JAPANESE
# =====================
from fugashi import Tagger as FugashiTagger

@st.cache_resource
def get_japanese_tagger():
    try:
        return FugashiTagger()
    except Exception as e:
        st.warning(f"Japanese tagger failed: {e}")
        return None

JP_TAGGER = get_japanese_tagger()

def run_tagger_japanese(text):
    if not JP_TAGGER:
        return []
    out = []
    for node in JP_TAGGER.parseToNodeList(text):
        if node.surface:
            pos = node.feature.pos1
            lemma = node.feature.lemma or node.surface
            out.append(f"{node.surface}\t{pos}\t{lemma}")
    return out

# =====================
# ENGLISH
# =====================
from textblob import TextBlob

@st.cache_resource
def init_textblob():
    try:
        import nltk
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        subprocess.check_call([sys.executable, '-m', 'textblob.download_corpora'])
    return True

EN_READY = init_textblob()

def run_tagger_english(text):
    if not EN_READY:
        return []
    blob = TextBlob(text)
    return [f"{w}\t{p}\t{w.lemmatize()}" for w, p in blob.tags]

# =====================
# INDONESIAN (TreeTagger native)
# =====================
TT_BIN = "treetagger/bin/tree-tagger"
TT_CMD = "treetagger/cmd/tag-indonesian"
TT_PAR = "treetagger/lib/indonesian_v311225.par"

@st.cache_resource
def ensure_exec():
    if sys.platform.startswith('linux'):
        try:
            subprocess.run(["chmod", "+x", TT_BIN], check=False)
            subprocess.run(["chmod", "+x", TT_CMD], check=False)
            subprocess.run("chmod +x treetagger/cmd/*.perl", shell=True, check=False)
        except Exception:
            pass
    return True

ensure_exec()

def run_tagger_indonesian(text, use_mwu=False):
    if not text.strip():
        return []

    if not os.path.exists(TT_PAR):
        return ["ERROR\tERROR\tParameter file not found"]

    cmd = ["bash", TT_CMD]
    if use_mwu:
        cmd.append("-mwu")

    try:
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        out, err = p.communicate(text)
        if p.returncode != 0:
            return [f"ERROR\tERROR\t{err.strip()}"]
        return [l for l in out.splitlines() if l.strip()]
    except Exception as e:
        return [f"ERROR\tERROR\t{e}"]

# =====================
# XML-safe processing
# =====================
def process_text(text, lang, tagger):
    try:
        root = ET.fromstring(f"<r>{text}</r>")
        for el in root.iter():
            if el.text and el.text.strip():
                tagged = tagger(el.text)
                el.text = '\n' + '\n'.join(tagged) + '\n'
        xml = ET.tostring(root, encoding='unicode')
        return re.sub(r'^<r>|</r>$', '', xml)
    except ET.ParseError:
        tagged = tagger(text)
        return f'<text lang="{lang}">\n' + '\n'.join(tagged) + '\n</text>'

# =====================
# UI
# =====================
def tokenizer_ui(name, code, tagger, mwu=False):
    st.header(f"{name} ({code})")
    mode = st.radio("Input mode", ["Direct input", "File upload"], horizontal=True)

    if mode == "Direct input":
        txt = st.text_area("Text", height=200)
        if st.button("Tag"):
            res = tagger(txt) if code != 'ID' else tagger(txt, mwu)
            rows = [r.split('\t') for r in res if '\t' in r]
            if rows:
                df = pd.DataFrame(rows, columns=["Token", "POS", "Lemma"])
                st.dataframe(df, use_container_width=True)

    else:
        files = st.file_uploader("Upload txt/xml", type=['txt', 'xml'], accept_multiple_files=True)
        if files and st.button("Process files"):
            out = {}
            for f in files:
                text = f.read().decode('utf-8')
                tagged = process_text(text, code, tagger if code != 'ID' else lambda t: tagger(t, mwu))
                out[f.name] = f"<?xml version='1.0' encoding='UTF-8'?>\n{tagged}"
            buf = BytesIO()
            with zipfile.ZipFile(buf, 'w') as z:
                for k, v in out.items():
                    z.writestr(k.replace('.txt', '_tagged.xml'), v)
            st.download_button("Download ZIP", buf.getvalue(), "tagged.zip")

# =====================
# MAIN
# =====================
def main():
    st.set_page_config("Multilingual Tagger", layout="wide")
    st.title("🌐 Multilingual Tokenizer & Tagger")

    lang = st.sidebar.radio("Language", ['JAPANESE', 'ENGLISH', 'INDONESIAN'])

    if lang == 'JAPANESE':
        tokenizer_ui("Japanese", "JP", run_tagger_japanese)
    elif lang == 'ENGLISH':
        tokenizer_ui("English", "EN", run_tagger_english)
    else:
        use_mwu = st.sidebar.checkbox("Use MWU", False)
        tokenizer_ui("Indonesian", "ID", run_tagger_indonesian, use_mwu)

if __name__ == '__main__':
    main()
