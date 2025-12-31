import streamlit as st
import pandas as pd
import os
import zipfile
import re
from io import BytesIO
import subprocess
import sys
import xml.etree.ElementTree as ET

# Import language libraries
from fugashi import Tagger as FugashiTagger # for Japanese
from textblob import TextBlob # for English
import stanza # for Indonesian
# Sastrawi import moved to runtime to allow auto-install

# --- Global Configuration and State Management ---

# --- JAPANESE TOKENIZER ---
@st.cache_resource
def get_japanese_tokenizer():
    """Initializes and returns the Fugashi Tagger with unidic-lite."""
    try:
        tagger = FugashiTagger()
        return tagger
    except Exception as e:
        # Warning instead of error to allow app to run if just one lang fails
        print(f"Error initializing Japanese Tokenizer: {e}") 
        return None

# --- ENGLISH TEXTBLOB SETUP ---
@st.cache_resource
def initialize_english_textblob():
    """Ensures TextBlob data is downloaded."""
    try:
        import nltk
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading TextBlob data...")
        subprocess.check_call([sys.executable, "-m", "textblob.download_corpora"])
    
    return True

import requests

# --- DEPENDENCY AUTO-INSTALLER ---
# PySastrawi removed as per user request to use wordlist instead.

# --- INDONESIAN WORDLIST SETUP ---
@st.cache_resource
def get_indonesian_dictionary():
    """
    Downloads and parses the Indonesian token-tag-lemma dictionary.
    Returns a dictionary mapping token -> lemma.
    """
    url = "https://raw.githubusercontent.com/prihantoro-corpus/tokeniser-tagger/main/ID-token-tag-lemma.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        lemma_dict = {}
        # Expected format: token tag lemma
        # We will map token (lower) -> lemma
        # If duplicates exist, later entries will overwrite earlier ones (simple approach)
        
        lines = response.text.strip().split('\n')
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                # Assuming first column is token, last is lemma. 
                # Tag (middle) is ignored for now as we use Stanza tags.
                token = parts[0]
                lemma = parts[-1]
                
                # key by lowercase token for robustness
                lemma_dict[token.lower()] = lemma
                
        return lemma_dict
    except Exception as e:
        st.error(f"Failed to load Indonesian dictionary: {e}")
        return {}

# --- INDONESIAN STANZA SETUP ---
@st.cache_resource
def get_stanza_pipeline():
    """
    Initializes the Stanza pipeline for Indonesian.
    This handles downloading the model if it's not present.
    """
    try:
        # Download stanza model if not exists. 
        # 'processors': 'tokenize,pos' - lemma processor removed as we use dictionary
        stanza.download('id', processors='tokenize,pos', verbose=False)
        
        # Initialize the pipeline
        nlp = stanza.Pipeline('id', processors='tokenize,pos', use_gpu=False, verbose=False)
        
        return nlp
    except Exception as e:
        print(f"Error initializing Indonesian Stanza Pipeline. Error: {e}")
        return None

# Global Variables (Lazy loading recommended, but here we init for cache)
JAPANESE_TAGGER = get_japanese_tokenizer()
ENGLISH_TAGGER_READY = initialize_english_textblob()
# We initialize stanza on demand or globally? Globally is okay if cached.
# However, for startup speed, we might want to do it only if selected.
# But st.cache_resource handles the singleton pattern nicely.
INDONESIAN_RESOURCES = None # Will be loaded if needed

# --- Core Processing Functions ---

# --- JAPANESE PROCESSING ---
def run_tagger_japanese(text):
    if JAPANESE_TAGGER is None:
        return []
    nodes = JAPANESE_TAGGER.parseToNodeList(text)
    results = []
    for node in nodes:
        if node.surface:
            token = node.surface
            # Pos1 is usually top level POS
            pos = node.feature.pos1
            lemma = node.feature.lemma if node.feature.lemma else token
            results.append(f"{token}\t{pos}\t{lemma}")
    return results

# --- ENGLISH PROCESSING ---
def run_tagger_english(text):
    if not ENGLISH_TAGGER_READY:
        return []
    
    blob = TextBlob(text)
    results = []
    for token, pos_tag in blob.tags:
        lemma = token.lemmatize() # TextBlob (Word) has lemmatize method
        results.append(f"{token}\t{pos_tag}\t{lemma}")
    return results

# --- INDONESIAN PROCESSING ---
def run_tagger_indonesian(text):
    global INDONESIAN_RESOURCES
    if INDONESIAN_RESOURCES is None:
        with st.spinner("Loading Indonesian Model & Dictionary..."):
            stanza_pipeline = get_stanza_pipeline()
            lemma_dict = get_indonesian_dictionary()
            INDONESIAN_RESOURCES = (stanza_pipeline, lemma_dict)
    
    stanza_pipeline, lemma_dict = INDONESIAN_RESOURCES
    
    if stanza_pipeline is None:
        return ["Error: Model failed to load."]

    # Stanza processes the text into a Document object
    doc = stanza_pipeline(text)
    
    results = []
    # Stanza structure: doc -> sentences -> words
    for sent in doc.sentences:
        for word in sent.words:
            # Output: token \t POS \t lemma
            # Use dictionary for lookup (case-insensitive)
            token_text = word.text
            lemma = lemma_dict.get(token_text.lower(), token_text)
            
            results.append(f"{token_text}\t{word.upos}\t{lemma}")
            
    return results


def process_xml_content(xml_string, lang_code, tagger_function):
    """
    Parses the XML string and tags ONLY the plain text content, 
    preserving all XML tags and attributes.
    """
    temp_root_tag = 'TEMP_WRAPPER'
    cleaned_xml_string = re.sub(r'<\?xml[^>]*\?>', '', xml_string, flags=re.IGNORECASE).strip()
    wrapped_xml = f"<{temp_root_tag}>{cleaned_xml_string}</{temp_root_tag}>"
    
    try:
        root = ET.fromstring(wrapped_xml)
    except ET.ParseError as e:
        st.warning(f"Input failed XML parsing ({e}). Processing as raw text only.")
        tagged_lines = tagger_function(xml_string)
        return f'<text lang="{lang_code}">\n' + "\n".join(tagged_lines) + '\n</text>'
        
    def traverse_and_tag(element):
        if element.text and element.text.strip():
            tagged_lines = tagger_function(element.text)
            element.text = '\n' + '\n'.join(tagged_lines) + '\n'

        for child in element:
            traverse_and_tag(child)

        if element.tail and element.tail.strip():
            tagged_lines = tagger_function(element.tail)
            element.tail = '\n' + '\n'.join(tagged_lines) + '\n'

    traverse_and_tag(root)
    
    full_xml = ET.tostring(root, encoding='unicode')
    full_xml = re.sub(r'^<TEMP_WRAPPER>', '', full_xml)
    full_xml = re.sub(r'</TEMP_WRAPPER>$', '', full_xml).strip()
    
    return full_xml

def process_text(text, lang_code, tagger_function):
    return process_xml_content(text, lang_code, tagger_function)


# --- XML Creation and Zipping ---
def create_output_file_content(processed_xml, original_filename):
    base_filename = os.path.splitext(original_filename)[0]
    sanitized_base_name = re.sub(r' \(\d+\)$', '', base_filename).strip()
    final_output = f'<?xml version="1.0" encoding="UTF-8"?>\n{processed_xml}'
    return final_output, f"{sanitized_base_name}_tagged.xml"

def create_zip_archive(output_data):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for original_name, processed_content in output_data.items():
            final_content, xml_name = create_output_file_content(processed_content, original_name)
            zf.writestr(xml_name, final_content.encode('utf-8'))
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# --- Streamlit UI Components ---

def tokenizer_interface(lang_name, lang_code, tagger_function):
    st.header(f"🌎 {lang_name} Tokenizer ({lang_code})")
    st.markdown("---")
    
    # Input Method Selection
    input_method = st.radio("Choose Input Method:", ["📂 File Upload", "✍️ Direct Input"], horizontal=True)
    st.markdown("---")

    # --- MODE 1: FILE UPLOAD ---
    if input_method == "📂 File Upload":
        st.subheader("Upload Text or XML Files")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'xml'],
            accept_multiple_files=True,
            key=f"uploader_{lang_code}"
        )

        if uploaded_files:
            if st.button(f"Start Tagging", key=f"btn_upload_{lang_code}"):
                output_data = {}
                progress_bar = st.progress(0, text="Processing files...")
                
                for i, uploaded_file in enumerate(uploaded_files):
                    filename = uploaded_file.name
                    try:
                        content_bytes = uploaded_file.read()
                        text = content_bytes.decode('utf-8')
                        processed_xml = process_text(text, lang_code, tagger_function)
                        output_data[filename] = processed_xml
                        st.success(f"✅ Processed: **{filename}**")
                    except Exception as e:
                        st.error(f"❌ Failed to process {filename}: {e}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processed {i+1} of {len(uploaded_files)}")
                
                progress_bar.empty()
                
                if output_data:
                    zip_bytes = create_zip_archive(output_data)
                    st.download_button(
                        label=f"⬇️ Download Results",
                        data=zip_bytes,
                        file_name=f"{lang_code.lower()}_tagged.zip",
                        mime="application/zip",
                        key=f"dl_upload_{lang_code}"
                    )
    
    # --- MODE 2: DIRECT INPUT ---
    elif input_method == "✍️ Direct Input":
        st.subheader("Type or Paste Text")
        user_input = st.text_area("Enter text here:", height=200, key=f"text_input_{lang_code}")
        
        if st.button("Tag Text", key=f"btn_input_{lang_code}"):
            if user_input.strip():
                with st.spinner("Processing..."):
                    # Process and get list of strings
                    try:
                        tagged_lines = tagger_function(user_input)
                        
                        # PREVIEW: Create DataFrame
                        data = []
                        for line in tagged_lines:
                            parts = line.split('\t')
                            if len(parts) == 3:
                                data.append({"Token": parts[0], "POS": parts[1], "Lemma": parts[2]})
                        
                        if data:
                            st.write("### Result Preview")
                            df = pd.DataFrame(data)
                            st.dataframe(df, use_container_width=True)
                            
                            # DOWNLOAD: Create XML
                            # Wrap wrapped lines into XML structure
                            full_xml = f'<text lang="{lang_code}">\n' + "\n".join(tagged_lines) + '\n</text>'
                            final_output = f'<?xml version="1.0" encoding="UTF-8"?>\n{full_xml}'
                            
                            st.download_button(
                                label="⬇️ Download XML",
                                data=final_output,
                                file_name=f"{lang_code.lower()}_input_tagged.xml",
                                mime="text/xml",
                                key=f"dl_input_{lang_code}"
                            )
                        else:
                            st.warning("No tokens found.")
                            
                    except Exception as e:
                        st.error(f"Error processing text: {e}")
            else:
                st.warning("Please enter some text.")

def main():
    st.set_page_config(page_title="Multilingual Tagger", layout="wide")
    st.title("🌐 Multilingual Tokenizer")
    
    st.sidebar.title("Configuration")
    language = st.sidebar.radio(
        "Choose Language:",
        ('JAPANESE', 'ENGLISH', 'INDONESIAN'),
        index=0
    )
    
    if language == 'JAPANESE':
        tokenizer_interface("Japanese", "JP", run_tagger_japanese)
    elif language == 'ENGLISH':
        tokenizer_interface("English", "EN", run_tagger_english)
    elif language == 'INDONESIAN':
        tokenizer_interface("Indonesian", "ID", run_tagger_indonesian)

if __name__ == "__main__":
    main()
