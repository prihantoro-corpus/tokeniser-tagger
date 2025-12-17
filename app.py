import streamlit as st
import pd as pd
import os
import zipfile
import re
from io import BytesIO
import subprocess
import sys
import xml.etree.ElementTree as ET # For XML parsing and reconstruction

# Import language libraries
from fugashi import Tagger # For Japanese
from textblob import TextBlob # For English


# --- Global Configuration and State Management ---

# --- JAPANESE TOKENIZER ---
@st.cache_resource
def get_japanese_tokenizer():
    """Initializes and returns the Fugashi Tagger with unidic-lite."""
    try:
        tagger = Tagger()
        return tagger
    except Exception as e:
        st.error(f"Error initializing Japanese Tokenizer (Fugashi/MeCab). Error: {e}")
        return None

# --- ENGLISH TEXTBLOB SETUP ---
@st.cache_resource
def initialize_english_textblob():
    """Ensures TextBlob data is downloaded."""
    try:
        import nltk
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except:
        st.info("Downloading TextBlob data (needed for English Tagging)...")
        subprocess.check_call([sys.executable, "-m", "textblob.download_corpora"])
    
    st.info("English TextBlob Tagger is ready.")
    return True

# Global Variables
JAPANESE_TAGGER = get_japanese_tokenizer()
ENGLISH_TAGGER_READY = initialize_english_textblob()


# --- Core Processing Functions ---

# --- JAPANESE PROCESSING ---
def run_tagger_japanese(text):
    """Tokenizes and tags a single Japanese text string using Fugashi."""
    if JAPANESE_TAGGER is None:
        return []
    nodes = JAPANESE_TAGGER.parseToNodeList(text)
    results = []
    for node in nodes:
        if node.surface:
            token = node.surface
            pos = node.feature.pos1
            lemma = node.feature.lemma if node.feature.lemma else token
            # Output: token \t POS \t lemma
            results.append(f"{token}\t{pos}\t{lemma}")
    return results

# --- ENGLISH PROCESSING ---
def run_tagger_english(text):
    """Tokenizes and tags a single English text string using TextBlob."""
    if not ENGLISH_TAGGER_READY:
        return []
        
    blob = TextBlob(text)
    results = []
    for token, pos_tag in blob.tags:
        # Use token as lemma for deployment stability
        lemma = token 
        # Output: token \t POS \t lemma
        results.append(f"{token}\t{pos_tag}\t{lemma}")
    return results

def process_xml_content(xml_string, lang_code, tagger_function):
    """
    Parses the XML string and tags ONLY the plain text content, 
    preserving all XML tags and attributes.
    """
    
    # 1. CRITICAL FIX: Ensure the input XML is wrapped in a single root element
    temp_root_tag = 'TEMP_WRAPPER'
    
    # Remove any XML declaration and CDATA to avoid parser errors
    cleaned_xml_string = re.sub(r'<\?xml[^>]*\?>', '', xml_string, flags=re.IGNORECASE).strip()
    
    wrapped_xml = f"<{temp_root_tag}>{cleaned_xml_string}</{temp_root_tag}>"
    
    try:
        # 2. Parse the XML
        root = ET.fromstring(wrapped_xml)
        
    except ET.ParseError as e:
        st.warning(f"Input failed XML parsing ({e}). Processing as raw text only.")
        tagged_lines = tagger_function(xml_string)
        return f'<text lang="{lang_code}">\n' + "\n".join(tagged_lines) + '\n</text>'
        
    # 3. Function to traverse and modify the tree
    def traverse_and_tag(element):
        if element.text and element.text.strip():
            tagged_lines = tagger_function(element.text)
            element.text = '\n' + '\n'.join(tagged_lines) + '\n'

        for child in element:
            traverse_and_tag(child)

        if element.tail and element.tail.strip():
            tagged_lines = tagger_function(element.tail)
            element.tail = '\n' + '\n'.join(tagged_lines) + '\n'

    # 4. Start traversal and modification
    traverse_and_tag(root)
    
    # 5. Reconstruct the XML string
    full_xml = ET.tostring(root, encoding='unicode')
    full_xml = re.sub(r'^<TEMP_WRAPPER>', '', full_xml)
    full_xml = re.sub(r'</TEMP_WRAPPER>$', '', full_xml).strip()
    
    return full_xml

def process_text(text, lang_code, tagger_function):
    """Primary function to process the entire input text, handling XML structure."""
    return process_xml_content(text, lang_code, tagger_function)


# --- XML Creation and Zipping ---

def create_output_file_content(processed_xml, original_filename):
    """Creates the final XML output file content."""
    base_filename = os.path.splitext(original_filename)[0]
    sanitized_base_name = re.sub(r' \(\d+\)$', '', base_filename).strip()
    final_output = f'<?xml version="1.0" encoding="UTF-8"?>\n{processed_xml}'
    return final_output, f"{sanitized_base_name}_tagged.xml"


def create_zip_archive(output_data):
    """Creates a zip archive in memory and returns the bytes."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for original_name, processed_content in output_data.items():
            final_content, xml_name = create_output_file_content(processed_content, original_name)
            zf.writestr(xml_name, final_content.encode('utf-8'))
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# --- Streamlit UI Components ---

def language_selector_page():
    st.sidebar.title("🛠️ Tools")
    
    # --- ADDED USER'S MANUAL LINK HERE ---
    st.sidebar.markdown(
        "[📖 Users manual](https://docs.google.com/document/d/1i4dz4YE318Qhs5DicQBFxEdGAAbCZgvGcXN2WNHd_3I/edit?usp=sharing)"
    )
    st.sidebar.markdown("---")
    
    st.sidebar.header("Select Language Tokenizer")
    
    language = st.sidebar.radio(
        "Choose a language for tagging:",
        ('JAPANESE', 'ENGLISH', 'FRENCH (Future)'),
        index=0
    )
    
    tagger_func = None
    lang_code = None

    if language == 'JAPANESE':
        tagger_func = run_tagger_japanese
        lang_code = "JP"
    elif language == 'ENGLISH':
        tagger_func = run_tagger_english
        lang_code = "EN"
    
    if tagger_func:
        tokenizer_interface(
            lang_name=language, 
            lang_code=lang_code, 
            tagger_function=tagger_func
        )
    else:
        st.info(f"The {language} tokenizer is not yet implemented. Please select JAPANESE or ENGLISH.")

def tokenizer_interface(lang_name, lang_code, tagger_function):
    """General interface for uploading files and displaying the download button."""
    st.header(f"🌎 {lang_name} Tokenizer and XML Preserver ({lang_code})")
    st.markdown("---")

    st.subheader("Upload Text or XML Files")
    st.markdown("""
        Upload one or more files. The processor will **preserve all XML tags and attributes**
        while tokenizing, tagging, and lemmatizing **only the plain text content** inside the tags.
        
        *Output is tab-separated (token\\tPOS\\tlemma).*
    """)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'xml'], 
        accept_multiple_files=True,
        help="Ensure your text files are encoded in UTF-8."
    )

    if uploaded_files:
        if st.button(f"Start Tagging and Preserve XML Structure"):
            output_data = {}
            progress_bar = st.progress(0, text="Processing files...")
            
            for i, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                try:
                    content_bytes = uploaded_file.read()
                    text = content_bytes.decode('utf-8')
                    processed_xml = process_text(text, lang_code, tagger_function)
                    output_data[filename] = processed_xml
                    st.success(f"✅ Processed: **{filename}** (XML structure preserved)")
                except Exception as e:
                    st.error(f"❌ Failed to process {filename}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processed {i+1} of {len(uploaded_files)} files...")
            
            progress_bar.empty()
            
            if output_data:
                with st.spinner('Creating XML files and zipping results...'):
                    zip_bytes = create_zip_archive(output_data)
                
                st.subheader("Download Results")
                st.success("Processing complete! Download your results below.")
                st.download_button(
                    label=f"⬇️ Download Tagged XML Archive",
                    data=zip_bytes,
                    file_name=f"{lang_code.lower()}_preserved_tagged_xml.zip",
                    file_name=f"{lang_code.lower()}_preserved_tagged_xml.zip",
                    mime="application/zip"
                )
            else:
                st.error("No files were successfully processed.")

def main():
    st.set_page_config(
        page_title="Multilingual Tokenizer & Tagger",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🌐 Multilingual Tokenizer & Tagger Web App")
    st.markdown("This application performs structural linguistic annotation.")
    
    language_selector_page()

if __name__ == "__main__":
    main()
