import streamlit as st
import pandas as pd
import os
import zipfile
import re
import subprocess
import sys
from io import BytesIO

# Import components from NLTK
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# Import components for Japanese
from fugashi import Tagger 


# --- NLTK WordNet Data Installation (The Robust Fix) ---
@st.cache_resource
def download_nltk_data():
    """
    Downloads required NLTK data packages (tagger, wordnet, punkt) 
    using the recommended internal method.
    """
    
    # List of resources required for the NLTK pipeline
    resources = ['averaged_perceptron_tagger', 'wordnet', 'punkt']

    for resource in resources:
        try:
            # Check if resource is available
            nltk.data.find(f'taggers/{resource}') # Using a general check path
        except LookupError:
            # If not found, use the downloader. This usually succeeds in Streamlit Cloud.
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                # If the simple download fails, try the system call (less likely needed, but safer)
                subprocess.check_call([sys.executable, "-m", "nltk.downloader", resource])

    # Initialize the lemmatizer instance after data is downloaded
    lemmatizer = WordNetLemmatizer()
    st.info("NLTK data (POS Tagging & WordNet) loaded successfully.")
    return lemmatizer

# Global Variables
WORDNET_LEMMATIZER = download_nltk_data()


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

# Global Variable
JAPANESE_TAGGER = get_japanese_tokenizer()


# --- Helper Function for Lemmatization ---
def get_wordnet_pos(tag):
    """Maps NLTK's Penn Treebank tags to the simplified WordNet format."""
    if tag.startswith('J'): # Adjective
        return wn.ADJ
    elif tag.startswith('V'): # Verb
        return wn.VERB
    elif tag.startswith('N'): # Noun
        return wn.NOUN
    elif tag.startswith('R'): # Adverb
        return wn.ADV
    else:
        return wn.NOUN # Default to noun for punctuation or unknown


# --- Core Processing Functions ---

# --- JAPANESE PROCESSING (No Change) ---
def process_text_japanese(text):
    """Tokenizes and tags a single Japanese text string using Fugashi."""
    if JAPANESE_TAGGER is None:
        return None

    nodes = JAPANESE_TAGGER.parseToNodeList(text)
    results = []
    for node in nodes:
        if node.surface:
            token = node.surface
            pos = node.feature.pos1
            lemma = node.feature.lemma if node.feature.lemma else token
            results.append([token, pos, lemma])
    return results if results else None

# --- ENGLISH PROCESSING (NLTK WordNet Lemmatization) ---
def process_text_english(text):
    """Tokenizes, tags, and accurately lemmatizes English text using NLTK."""
    
    if WORDNET_LEMMATIZER is None:
        st.error("English Lemmatizer is not initialized.")
        return None
        
    # 1. Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # 2. POS Tagging (Penn Treebank format)
    tagged_tokens = nltk.pos_tag(tokens)
    
    results = []
    
    # 3. Lemmatize using the WordNetLemmatizer instance
    for token, pos_tag in tagged_tokens:
        # Get the WordNet POS tag (e.g., 'v', 'a', 'n')
        wn_tag = get_wordnet_pos(pos_tag)
        
        # Perform Lemmatization
        lemma = WORDNET_LEMMATIZER.lemmatize(token, wn_tag)
        
        # Output format: Token [TAB] POS_Tag [TAB] Lemma
        results.append([token, pos_tag, lemma])
    
    return results if results else None


# --- XML Creation and Zipping (No Change) ---

def create_xml_content(data_list, original_filename, lang_code):
    """
    Converts the tokenized list into the requested XML string format 
    (token\tpos\tlemma content) and sanitizes the filename for the 'id' attribute.
    """
    # Sanitize the filename for the XML ID
    base_filename = os.path.splitext(original_filename)[0]
    # Remove Colab's default duplicate-naming pattern: ' (n)'
    sanitized_id = re.sub(r' \(\d+\)$', '', base_filename).strip()
    
    # 1. Start the corpus tag
    xml_lines = [f'<corpus lang="{lang_code}" id="{sanitized_id}">']
    
    # 2. Generate the tab-separated content block
    content_block = []
    for token, pos, lemma in data_list:
        # TreeTagger format: token \t POS tag \t lemma
        line = f"{token}\t{pos}\t{lemma}"
        content_block.append(line)
        
    # Join the content block
    xml_lines.append("\n".join(content_block))
        
    # 3. End the corpus tag
    xml_lines.append(f'</corpus lang="{lang_code}" id="{sanitized_id}">')
    
    return "\n".join(xml_lines)


def create_zip_archive(output_data, lang_code):
    """Creates a zip archive in memory and returns the bytes."""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for original_name, data_list in output_data.items():
            
            xml_content = create_xml_content(data_list, original_name, lang_code)
            
            # Sanitize the filename for the XML file itself (similar to ID)
            base_name = os.path.splitext(original_name)[0]
            sanitized_base_name = re.sub(r' \(\d+\)$', '', base_name).strip()
            xml_name = f"{sanitized_base_name}_tagged.xml"
            
            # Write the XML content to the zip file
            zf.writestr(xml_name, xml_content.encode('utf-8'))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# --- Streamlit UI Components (No Change) ---

def language_selector_page():
    st.sidebar.title("🛠️ Tools")
    st.sidebar.header("Select Language Tokenizer")
    
    language = st.sidebar.radio(
        "Choose a language for tagging:",
        ('JAPANESE', 'ENGLISH', 'FRENCH (Future)'),
        index=0 # Default to JAPANESE
    )
    
    if language == 'JAPANESE':
        tokenizer_interface(
            lang_name="Japanese", 
            lang_code="JP", 
            tagger_func=process_text_japanese
        )
    elif language == 'ENGLISH':
        tokenizer_interface(
            lang_name="English", 
            lang_code="EN", 
            tagger_func=process_text_english
        )
    else:
        st.info(f"The {language} tokenizer is not yet implemented. Please select JAPANESE or ENGLISH.")

def tokenizer_interface(lang_name, lang_code, tagger_func):
    """General interface for uploading files and displaying the download button."""
    
    st.header(f"🌎 {lang_name} Tokenizer and Tagger ({lang_code})")
    st.markdown("---")

    st.subheader("Upload Text Files")
    st.markdown("Upload one or more `.txt` files. Results will be returned as XML files (TreeTagger format) in a single ZIP file.")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt'],
        accept_multiple_files=True,
        help="Ensure your text files are encoded in UTF-8."
    )

    if uploaded_files:
        if st.button(f"Start {lang_name} Tagging and Create XML Archive"):
            
            output_data = {}
            progress_bar = st.progress(0, text="Processing files...")
            
            # --- Processing Loop ---
            for i, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                
                try:
                    content_bytes = uploaded_file.read()
                    text = content_bytes.decode('utf-8')
                    
                    data_list = tagger_func(text)
                    
                    if data_list:
                        output_data[filename] = data_list
                        st.success(f"✅ Processed: **{filename}** ({len(data_list)} tokens)")
                    else:
                        st.warning(f"⚠️ Warning: No tokens found in {filename}.")
                        
                except Exception as e:
                    st.error(f"❌ Failed to process {filename}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processed {i+1} of {len(uploaded_files)} files...")
            
            progress_bar.empty()
            
            # --- Output/Download ---
            if output_data:
                
                with st.spinner('Creating XML and zipping results...'):
                    zip_bytes = create_zip_archive(output_data, lang_code)
                
                st.subheader("Download Results")
                st.success("Processing complete! Download your results below.")
                
                st.download_button(
                    label=f"⬇️ Download {lang_name} Tagged XML (ZIP)",
                    data=zip_bytes,
                    file_name=f"{lang_code.lower()}_tagged_results_xml_ttformat.zip",
                    mime="application/zip"
                )
                st.info("The ZIP file contains XML files in TreeTagger format (token\\tPOS\\tlemma).")
            else:
                st.error("No valid text files were successfully processed.")

def main():
    st.set_page_config(
        page_title="Multilingual Tokenizer & Tagger",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🌐 Multilingual Tokenizer & Tagger Web App")
    st.markdown("This application provides linguistic annotation services for various languages.")
    
    language_selector_page()

if __name__ == "__main__":
    main()
