# Indonesian TreeTagger Docker Image

This project provides a Docker image for tagging Indonesian text using TreeTagger, with custom enhancements for clitic splitting, morphological lemmatization, and multiword unit support.

## Features
*   **Tokenization**: Handles Indonesian punctuation.
*   **Clitic Splitting**: Smartly splits `ku-`, `-mu`, `-nya`, `-lah`, etc., while respecting the dictionary (e.g., preserves "adalah").
*   **Lemmatization Fixer**: Corrects `<unknown>` lemmas by attempting to strip suffixes (`-kan`, `-i`, `-an`).
*   **Multiword Support**: Optional merging of terms like "abdi negara" using the `-mwu` flag.

## How to use

### Prerequisite
Ensure Docker Desktop is installed and running.

### 1. Build the Image
If you haven't built it yet (or if you moved this folder):
```bash
docker build -t treetagger-indo .
```

### 2. Run the Tagger
**Basic Usage (Split all words):**
```bash
echo 'Dia adalah abdi negara' | docker run -i treetagger-indo
```

**With Multiword Unit Support (Merge "abdi negara"):**
```bash
echo 'Dia adalah abdi negara' | docker run -i treetagger-indo -mwu
```

## Exporting the Image
To share this image or move it to another computer:

1.  **Save to file:**
    ```bash
    docker save -o treetagger-indo.tar treetagger-indo
    ```
    This creates a large `.tar` file.

2.  **Load on another machine:**
    ```bash
    docker load -i treetagger-indo.tar
    ```

## Project Structure
*   `Dockerfile`: Build instructions.
*   `indonesian_v311225.par`: Parameter file.
*   `lexicon2.txt`: Base dictionary.
*   `indonesian-mwls.txt`: Multiword expressions list.
*   `cmd/`: Helper scripts (`tag-indonesian`, `split-clitics.perl`, `morph-fixer.perl`, `mwl-lookup.perl`).
