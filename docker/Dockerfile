# Use Debian Bullseye Slim for a small footprint
FROM debian:bullseye-slim

# Install dependencies (wget for downloading, perl for scripts)
RUN apt-get update && apt-get install -y \
    wget \
    perl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/treetagger

# Download TreeTagger files
RUN wget --no-check-certificate https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.4.tar.gz && \
    wget --no-check-certificate https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz && \
    wget --no-check-certificate https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh

# Install TreeTagger
RUN tar -xzvf tree-tagger-linux-3.2.4.tar.gz && \
    tar -xzvf tagger-scripts.tar.gz && \
    chmod +x install-tagger.sh && \
    ./install-tagger.sh && \
    chmod +x bin/tree-tagger

# Add TreeTagger binary and scripts to PATH
ENV PATH="/opt/treetagger/bin:/opt/treetagger/cmd:${PATH}"

# Copy the Indonesian parameter file
# We rename it to 'indonesian.par' for consistency, but you can keep the original name
COPY indonesian_v311225.par lib/indonesian.par

# Copy clitic splitter script
COPY split-clitics.perl cmd/split-clitics.perl

# Copy Morphological Fixer and Lexicon
COPY lexicon2.txt lib/lexicon.txt
COPY morph-fixer.perl cmd/morph-fixer.perl

# Copy Multi-Word Lexicon
COPY indonesian-mwls.txt lib/indonesian-mwls.txt

# Create a convenience script to run the tagger with Indonesian parameters
# Pipeline:
# 1. Tokenize (utf8-tokenize.perl)
# 2. Split Clitics (split-clitics.perl) - Now respects lexicon
# 3. MWL Lookup (mwl-lookup.perl) - OPTIONAL via -mwu flag
# 4. Tag (tree-tagger)
# 5. Fix Lemmas (morph-fixer.perl)
RUN echo '#!/bin/sh' > cmd/tag-indonesian && \
    echo 'USE_MWU=0' >> cmd/tag-indonesian && \
    echo 'ARGS=""' >> cmd/tag-indonesian && \
    echo 'for arg in "$@"; do' >> cmd/tag-indonesian && \
    echo '  if [ "$arg" = "-mwu" ]; then' >> cmd/tag-indonesian && \
    echo '    USE_MWU=1' >> cmd/tag-indonesian && \
    echo '  else' >> cmd/tag-indonesian && \
    echo '    ARGS="$ARGS $arg"' >> cmd/tag-indonesian && \
    echo '  fi' >> cmd/tag-indonesian && \
    echo 'done' >> cmd/tag-indonesian && \
    echo '' >> cmd/tag-indonesian && \
    echo 'if [ "$USE_MWU" -eq 1 ]; then' >> cmd/tag-indonesian && \
    echo '  utf8-tokenize.perl -a /opt/treetagger/lib/indonesian-abbreviations $ARGS | perl /opt/treetagger/cmd/split-clitics.perl | perl /opt/treetagger/cmd/mwl-lookup.perl -f /opt/treetagger/lib/indonesian-mwls.txt | tree-tagger /opt/treetagger/lib/indonesian.par -token -lemma | perl /opt/treetagger/cmd/morph-fixer.perl' >> cmd/tag-indonesian && \
    echo 'else' >> cmd/tag-indonesian && \
    echo '  utf8-tokenize.perl -a /opt/treetagger/lib/indonesian-abbreviations $ARGS | perl /opt/treetagger/cmd/split-clitics.perl | tree-tagger /opt/treetagger/lib/indonesian.par -token -lemma | perl /opt/treetagger/cmd/morph-fixer.perl' >> cmd/tag-indonesian && \
    echo 'fi' >> cmd/tag-indonesian && \
    chmod +x cmd/tag-indonesian

# Set Entrypoint to allow passing arguments (like -mwu)
ENTRYPOINT ["tag-indonesian"]

