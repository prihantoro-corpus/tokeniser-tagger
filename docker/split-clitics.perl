#!/usr/bin/perl

# Simple Clitic Splitter for Indonesian
# Splits attached pronouns and particles:
# Prefixes: ku-, kau-
# Suffixes: -ku, -mu, -nya, -lah, -kah, -pun

# Load Lexicon to avoid splitting known words (e.g., "adalah", "sekolah")
my %lexicon;
my $lex_file = "/opt/treetagger/lib/lexicon.txt";

if (-e $lex_file) {
    open my $fh, '<', $lex_file or die "Cannot open lexicon: $!";
    while (<$fh>) {
        chomp;
        my @parts = split(/\t/, $_);
        if (@parts >= 1) {
            $lexicon{lc($parts[0])} = 1;
        }
    }
    close $fh;
}

while (<>) {
    # If the word is already in the lexicon, don't split it!
    # (Except maybe for some specific cases, but generally we want to keep dictionary words intact)
    my $word = $_;
    chomp($word);
    
    # Simple check: if exact match in lexicon, print and next
    # Note: Input contains punctuation from tokenizer, so we might need to be careful.
    # But usually tokenizer puts punctuation on separate lines.
    
    if (exists $lexicon{lc($word)}) {
        print "$word\n";
        next;
    }

    # Split suffixes (nya, ku, mu, lah, kah, pun)
    # We use a loop to handle multiple suffixes (e.g., "bukunyalah" -> "buku" "nya" "lah")
    # Matches alphabetic characters followed by one of the suffixes at the end of the word
    $_ = $word; # Reset $_ to work with s///
    
    s/([a-zA-Z]+)(ku|mu|nya|lah|kah|pun)\b/$1\n$2/g;
    
    # Repeat for second level suffixes if necessary (e.g. makanannya -> makanan + nya)
    s/([a-zA-Z]+)(ku|mu|nya|lah|kah|pun)\b/$1\n$2/g;

    # Split prefixes (ku-, kau-)
    # Matches ku/kau at the start of a word followed by uppercase or lowercase letter
    s/\b(ku|kau)([a-zA-Z]+)/$1\n$2/g;

    print "$_\n";
}
