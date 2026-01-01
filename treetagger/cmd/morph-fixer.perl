#!/usr/bin/perl

# Morphological Fixer for Indonesian TreeTagger
# 1. Loads a lexicon (word -> lemma)
# 2. Replaces <unknown> lemmas with lexicon lookups
# 3. If still unknown, tries stripping suffixes (-kan, -i, -an) and checking lexicon again

use strict;
use warnings;

my %lexicon;
my $lex_file = "/opt/treetagger/lib/lexicon.txt";

# Load Lexicon
if (-e $lex_file) {
    open my $fh, '<', $lex_file or die "Cannot open lexicon: $!";
    while (<$fh>) {
        chomp;
        # Format: Word \t Tag \t Lemma
        my @parts = split(/\t/, $_);
        if (@parts >= 3) {
            # Map word to lemma (lowercase for lookup)
            $lexicon{lc($parts[0])} = $parts[2];
        }
    }
    close $fh;
}

while (<STDIN>) {
    # Expected format: Token \t Tag \t Lemma
    chomp;
    my @parts = split(/\t/, $_);
    
    if (@parts == 3) {
        my $token = $parts[0];
        my $tag = $parts[1];
        my $lemma = $parts[2];

        if ($lemma eq '<unknown>') {
             my $lc_token = lc($token);

             # 1. Direct Lookup
             if (exists $lexicon{$lc_token}) {
                 $lemma = $lexicon{$lc_token};
             } else {
                 # 2. Suffix Stripping & Lookup
                 # Try stripping -kan
                 if ($lc_token =~ /(.+)kan$/) {
                     my $root = $1;
                     if (exists $lexicon{$root}) {
                         $lemma = $lexicon{$root};
                     }
                 }
                 # Try stripping -i
                 elsif ($lc_token =~ /(.+)i$/) {
                     my $root = $1;
                     if (exists $lexicon{$root}) {
                         $lemma = $lexicon{$root};
                     }
                 }
                 # Try stripping -an
                 elsif ($lc_token =~ /(.+)an$/) {
                     my $root = $1;
                     if (exists $lexicon{$root}) {
                         $lemma = $lexicon{$root};
                     }
                 }
             }
        }
        print "$token\t$tag\t$lemma\n";
    } else {
        print "$_\n";
    }
}
