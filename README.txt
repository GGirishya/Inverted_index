CSC734 - Information Retrieval and Web Search
Homework Assignment 01
Name: Guillaume Girishya

FILES
-----
hw01_girishya.py   Main Python program (text processing + inverted index).
README.txt         This file.

REQUIREMENTS
------------
- Python 3.9+
- NLTK (pip install nltk --break-system-packages, or just pip install nltk)
  The script will automatically download the small "punkt" / "punkt_tab"
  tokenizer models on first run if they are not already present.

FOLDER SETUP
------------
Place the program in a folder alongside:
  - a "documents" folder containing the .txt files from documents.zip
    (unzip documents.zip so the .txt files sit directly inside a folder
    named "documents"), and
  - stopwords.txt (the provided stop word list).

HOW TO RUN
----------
Basic (sequential) run, using the defaults:

    python hw01_girishya.py

Equivalent to:

    python hw01_girishya.py --docs_dir documents --stopwords stopwords.txt \
        --top_n 20 --index_file inverted_index.pkl

Extra-credit parallel run (builds the index using multiple processes):

    python hw01_girishya.py --parallel

Optionally choose the number of worker processes:

    python hw01_girishya.py --parallel --processes 4

All options:

    python hw01_girishya.py --help

WHAT THE PROGRAM DOES
----------------------
1. Prints the required course/name banner.
2. Reads every .txt file in --docs_dir into memory.
3. Preprocesses each document: NLTK tokenization, lower-casing,
   punctuation stripping, stop-word removal (using the provided
   stopwords.txt list), and Porter stemming.
4. Builds an inverted index: {term: {doc_id: term_frequency}}.
5. Reports the size of the index in bytes and MB (based on its pickled
   size).
6. Displays the top N most frequent terms in the entire collection.
7. Saves the index to disk (pickle) and reloads it to verify
   persistence.
8. (Extra credit, with --parallel) Builds the same index using a
   multiprocessing.Pool, splitting document preprocessing across
   worker processes, then merges the partial results.

The code contains 11 functions, each documented with a docstring
describing its inputs and outputs, as required by the assignment.
