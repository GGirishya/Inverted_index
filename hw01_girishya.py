"""
CSC734 - Information Retrieval and Web Search
Homework Assignment 01 - Text Processing and Inverted Index
Name: Guillaume Girishya

Description
-----------
This program reads a collection of plain-text documents, cleans and
normalizes the text using NLTK (tokenization, stop word removal,
lower-casing, punctuation removal, and Porter stemming), builds an
inverted index (term -> {doc_id: term_frequency}), reports the size of
the index in bytes/MB, displays the most frequent terms in the
collection, and saves/loads the index to/from disk using pickle.

Usage
-----
    python hw01_girishya.py --docs_dir documents --stopwords stopwords.txt
                             --top_n 20 --index_file inverted_index.pkl

Run "python hw01_girishya.py --help" for all options, including
--parallel to build the index with multiple processes (extra credit).
"""

import os
import re
import sys
import time
import pickle
import string
import argparse
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# ---------------------------------------------------------------------------
# Function 1: display_course_info
# ---------------------------------------------------------------------------
def display_course_info(first_name="Guillaume", last_name="Girishya"):
    """
    Display the course/assignment banner with student name, as required
    by the assignment instructions.

    Parameters
    ----------
    first_name : str
        Student's first name.
    last_name : str
        Student's last name.

    Returns
    -------
    None
    """
    print("=================== CSC734-IR Homework 01 ==============")
    print(f"First Name: {first_name}")
    print(f"Last Name : {last_name}")
    print("=======================================================")


# ---------------------------------------------------------------------------
# Function 2: ensure_nltk_resources
# ---------------------------------------------------------------------------
def ensure_nltk_resources():
    """
    Make sure the NLTK resources needed for tokenization are available
    locally, downloading them if necessary (only tokenizer models are
    fetched from NLTK; the stop word list used for filtering is the
    course-provided stopwords.txt file, not NLTK's own list).

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    resources = ["tokenizers/punkt", "tokenizers/punkt_tab"]
    for resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            package_name = resource_path.split("/")[-1]
            try:
                nltk.download(package_name, quiet=True)
            except Exception as exc:  # pragma: no cover - network may be unavailable
                print(f"Warning: could not download NLTK resource '{package_name}': {exc}")


# ---------------------------------------------------------------------------
# Function 3: load_stopwords
# ---------------------------------------------------------------------------
def load_stopwords(stopwords_path):
    """
    Load the course-provided stop word list from a text file (one word
    per line).

    Parameters
    ----------
    stopwords_path : str
        Path to the stopwords.txt file.

    Returns
    -------
    set of str
        The set of stop words, used for O(1) membership tests.
    """
    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = {line.strip().lower() for line in f if line.strip()}
    return stopwords


# ---------------------------------------------------------------------------
# Function 4: read_documents
# ---------------------------------------------------------------------------
def read_documents(docs_dir):
    """
    Read every .txt file in a directory (recursively) into memory.

    Parameters
    ----------
    docs_dir : str
        Path to the directory that contains the document collection.

    Returns
    -------
    dict
        Mapping of doc_id (the file name, used as the document
        identifier) -> raw text content of that document.
    """
    documents = {}
    for root, _dirs, files in os.walk(docs_dir):
        for filename in files:
            if filename.lower().endswith(".txt"):
                doc_id = filename
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        documents[doc_id] = f.read()
                except OSError as exc:
                    print(f"Warning: could not read {filepath}: {exc}")
    return documents


# ---------------------------------------------------------------------------
# Function 5: preprocess_text
# ---------------------------------------------------------------------------
_stemmer = PorterStemmer()
_punct_table = str.maketrans("", "", string.punctuation)


def preprocess_text(text, stopwords):
    """
    Clean and normalize a block of raw text into a list of index-ready
    terms: tokenize with NLTK, lower-case, strip punctuation, drop stop
    words (using the provided list) and purely numeric/empty tokens,
    then apply Porter stemming.

    Parameters
    ----------
    text : str
        Raw document text.
    stopwords : set of str
        Set of stop words to remove.

    Returns
    -------
    list of str
        Cleaned, stemmed tokens, in order of appearance.
    """
    text = text.lower()
    tokens = word_tokenize(text)

    cleaned_tokens = []
    for token in tokens:
        token = token.translate(_punct_table)
        if not token:
            continue
        if token in stopwords:
            continue
        if token.isdigit():
            continue
        stemmed = _stemmer.stem(token)
        if stemmed and stemmed not in stopwords:
            cleaned_tokens.append(stemmed)
    return cleaned_tokens


# ---------------------------------------------------------------------------
# Function 6: build_inverted_index
# ---------------------------------------------------------------------------
def build_inverted_index(documents, stopwords):
    """
    Build an inverted index from a collection of documents. Each term
    maps to a dictionary of {doc_id: term_frequency_in_that_doc}, which
    supports both boolean retrieval and frequency-based ranking.

    Parameters
    ----------
    documents : dict
        Mapping of doc_id -> raw text, as returned by read_documents().
    stopwords : set of str
        Stop words to remove during preprocessing.

    Returns
    -------
    dict
        The inverted index: {term: {doc_id: frequency}}.
    """
    index = defaultdict(dict)
    for doc_id, text in documents.items():
        tokens = preprocess_text(text, stopwords)
        term_counts = defaultdict(int)
        for term in tokens:
            term_counts[term] += 1
        for term, count in term_counts.items():
            index[term][doc_id] = count
    return dict(index)


# ---------------------------------------------------------------------------
# Helper for the parallel (extra-credit) version. Must be a top-level
# function so it can be pickled and sent to worker processes.
# ---------------------------------------------------------------------------
def _process_single_document(args):
    """
    Worker function used by the multiprocessing pool: preprocesses one
    document and returns its per-term counts.

    Parameters
    ----------
    args : tuple
        (doc_id, text, stopwords)

    Returns
    -------
    tuple
        (doc_id, {term: frequency})
    """
    doc_id, text, stopwords = args
    tokens = preprocess_text(text, stopwords)
    term_counts = defaultdict(int)
    for term in tokens:
        term_counts[term] += 1
    return doc_id, dict(term_counts)


# ---------------------------------------------------------------------------
# Function 7: build_inverted_index_parallel (Extra Credit)
# ---------------------------------------------------------------------------
def build_inverted_index_parallel(documents, stopwords, num_processes=None):
    """
    Extra-credit version of build_inverted_index() that distributes the
    per-document preprocessing work across multiple processes using
    multiprocessing.Pool, then merges the partial results into a single
    inverted index.

    Parameters
    ----------
    documents : dict
        Mapping of doc_id -> raw text.
    stopwords : set of str
        Stop words to remove during preprocessing.
    num_processes : int, optional
        Number of worker processes to use (defaults to the number of
        available CPU cores).

    Returns
    -------
    dict
        The inverted index: {term: {doc_id: frequency}}.
    """
    if num_processes is None:
        num_processes = cpu_count()

    work_items = [(doc_id, text, stopwords) for doc_id, text in documents.items()]

    index = defaultdict(dict)
    with Pool(processes=num_processes) as pool:
        for doc_id, term_counts in pool.imap_unordered(_process_single_document, work_items):
            for term, count in term_counts.items():
                index[term][doc_id] = count
    return dict(index)


# ---------------------------------------------------------------------------
# Function 8: report_index_size
# ---------------------------------------------------------------------------
def report_index_size(index):
    """
    Compute and display the size of the inverted index in bytes and
    megabytes, based on its serialized (pickled) representation.

    Parameters
    ----------
    index : dict
        The inverted index.

    Returns
    -------
    tuple
        (size_in_bytes, size_in_mb)
    """
    serialized = pickle.dumps(index)
    size_bytes = len(serialized)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Inverted index size: {size_bytes:,} bytes ({size_mb:.4f} MB)")
    return size_bytes, size_mb


# ---------------------------------------------------------------------------
# Function 9: top_n_frequent_terms
# ---------------------------------------------------------------------------
def top_n_frequent_terms(index, n=20):
    """
    Determine the n most frequent terms across the entire collection
    (i.e., total occurrences summed over all documents) and print them.

    Parameters
    ----------
    index : dict
        The inverted index: {term: {doc_id: frequency}}.
    n : int
        Number of top terms to display.

    Returns
    -------
    list of tuple
        [(term, total_frequency), ...] sorted descending, length n.
    """
    collection_freq = {term: sum(doc_freqs.values()) for term, doc_freqs in index.items()}
    top_terms = sorted(collection_freq.items(), key=lambda kv: kv[1], reverse=True)[:n]

    print(f"\nTop {n} most frequent terms in the collection:")
    print(f"{'Rank':<6}{'Term':<20}{'Frequency':>10}")
    for rank, (term, freq) in enumerate(top_terms, start=1):
        print(f"{rank:<6}{term:<20}{freq:>10}")
    return top_terms


# ---------------------------------------------------------------------------
# Function 10: save_index / load_index
# ---------------------------------------------------------------------------
def save_index(index, filepath):
    """
    Persist the inverted index to disk using pickle.

    Parameters
    ----------
    index : dict
        The inverted index to save.
    filepath : str
        Destination file path.

    Returns
    -------
    None
    """
    with open(filepath, "wb") as f:
        pickle.dump(index, f)
    print(f"Index saved to '{filepath}'.")


def load_index(filepath):
    """
    Load a previously saved inverted index from disk.

    Parameters
    ----------
    filepath : str
        Path to the pickled index file.

    Returns
    -------
    dict
        The reconstructed inverted index.
    """
    with open(filepath, "rb") as f:
        index = pickle.load(f)
    print(f"Index loaded from '{filepath}'.")
    return index


# ---------------------------------------------------------------------------
# Function 11: main - orchestrates the whole pipeline
# ---------------------------------------------------------------------------
def main():
    """
    Parse command-line arguments and run the full pipeline: display the
    course banner, load stop words, read documents, build the inverted
    index (sequentially or in parallel), report its size, display the
    top-N frequent terms, and save/reload the index to verify
    persistence.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="CSC734 HW01 - Build an inverted index.")
    parser.add_argument("--docs_dir", default="documents",
                         help="Directory containing the .txt document collection.")
    parser.add_argument("--stopwords", default="stopwords.txt",
                         help="Path to the stop word list.")
    parser.add_argument("--top_n", type=int, default=20,
                         help="Number of top frequent terms to display.")
    parser.add_argument("--index_file", default="inverted_index.pkl",
                         help="File path used to save/load the index.")
    parser.add_argument("--parallel", action="store_true",
                         help="Build the index using multiple processes (extra credit).")
    parser.add_argument("--processes", type=int, default=None,
                         help="Number of worker processes to use with --parallel.")
    args = parser.parse_args()

    display_course_info()
    ensure_nltk_resources()

    print(f"\nReading documents from '{args.docs_dir}' ...")
    documents = read_documents(args.docs_dir)
    print(f"Loaded {len(documents)} documents.")

    stopwords = load_stopwords(args.stopwords)
    print(f"Loaded {len(stopwords)} stop words from '{args.stopwords}'.")

    start = time.time()
    if args.parallel:
        print(f"\nBuilding inverted index in PARALLEL "
              f"({args.processes or cpu_count()} processes) ...")
        index = build_inverted_index_parallel(documents, stopwords, args.processes)
    else:
        print("\nBuilding inverted index sequentially ...")
        index = build_inverted_index(documents, stopwords)
    elapsed = time.time() - start
    print(f"Index built with {len(index)} unique terms in {elapsed:.2f} seconds.")

    report_index_size(index)
    top_n_frequent_terms(index, args.top_n)

    save_index(index, args.index_file)
    reloaded_index = load_index(args.index_file)
    assert len(reloaded_index) == len(index), "Reloaded index does not match saved index!"
    print("\nVerified: reloaded index matches the saved index.")


if __name__ == "__main__":
    main()
