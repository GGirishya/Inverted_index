#
# Inverted Index Builder
# Guillaume Girishya
#
# This program reads a folder of text files, cleans up the words in
# each one, and builds an inverted index (word -> which docs it's in
# and how many times). It also prints some basic stats and saves the
# index to a file so it can be loaded back later.
#
# set USE_PARALLEL = True below to build the index using
# multiple processes instead of one at a time.

import os
import re
import string
import json
import time
from multiprocessing import Pool, cpu_count

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ---- Global variables----
DOCS_DIR = "documents"       # folder with the .txt files
STOPWORDS_FILE = "stopwords.txt"
INDEX_FILE = "index.json"
TOP_N = 20                   # how many top words to print
USE_PARALLEL = True      #True = build index with multiple processes

stemmer = PorterStemmer()


def print_header():
    # this is just the required banner with course + name info
    print("================================================================")
    print("First Name: Guillaume")
    print("Last Name : Girishya")
    print("Git : https://github.com/GGirishya")
    print("=================================================================")


def get_stopwords():
    # reads stopwords.txt into a set (one word per line)
    words = set()
    with open(STOPWORDS_FILE) as f:
        for line in f:
            line = line.strip().lower()
            if line:
                words.add(line)
    print(f"Loaded {len(words)} stop words from '{STOPWORDS_FILE}'.")
    return words


def load_documents():
    # reads every .txt file in DOCS_DIR into a dict: {filename: text}
    print(f"\nReading documents from '{DOCS_DIR}' ...")
    docs = {}
    for fname in os.listdir(DOCS_DIR):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(DOCS_DIR, fname)
        with open(path, encoding="utf-8", errors="ignore") as f:
            docs[fname] = f.read()
    print(f"Loaded {len(docs)} documents.")
    return docs


def clean_text(text, stopwords):
    # takes raw text and returns a list of clean, stemmed words:
    # tokenize -> lowercase -> remove punctuation -> remove stopwords -> stem
    words = word_tokenize(text.lower())

    result = []
    for w in words:
        w = w.translate(str.maketrans("", "", string.punctuation))  # strip punctuation
        w = re.sub(r"[^a-z0-9]", "", w)  # keep only letters/digits and will drops any punctuation, symbol or stray space

        if w == "" or w in stopwords or w.isdigit():
            continue
        w = stemmer.stem(w)
        if w not in stopwords:
            result.append(w)
    return result


def count_words(doc_name, text, stopwords):
    # helper: cleans one document and counts how many times each word shows up in it
    # returns (doc_name, {word: count}) so it can be merged into the index
    words = clean_text(text, stopwords)
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return doc_name, counts


def _worker(args):
    # small wrapper needed because Pool.map only takes one argument per call
    doc_name, text, stopwords = args
    return count_words(doc_name, text, stopwords)


def build_index(docs, stopwords):
    # normal, one after the otherversion: goes through every doc, counts its
    # words, and merges those counts into the big index dict
    index = {}
    for doc_name, text in docs.items():
        _, counts = count_words(doc_name, text, stopwords)
        for w, c in counts.items():
            if w not in index:
                index[w] = {}
            index[w][doc_name] = c
    return index


def build_index_parallel(docs, stopwords):
    # extra credit version: same thing as build_index, but splits the
    # documents across several processes to go faster
    args_list = [(name, text, stopwords) for name, text in docs.items()]

    index = {}
    with Pool(processes=cpu_count()) as pool:
        for doc_name, counts in pool.map(_worker, args_list):
            for w, c in counts.items():
                if w not in index:
                    index[w] = {}
                index[w][doc_name] = c
    return index


def print_index_size(index):
    # figures out how big the index is by converting it to a JSON string
    # and checking how many bytes that string takes up
    size = len(json.dumps(index).encode("utf-8"))
    print(f"Inverted index size: {size:,} bytes ({size / 1024 / 1024:.4f} MB)")


def print_top_terms(index, n):
    # adds up how many times each word appears across ALL documents,
    # then prints the n words with the highest total
    totals = {}
    for term, doc_counts in index.items():
        totals[term] = sum(doc_counts.values())

    top = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:n]

    print(f"\nTop {n} most frequent terms in the collection:")
    print(f"{'Rank':<6}{'Term':<20}{'Frequency':>10}")
    for rank, (term, count) in enumerate(top, start=1):
        print(f"{rank:<6}{term:<20}{count:>10}")


def save_index(index, path):
    # saves the index to disk as JSON so we don't have to rebuild it every
    # time, it's plain text, so you can open the file and read it yourself
    with open(path, "w") as f:
        json.dump(index, f)
    print(f"Index saved to '{path}'.")


def load_index(path):
    # loads a previously saved index back into memory
    with open(path) as f:
        index = json.load(f)
    print(f"Index loaded from '{path}'.")
    return index


def main():
    #  the nltk needs these small models downloaded once to tokenize the whole text
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print_header()

    docs = load_documents()
    stopwords = get_stopwords()

    start = time.time()
    if USE_PARALLEL:
        print(f"\nBuilding inverted index in PARALLEL ({cpu_count()} processes) ...")
        index = build_index_parallel(docs, stopwords)
    else:
        print("\nBuilding inverted index sequentially ...")
        index = build_index(docs, stopwords)
    elapsed = time.time() - start
    print(f"Index built with {len(index)} unique terms in {elapsed:.2f} seconds.")

    print_index_size(index)
    print_top_terms(index, TOP_N)

    save_index(index, INDEX_FILE)
    index2 = load_index(INDEX_FILE)
    print(f"\nVerified: reloaded index has {len(index2)} terms, matches the saved index: {len(index2) == len(index)}")


if __name__ == "__main__":
    main()
