# CSC734 - Homework 01
# Guillaume Girishya
#
# This program reads a folder of text files, cleans up the words in
# each one, and builds an inverted index (word -> which docs it's in
# and how many times). It also prints some basic stats and saves the
# index to a file so it can be loaded back later.
#
# Extra credit: set USE_PARALLEL = True below to build the index using
# multiple processes instead of one at a time.

import os
import re
import string
import pickle
from multiprocessing import Pool, cpu_count

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ---- settings you can tweak ----
DOCS_DIR = "documents"       # folder with the .txt files
STOPWORDS_FILE = "stopwords.txt"
INDEX_FILE = "index.pkl"
TOP_N = 20                   # how many top words to print
USE_PARALLEL = False         # extra credit: True = build index with multiple processes

stemmer = PorterStemmer()


def print_header():
    # this is just the required banner with course + name info
    print("=================== CSC734-IR Homework 01 ==============")
    print("First Name: Guillaume")
    print("Last Name : Girishya")
    print("================================================")


def get_stopwords():
    # reads stopwords.txt into a set (one word per line)
    words = set()
    with open(STOPWORDS_FILE) as f:
        for line in f:
            line = line.strip().lower()
            if line:
                words.add(line)
    return words


def load_documents():
    # reads every .txt file in DOCS_DIR into a dict: {filename: text}
    docs = {}
    for fname in os.listdir(DOCS_DIR):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(DOCS_DIR, fname)
        with open(path, encoding="utf-8", errors="ignore") as f:
            docs[fname] = f.read()
    return docs


def clean_text(text, stopwords):
    # takes raw text and returns a list of clean, stemmed words:
    # tokenize -> lowercase -> remove punctuation -> remove stopwords -> stem
    words = word_tokenize(text.lower())

    result = []
    for w in words:
        w = w.translate(str.maketrans("", "", string.punctuation))  # strip punctuation
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
    # normal, one-at-a-time version: goes through every doc, counts its
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
    # figures out how big the index is by pickling it and checking the size
    size = len(pickle.dumps(index))
    print("Index size:", size, "bytes  (", round(size / 1024 / 1024, 4), "MB )")


def print_top_terms(index, n):
    # adds up how many times each word appears across ALL documents,
    # then prints the n words with the highest total
    totals = {}
    for term, doc_counts in index.items():
        totals[term] = sum(doc_counts.values())

    top = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:n]

    print("\nTop", n, "terms:")
    for term, count in top:
        print(term, "-", count)


def save_index(index, path):
    # saves the index to disk so we don't have to rebuild it every time
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_index(path):
    # loads a previously saved index back into memory
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    # nltk needs these small models downloaded once to tokenize text
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print_header()

    stopwords = get_stopwords()
    docs = load_documents()
    print("Loaded", len(docs), "documents")

    if USE_PARALLEL:
        print("Building index in parallel...")
        index = build_index_parallel(docs, stopwords)
    else:
        index = build_index(docs, stopwords)

    print("Index has", len(index), "unique terms")

    print_index_size(index)
    print_top_terms(index, TOP_N)

    save_index(index, INDEX_FILE)
    index2 = load_index(INDEX_FILE)
    print("\nReloaded index has", len(index2), "terms - matches:", len(index2) == len(index))


if __name__ == "__main__":
    main()
