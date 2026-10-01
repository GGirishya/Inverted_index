#
#
# CSC734 - Homework 02
# Guillaume Girishya
#
#  This program loads the inverted index built by hw01_girishya.py, loads a list of
# search queries from a text file (up to 4 words each), and for every
# query prints the matching documents under every possible AND/OR
# combination between the query words.
#
# Example: query "A B C" has 2 "gaps" between 3 words, so there are
# 2^2 = 4 combinations: A and B and C / A and B or C / A or B and C /
# A or B or C. A 4-word query has 3 gaps -> 2^3 = 8 combinations.

import re
import json
import itertools

import nltk
from nltk.stem import PorterStemmer

# ---- Global variables ----
INDEX_FILE = "index.json"        # index saved by hw01_girishya.py
STOPWORDS_FILE = "stopwords.txt"
QUERIES_FILE = "queries.txt"     # one query per line, max 4 words per line

stemmer = PorterStemmer()


def print_header():
    # this is just the required banner with course + name info
    print("================================================================")
    print("First Name: Guillaume")
    print("Last Name : Girishya")
    print("Git : https://github.com/GGirishya")
    print("=================================================================")



def load_index(path):
    # loads the inverted index that hw01_girishya.py saved as JSON the index.json file in the directory.
    # returns: {word: {doc_name: count}}
    with open(path) as f:
        index = json.load(f)
    print(f"Loaded index from '{path}' ({len(index)} unique terms).")
    return index


def get_stopwords(path):
    # reads the stopword list into a set, same as hw01_girishya.py, so
    # query words get normalized the exact same way the documents were normalized when the index was built
    words = set()
    with open(path) as f:
        for line in f:
            line = line.strip().lower()
            if line:
                words.add(line)
    return words


def normalize_word(word, stopwords):
    # cleans up a single query word so it matches how terms were
    # stored in the index: lower case -> remove punctuation -> stem.
    # returns the cleaned/stemmed word, or None if it's a stopword,
    # empty, or a pure number (same rules as document indexing)
    w = word.lower()
    w = re.sub(r"[^a-z0-9]", "", w)
    if w == "" or w in stopwords or w.isdigit():
        return None
    w = stemmer.stem(w)
    if w in stopwords:
        return None
    return w


def load_queries(path):
    # reads the queries file, one query per line
    # returns a list of queries, where each query is a list of the
    # original pre-normalized words typed on that line
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if len(words) > 4:
                print(f"Warning: The query '{line}' has more than 4 words, only using the first 4.")
                words = words[:4]
            queries.append(words)
    return queries


def docs_for_term(index, term):
    # looks up one (already normalized/stemmed) term in the index
    # returns a set of doc names that contain it (empty set if the
    # term isn't in the index at all)
    if term is None or term not in index:
        return set()
    return set(index[term].keys())


def combine(doc_sets, operators):
    # combines a list of doc-name sets left to right using the given
    # list of "and"/"or" operators (there's one fewer operator than
    # there are sets)
    # example: doc_sets = [setA, setB, setC], operators = ["and", "or"]
    #          -> (setA and setB) or setC
    #
    #
    result = doc_sets[0]
    for op, next_set in zip(operators, doc_sets[1:]):
        if op == "and":
            result = result & next_set
        else:
            result = result | next_set
    return result


def banner_line(text, width=60, fill="="):
    # small helper to print a "===== text =====" style line, padded
    # out to roughly the given width, matching the assignment's
    # required output style
    pad = max(width - len(text) - 2, 4)
    left = pad // 2
    right = pad - left
    print(f"{fill * left} {text} {fill * right}")


def run_query(query_num, words, index, stopwords):
    # runs one query: normalizes its words, then prints the results
    # for every AND/OR combination between them
    banner_line(f"User Query {query_num}: {' '.join(words)}")

    normalized = [normalize_word(w, stopwords) for w in words]
    doc_sets = [docs_for_term(index, t) for t in normalized]

    num_gaps = len(words) - 1
    if num_gaps == 0:
        # a single-word query has no AND/OR combinations to build
        banner_line(f"Results for: {words[0]}")
        for doc in sorted(doc_sets[0]):
            print(doc)
        banner_line("")
        return

    for operators in itertools.product(["and", "or"], repeat=num_gaps):
        # this will build a readable label like "A and B or C"
        label_parts = [words[0]]
        for op, w in zip(operators, words[1:]):
            label_parts.append(op)
            label_parts.append(w)
        label = " ".join(label_parts)

        matches = combine(doc_sets, operators)

        banner_line(f"Results for: {label}")
        for doc in sorted(matches):
            print(doc)
        banner_line("")


def main():
    print_header()

    index = load_index(INDEX_FILE)
    stopwords = get_stopwords(STOPWORDS_FILE)
    queries = load_queries(QUERIES_FILE)
    print(f"Loaded {len(queries)} queries from '{QUERIES_FILE}'.\n")

    for i, words in enumerate(queries, start=1):
        run_query(i, words, index, stopwords)


if __name__ == "__main__":
    main()
