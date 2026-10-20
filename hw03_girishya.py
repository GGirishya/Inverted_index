# CSC734 - Homework 03
# Guillaume Girishya
"""Document similarity using TF, TF-IDF, and sublinear TF-IDF.

HOW TO ADD THIS GRADUALLY
1. Put this file beside Inverted_index.py and stopwords.txt in your repo.
2. Read Section 1 first: it loads and indexes the existing documents.
3. Follow Sections 2 and 3: convert counts into document vectors and weights.
4. Follow Sections 4 and 5: calculate similarity and select the top K pairs.
5. Section 6 connects everything and prints the assignment output.

Mac terminal setup (inside your repository folder):
    python3 -m pip install -r requirements.txt
    python3 -m nltk.downloader punkt punkt_tab
    python3 hw03_girishya.py

The default collection is documents_big, matching Inverted_index.py.
For the smaller collection:
    python3 hw03_girishya.py --documents documents

Implementation choices:
- IDF = log10(N / df), without smoothing.
- Cosine similarity compares each unordered pair once, excluding self-pairs.
- A zero-length vector gets similarity 0, including empty documents.
- Equal scores are ordered alphabetically by the two filenames.
- Counts are rebuilt to keep the index consistent with the chosen collection.
- No CountVectorizer or TfidfVectorizer is used.

This file reuses your existing preprocessing rather than changing its rules.
It is not standalone: include Inverted_index.py, requirements.txt,
stopwords.txt, and the document folder when packaging the assignment.
"""

import argparse
import heapq
import math
from pathlib import Path
from time import perf_counter


# SECTION 1: Reuse the existing document loader and index builder.
def load_collection(folder):
    # Import here so the mathematical functions can be checked independently.
    import hw01_girishya as previous

    base = Path(__file__).resolve().parent
    folder = Path(folder)
    if not folder.is_absolute():
        folder = base / folder
    if not folder.is_dir():
        raise ValueError(f"Document folder not found: {folder}")

    previous.DOCS_DIR = str(folder)
    previous.STOPWORDS_FILE = str(base / "stopwords.txt")
    documents = previous.load_documents()
    if len(documents) < 2:
        raise ValueError("At least two .txt documents are required.")
    stopwords = previous.get_stopwords()
    index = previous.build_index(documents, stopwords)
    # Keep filenames even if preprocessing removed every word in a document.
    return sorted(documents), index


# SECTION 2: Change word -> documents into document -> word counts.
def build_document_vectors(names, index):
    vectors = {name: {} for name in names}
    for term, postings in index.items():
        for name, count in postings.items():
            vectors[name][term] = count
    return vectors


def calculate_idf(index, document_count):
    # A term's posting-list length is its document frequency, not its total TF.
    return {
        term: math.log10(document_count / len(postings))
        for term, postings in index.items()
    }


# SECTION 3: Calculate each weighting method manually.
def calculate_weights(vectors, idf, method):
    if method not in ("tf", "tf_idf", "wf_idf"):
        raise ValueError(f"Unknown weighting method: {method}")
    weighted = {}
    for name, counts in vectors.items():
        weights = {}
        for term, count in counts.items():
            if count <= 0:
                continue
            if method == "tf":
                weight = float(count)
            elif method == "tf_idf":
                weight = count * idf[term]
            else:
                weight = (1 + math.log10(count)) * idf[term]
            # Zero weights do not contribute to a dot product or vector length.
            if weight != 0:
                weights[term] = weight
        weighted[name] = weights
    return weighted


def normalize_vectors(vectors):
    # Divide by length once, instead of recalculating it for every pair.
    normalized = {}
    for name, weights in vectors.items():
        length = math.sqrt(math.fsum(w * w for w in weights.values()))
        normalized[name] = {
            term: weight / length for term, weight in weights.items()
        } if length else {}
    return normalized


# SECTION 4: Cosine similarity is a dot product after normalization.
def cosine_similarity(first, second):
    """Return cosine similarity for two already normalized sparse vectors."""
    # Iterate through the smaller dictionary to reduce lookups.
    if len(first) > len(second):
        first, second = second, first
    score = math.fsum(weight * second.get(term, 0.0)
                      for term, weight in first.items())
    # Floating-point roundoff can otherwise give a score just above 1.
    return max(0.0, min(1.0, score))


# SECTION 5: Consider all distinct pairs and retain only the best K.
def pair_scores(vectors):
    names = sorted(vectors)
    for i, first in enumerate(names):
        for j in range(i + 1, len(names)):
            second = names[j]
            yield (cosine_similarity(vectors[first], vectors[second]),
                   first, second)


def get_top_k_pairs(vectors, k):
    pair_count = len(vectors) * (len(vectors) - 1) // 2
    if not 1 <= k <= pair_count:
        raise ValueError(f"K must be between 1 and {pair_count}.")
    # A generator avoids a complete list of all N(N-1)/2 scores.
    # nsmallest uses negative similarity so highest scores are returned first.
    return heapq.nsmallest(
        k, pair_scores(vectors), key=lambda item: (-item[0], item[1], item[2])
    )


def ask_for_k(document_count):
    maximum = document_count * (document_count - 1) // 2
    while True:
        try:
            k = int(input(f"Enter K (1 to {maximum} document pairs): "))
            if 1 <= k <= maximum:
                return k
        except ValueError:
            pass
        print(f"Please enter a whole number between 1 and {maximum}.")


def print_top_terms(index):
    totals = [(term, sum(postings.values()))
              for term, postings in index.items()]
    top = sorted(totals, key=lambda item: (-item[1], item[0]))[:10]
    print(f"\nThe number of unique words is: {len(index)}")
    print("The top 10 most frequent words are:")
    for rank, (term, frequency) in enumerate(top, 1):
        print(f"  {rank}. {term} ({frequency})")


def print_results(method, pairs):
    print(f"\nUsing {method}:")
    for rank, (score, first, second) in enumerate(pairs, 1):
        print(f"  {rank}. {first}, {second} with similarity of {score:.6f}")


# SECTION 6: Run the complete assignment.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default="documents_big",
                        help="Document folder, relative to this script or absolute")
    args = parser.parse_args()
    print("================ CSC734 - Homework 03 ================")
    print("First Name: Guillaume")
    print("Last Name : Girishya")
    print("======================================================")

    started = perf_counter()
    names, index = load_collection(args.documents)
    print_top_terms(index)
    counts = build_document_vectors(names, index)
    idf = calculate_idf(index, len(names))
    print(f"Indexing and setup: {perf_counter() - started:.3f} seconds")
    k = ask_for_k(len(names))
    print(f"\nThe top {k} closest document pairs are:")

    for method in ("tf", "tf_idf", "wf_idf"):
        started = perf_counter()
        weighted = calculate_weights(counts, idf, method)
        normalized = normalize_vectors(weighted)
        pairs = get_top_k_pairs(normalized, k)
        print_results(method, pairs)
        print(f"  Processing time: {perf_counter() - started:.3f} seconds")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError, LookupError) as error:
        raise SystemExit(f"Error: {error}") from error
