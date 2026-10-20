# CSC734 - Homework 03
# Guillaume Girishya
# Comparing documents using TF, TF-IDF, and sublinear TF-IDF.
# 
# How to run:
# python3 -m pip install -r requirements.txt
# python3 -m nltk.downloader punkt punkt_tab
# python3 hw03_girishya.py
# 
# 
# 
import heapq
import math
from pathlib import Path
from time import perf_counter

DOCS_DIR = "documents"


def load_collection(folder):
    # Read and clean the documents using hw01_girishya.py.
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
    return sorted(documents), index


def build_document_vectors(names, index):
    # Store the word counts for each document.
    vectors = {}
    for name in names:
        vectors[name] = {}
    for term, postings in index.items():
        for name, count in postings.items():
            vectors[name][term] = count
    return vectors


def calculate_idf(index, document_count):
    # Calculate IDF using log10(total documents / document frequency).
    idf = {}
    for term, postings in index.items():
        idf[term] = math.log10(document_count / len(postings))
    return idf


def calculate_weights(vectors, idf, method):
    # Applying the selected weighting formula to each word count.
    if method not in ("tf", "tf_idf", "wf_idf"):
        raise ValueError(f"this is a Unknown weighting method: {method}")
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
            if weight != 0:
                weights[term] = weight
        weighted[name] = weights
    return weighted


def normalize_vectors(vectors):
    # Divide each weight by the document vector length.
    normalized = {}
    for name, weights in vectors.items():
        length = math.sqrt(math.fsum(w * w for w in weights.values()))
        normalized[name] = {}
        if length > 0:
            for term, weight in weights.items():
                normalized[name][term] = weight / length
    return normalized


def cosine_similarity(first, second):
    # Find cosine similarity between two normalized vectors.
    if len(first) > len(second):
        first, second = second, first
    score = math.fsum(weight * second.get(term, 0.0)
                      for term, weight in first.items())
    return max(0.0, min(1.0, score))


def pair_scores(vectors):
    # Compare each pair once, without comparing a document to itself.
    names = sorted(vectors)
    for i, first in enumerate(names):
        for j in range(i + 1, len(names)):
            second = names[j]
            yield (cosine_similarity(vectors[first], vectors[second]),
                   first, second)


def get_top_k_pairs(vectors, k):
    # Keep the K highest scores, breaking ties by filename.
    pair_count = len(vectors) * (len(vectors) - 1) // 2
    if not 1 <= k <= pair_count:
        raise ValueError(f"pleaseeee!!! K must be between 1 and {pair_count}.")
    return heapq.nsmallest(
        k, pair_scores(vectors), key=lambda item: (-item[0], item[1], item[2])
    )


def ask_for_k(document_count):
    # Ask for a valid number of document pairs.
    maximum = document_count * (document_count - 1) // 2
    while True:
        try:
            k = int(input(f"Enter K (1 to {maximum} document pairs): "))
            if 1 <= k <= maximum:
                return k
        except ValueError:
            pass
        print(f"Pleaseeee!!! enter a whole number between 1 and {maximum}.")


def print_top_terms(index):
    # Print the unique word count and the ten most frequent words.
    totals = [(term, sum(postings.values()))
              for term, postings in index.items()]
    top = sorted(totals, key=lambda item: (-item[1], item[0]))[:10]
    print(f"\nThe number of unique words is: {len(index)}")
    print("The top 10 most frequent words are:")
    for rank, (term, frequency) in enumerate(top, 1):
        print(f"  {rank}. {term} ({frequency})")


def print_results(method, pairs):
    # Print the ranked document pairs and their scores.
    print(f"\nUsing {method}:")
    for rank, (score, first, second) in enumerate(pairs, 1):
        print(f"  {rank}. {first}, {second} with similarity of {score:.6f}")


def main():
    # Build the index and run all three weighting methods.
    print("================ CSC734 - Homework 03 ==================")
    print("First Name: Guillaume")
    print("Last Name : Girishya")
    print("Git : https://github.com/GGirishya/Inverted_index")
    print("========================================================")

    started = perf_counter()
    names, index = load_collection(DOCS_DIR)
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
