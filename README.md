# Inverted Index Builder

A small Python tool that reads a folder of text documents and builds an
inverted index: for every word, which documents it appears in and how
many times. Includes text cleaning (tokenizing, stopword removal,
stemming), index size reporting, top-term stats, save/load to JSON, and
an optional multi-process mode.

---
## What it does

1. Reads every `.txt` file in a `documents/` folder.
2. Cleans and normalizes the text: lowercases it, tokenizes with NLTK,
   strips out punctuation/symbols, removes stopwords (using a
   provided `stopwords.txt` list), and stems each word with NLTK's
   Porter stemmer.
3. Builds an inverted index: `{word: {doc_name: count}}`.
4. Prints the index size (bytes/MB), and the top N most frequent words
   across the whole collection.
5. Saves the index to `index.json` and reloads it to confirm it round-trips.

---
## Setup

```
pip install -r requirements.txt
```

Place `Inverted_index.py`, `stopwords.txt`, and a `documents/` folder
(full of `.txt` files) in the same directory.

## Run it

```
python Inverted_index.py
```

## Settings

All at the top of the file:

- `DOCS_DIR` — folder of `.txt` files to index (default `documents`)
- `STOPWORDS_FILE` — path to the stopword list (default `stopwords.txt`)
- `INDEX_FILE` — where the index gets saved/loaded (default `index.json`)
- `TOP_N` — how many top words to print (default `20`)
- `USE_PARALLEL` — `True` builds the index using multiple processes
  (`multiprocessing.Pool`), `False` builds it one document at a time

---
## Sequential vs. parallel

The parallel mode splits documents across CPU cores instead of
processing them one at a time. On this project's dataset (~1,500 short
documents), parallel actually ran *slower* than sequential — spinning up
worker processes and shipping data to/from them costs more than the
small amount of work saved per document. Parallel processing only pays
off once there's enough work to outweigh that setup cost: a much larger
document collection, or documents that are individually much longer to
clean and stem. Worth remembering next time this looks tempting on a
small dataset — measure first.

---
## Files

- `Inverted_index.py` — the whole program
- `requirements.txt` — just `nltk`; everything else is standard library
- `index.json` — generated after running, holds the saved index
