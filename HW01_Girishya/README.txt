CSC734 Homework 01
Guillaume Girishya

Setup:
  pip install -r requirements.txt

Put the documents folder (from documents.zip) and stopwords.txt next to
hw01_girishya.py, then run:

  python hw01_girishya.py

It prints the course/name banner, loads the docs, builds the inverted
index, shows the index size and top 20 terms, and saves/reloads the
index file (index.pkl).

Extra credit: open hw01_girishya.py and set USE_PARALLEL = True near
the top to build the index using multiple processes instead of one.
