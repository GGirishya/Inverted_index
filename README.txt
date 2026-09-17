CSC734 Homework 01
Guillaume Girishya

Setup:
  pip install -r requirements.txt

Put the documents folder (from documents.zip) and stopwords.txt next to
hw01_girishya.py, then run:

  python hw01_girishya.py

It prints the course/name banner, loads the docs and stop words, builds
the inverted index (with timing), shows the index size and top 20
terms, and saves/reloads the index as a JSON file (index.json).

Extra credit: USE_PARALLEL is set to True near the top of the file, so
the index is built using multiple processes by default. Set it to
False to build it sequentially instead.


Conclusion: I think because the dataset is so small, the fixed cost of spinning up multiple processes 
and shipping data to and from them outweighs the tiny amount of actual work per document, 
so splitting it across cores ends up slower overall. With a much larger dataset, hundreds of thousands of documents, 
or documents that are much longer, the real cleaning and stemming work per process would grow large enough 
that the overhead becomes a small fraction of the total time, 
and at that point parallel processing would genuinely speed things up.