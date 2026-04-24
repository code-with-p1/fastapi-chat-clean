# init_bm25.py
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
load_dotenv()

# A sample corpus to fit the initial IDF weights. 
# For true production precision, use a representative sample of your actual data.
dummy_corpus = [
    "A fluffy cat is sleeping on the rug near the fireplace.",
    "The dog is chasing a ball in the backyard.",
    "Python is a versatile programming language for data science.",
    "Vector databases enable semantic search at scale.",
    "The feline curled up on the warm carpet beside the window."
]

bm25 = BM25Encoder()
bm25.fit(dummy_corpus)
bm25.dump("bm25_model.json")

print("Successfully generated bm25_model.json!")