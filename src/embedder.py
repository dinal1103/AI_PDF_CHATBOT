"""

Wraps SentenceTransformer to produce dense embeddings.
 
• Uses functools.lru_cache so the model loads once per process.
  Works in both Streamlit (via st.cache_resource wrapper in ui layer)
  AND FastAPI (plain Python cache)
• Supports batch encoding with a progress bar.
"""

#convert human language to number

#type hints are stored as strings and evaluated only when needed.
from __future__ import annotations
#type annotation like List[str]
import functools
from typing import List

import numpy as np


@functools.lru_cache(maxsize=4)#caches the model so it loads only once and reuse

def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    #load and cache model MiniLM and return ready to use model(Cache per process)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


class Embedder:#wrapper class

    def __init__(self , model_name : str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = load_embedder(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()


    def embed_texts(
                self,
                texts: List[str], 
                batch_size: int=64,#instead of one chunk 64 together
                show_progress: bool = False,
                ) -> np.ndarray:#convert many chunk together to vector

        embeddings = self.model.encode(
                texts,
                batch_size = batch_size,
                show_progress_bar = show_progress,
                convert_to_numpy = True,
                normalize_embeddings = True,#cosine similarity via dot product
                # all arrow in same length so direction matter which means meaning 
                # all arrow(vector) length 1
                )#we give input and output is vector
        return embeddings.astype(np.float32)
            # # return float32 numpy array of shape(len(texts) ,dimension)
            #322 because half memory faster 


    def embed_query(self, query: str) -> np.ndarray:
            #embed a single query string to shap(dimension,)
            emb = self.model.encode(
                [query],
                convert_to_numpy = True,
                normalize_embeddings = True,
            )# list of texts to 1 vector 
            return emb[0].astype(np.float32)