#store vector in FAISS
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List,Optional,Tuple

import faiss

@dataclass
class RetrievalResult:#if faiss return 5 relevant chunk then each represent one retrievalresult
    chunk_id : int#index in faiss
    text : str#actual para
    score : float#similarity score
    source : str#which doc
    page : Optional[int] = None


class FATSSVectorStore:#all releated to vectordb stay in mini db manager

    def __init__(self, dimension: int):
        self.dimension = dimension#for any word give in 384d
        self.index = faiss.IndexFlatIP(dimension)#create empty vectordb with dimension
        self.texts: List[str] = []#matching text separately
        self.sources: List[str] = []
        self.pages: List[Optional[int]] = []

    
    def add(#insert vector to faiss
            self,
            embeddings: np.ndarray, #(n,dim) float32 if 5 vector(5,384)
            texts = List[str],
            sources: Optional[List[str]] = None,
            pages: Optional[List[Optional[int]]] = None,
    ) -> None:

            assert embeddings.shape[0] == len(texts)#check no. of vector eual to no of para
            self.index.add(embeddings)#copy every emb to faiss
            self.texts.extend(texts)#store metadata
            self._sources.extend(sources or [""] * len(texts))
            self._pages.extend(pages or [None] * len(texts))


    def search(#find nearest vector
            self,
            query_embedding: np.ndarray, #(dim,) float 32
            top_k: int = 5,
    ) -> List[RetrievalResult]:
        #nearest neighbour search return top_k results

        if self.index.ntotal == 0:#check if no. of vector 0
              return []
         
        k = min(top_k , self.index.ntotal)
        scores , indices = self.index.search(query_embedding[np.newaxis , :] , k)#faiss give similarity score and index of nearest

        results = []
        for score,idx in zip(scores[0] , indices[0]) :#build result zip pair together score and indixes 
             if idx == -1:
                  continue
             
             results.append(
                  RetrievalResult(#we return metadata along with text rather tha vector index
                       chunk_id= int(idx) ,
                       text = self._texts[idx],
                       score = float(score),
                       source = self._sources[idx],
                       page=self._pages[idx],
                  )
             )

        return results
    

    def search_mmr(#return chunks that are both relevant and different from each other.
              self,
              query_embedding: np.ndarray,
              top_k: int = 5,
              fetch_k: int = 20,
              lambda_mult: float = 0.7,
    ) -> List[RetrievalResult]:
         
        candidates = self.search(query_embedding, top_k = min(fetch_k , self.index.ntotal))# call seacrh() to get many candidates 
        #search candidate chunk
        if not candidates:
             return []
        
        cand_embeds = np.array([self._get_embedding(c.chunk_id) for c in candidates] , dtype=np.float32)#reconstruct orginial vector
        selected_indices : List[int] = []#selected 
        remaining = List(range(len(candidates)))

        while len(selected_indices) < top_k and remaining:
            if not selected_indices:
                  best = max(remaining , key = lambda i: candidates[i].score)

            else:
                  sel_embeds = cand_embeds[selected_indices]
                  best_score = -np.inf
                  best = remaining[0]
                  for i in remaining:
                       rel = candidates[i].score
                       sim = float(np.max(cand_embeds[i] @ sel_embeds.T))
                       mmr_score = lambda_mult * rel - (1 - lambda_mult) * sim
                       if mmr_score > best_score:
                            best_score = mmr_score
                            best = i

            selected_indices.append(best)
            remaining.remove(best)


        return [candidates[i] for i in selected_indices]          


    def _get_embedding(self, idx: int) -> np.ndarray:
        # reconstruct stored emb from flat faiss index         
         emb = np.zeros(self.dimension, dtype=np.float32)
         self.index.reconstruct(idx,emb)
         return emb
    
    @property
    def total_vectors(self) -> int:
         return self.index.ntotal
    
    def reset(self) -> None:
         #clearn all vector and metadata
         self.index.reset()
         self._texts.clear()
         self.sources.clear()
         self._pages.clear()
