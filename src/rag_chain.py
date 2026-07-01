"""
RAG Chain
─────────
Orchestrates retrieval → prompt construction → LLM call → answer.

Flow
----
1. Embed the user query.
2. Retrieve top-k chunks from FAISSVectorStore (with optional MMR).
3. Build a structured prompt with the retrieved context.
4. Call Groq LLaMA-3 via the OpenAI-compatible SDK.
5. Return the answer + source chunks.
"""

from __future__ import annotations

import functools
import textwrap#format multi line strings neatly
from typing import Generator, List, Tuple,Union
#Genertor return value one by one(Streaming)

from .vector_store import FAISSVectorStore, RetrievalResult
from .embedder import Embedder


# ── Prompt template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are DocuMind AI, an expert document analyst and question-answering assistant.
    Your job is to answer questions accurately and concisely using ONLY the context
    excerpts provided below.

    Rules:
    1. Answer ONLY from the provided context. Never fabricate information.
    2. If the answer is not found in the context, say:
       "I couldn't find relevant information in the document for this question."
    3. Quote short, key phrases directly from the document when helpful.
    4. Structure long answers with bullet points or numbered lists.
    5. Be concise yet complete. Avoid repeating yourself.
    6. If the question is ambiguous, ask for clarification.
    7. Never hallucinate
""").strip()#rule given grpd model to follow while generating answer


def _build_context_block(results: List[RetrievalResult]) -> str:
    """Format retrieved chunks into a numbered context block."""
    """blocks = [
            "[Excerpt 1 | score=0.982]#header
            Random Forest is...",#r.text

            "[Excerpt 2 | score=0.934]
            Decision Tree is..."
        ]"""
    blocks = []
    for i, r in enumerate(results, 1):#go to every retrieved chunk one by one
        #i -no from 1 and r - one RetrievalResult object
        header = f"[Excerpt {i} | score={r.score:.3f}]"
        blocks.append(f"{header}\n{r.text}")
    return "\n\n---\n\n".join(blocks)



def _build_user_message(query: str, context: str) -> str:
    #combine retrievd context and user que into one message
    #so the model knows:
        #What information it can use (the context).
        #What it needs to answer (the user's question).
    return (
        f"Context excerpts from the document:\n\n{context}"
        f"\n\n---\n\nUser question: {query}"
    )

def _retrieve(vector_store, embedder, query, top_k, use_mmr):
    query_emb = embedder.embed_query(query)
    if use_mmr:
        return vector_store.search_mmr(query_emb, top_k=top_k, fetch_k=top_k * 3)
    return vector_store.search(query_emb, top_k=top_k)

# ── Groq client (cached) ─────────────────────────────────────────────────────

@functools.lru_cache(maxsize=8)
def _get_groq_client(api_key: str):
    """Return a cached Groq client for the given API key. 
        to create groq client once and reuse it"""
    from groq import Groq
    return Groq(api_key=api_key)


# ── Public RAG function ──────────────────────────────────────────────────────

def answer_query(#answer user question
    query: str,
    vector_store: FAISSVectorStore,
    embedder: Embedder,
    groq_api_key: str,
    model: str = "llama3-8b-8192",
    top_k: int = 5,
    temperature: float = 0.2,#control creativity small value more factual
    use_mmr: bool = True,
    stream: bool = True,#answer appear word b word 
) -> Tuple[Union[Generator, str], List[RetrievalResult]]:
    """
    Run the full RAG pipeline.

    Returns
    -------
    answer  : streaming generator (if stream=True) or full string
    sources : list of RetrievalResult used as context
    """
    results = _retrieve(vector_store, embedder, query, top_k, use_mmr)
    
    if not results:
        no_doc = "No document has been indexed yet. Please upload a document first."
        return ((lambda: (yield no_doc))() if stream else no_doc), []

   
    context = _build_context_block(results)#received chunks to context as str
    user_msg = _build_user_message(query, context)#context and user que to user_msg as str
    client = _get_groq_client(groq_api_key)#create groq client once and reuse it 

    #it is liking packing everything 
    """messages = [
            {
                "role": "system",
                "content": "Answer only from the given context."
            },

            {
                "role": "user",
                "content": 
                            Context:
                            Random Forest is an ensemble algorithm.

                            User Question:
                            What is Random Forest?
                    
            }
    ]"""

    messages = [#create conversation sent to Llama
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        stream=stream,
    )

 
    if stream:
        def token_generator():
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return token_generator(), results
    else:
        return response.choices[0].message.content, results

#    if stream:
#        #respomse_stream is stream object
#        response_stream = client.chat.completions.create(
#            model=model,
#            messages=messages,
#            temperature=temperature,
#            max_tokens=1024,
#            stream=True,
#        )#Think of this as sending a message to ChatGPT and press Send. this line does exactly that
#        """ 
#            POST https://api.groq.com/openai/v1/chat/completions
#
#            {
#              "model": "llama3-8b-8192",
#
#              "messages": [
#                  {
#                    "role":"system",
#                    "content":"Answer only from context..."
#                  },
#
#                  {
#                    "role":"user",
#                    "content":"Context...
#
#            User Question:
#            What is Random Forest?"
#                  }
#              ],
#
#              "temperature":0.2,
#
#              "stream":true
#            }
#        """
#
#        def token_generator():
#            for chunk in response_stream:#keep reading whatever come from groq
#                #RECEIVE EVERY CHUNK GIVEN BY GROQ
#                delta = chunk.choices[0].delta.content
#                #Take only the text (content) from the chunk and store it in a variable named delta
#                """
#                Suppose Groq generated: "Random Forest is an ensemble algorithm."
#                Because stream=True it sends small pieces.
#                First iteration:
#                    chunk =
#                        {
#                            "choices": [
#                                {
#                                    "delta": {
#                                        "content": "Random"
#                                    }
#                                }
#                            ]
#                        }
#
#                Second iteration:
#                    chunk =
#                        {
#                            "choices": [
#                                {
#                                    "delta": {
#                                        "content": " Forest"
#                                    }
#                                }
#                            ]
#                        }
#                """
#                if delta:#if None send by groq then conitune not stop 
#                    yield delta
#                    #return = Give everything once and finish
#                    #yield = Give one thing, pause, then continue later. 
#                    #beacause strean true so evry chunk given by groq is received 
#
#        return token_generator(), results
#    else:
#        response = client.chat.completions.create(
#            model=model,
#            messages=messages,
#            temperature=temperature,
#            max_tokens=1024,
#            stream=False,
#        )
#        return response.choices[0].message.content, results#full at once no need of delta 


# ── Conversation-aware variant ───────────────────────────────────────────────

def answer_with_history(
    query: str,
    history: list[dict],          # [{"role": ..., "content": ...}]
    #it accpet history stored somewhwere say in streamit 
    vector_store: FAISSVectorStore,
    embedder: Embedder,
    groq_api_key: str,
    model: str = "llama3-8b-8192",
    top_k: int = 5,
    temperature: float = 0.2,
    use_mmr: bool = True,
    stream: bool = True
) -> Tuple[Union[Generator, str], List[RetrievalResult]]:
    """
    Streaming RAG answer that includes prior conversation turns so the
    model can handle follow-up questions correctly.
    """
    results = _retrieve(vector_store, embedder, query, top_k, use_mmr)
 
    if not results:
        no_doc = "No document indexed. Please upload a document first."
        if stream:
            def _no_doc_gen():
                yield no_doc
            return _no_doc_gen(), []
        return no_doc, []


    context = _build_context_block(results)
    user_msg = _build_user_message(query, context)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Include up to last 6 turns to stay within context limits
    for turn in history[-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_msg})
    """
        first call user ask  : what is random forest?
                  app does  : answer_with_history(
                            query="What is Random Forest?",
                            history=[]
                            )
        History is empty. 

        Llama answers : Random Forest is..
        
        app does : 
            history.append(
            {
            "role":"user",
            "content":"What is Random Forest?"
            }
            )

            history.append(
            {
            "role":"assistant",
            "content":"Random Forest is..."
            }
            )

        now if second quest : Why is it better?
        streamlit call answer_with_history 
        answer_with_history(
            query="Why is it better?",
            history=history
        )
        history already contains previous data so function receive it
        so now this happen
        for turn in history[-6:]:
        messages.append(turn)
        current history 
        history = [
            {
            "role":"user",
            "content":"What is Random Forest?"
            },

            {
            "role":"assistant",
            "content":"Random Forest is..."
            }
        ]
            Loop 1
            turn =
            User: What is Random Forest?
            Append it.
            messages becomes
                System Prompt
                    ↓
                User: What is Random Forest?

            Loop 2
            turn =
            Assistant:Random Forest is...
            Append.
            Now messages becomes
                System Prompt
                    ↓
                User:What is Random Forest?
                    ↓
                Assistant:Random Forest is...

            Then
            messages.append({
            "role":"user",
            "content":user_msg
            })
            adds
            User:Why is it better?

            Final messages
                System Prompt
                    ↓
                User:What is Random Forest?
                    ↓
                Assistant:Random Forest is...
                    ↓
                User:Why is it better?

            THIS is sent to
            client.chat.completions.create(...)
            Now Llama sees the old conversation.
            That is why it remembers.

    """

    client = _get_groq_client(groq_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        stream=stream,
    )
 
    if stream:
        def token_generator():
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return token_generator(), results
    else:
        return response.choices[0].message.content, results

