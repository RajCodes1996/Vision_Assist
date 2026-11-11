"""
graph.py
--------
LangGraph retrieval pipeline for Lumen Reader.
Implements: label-aware retrieval, corrective RAG, multi-turn memory.

Flow:
  classify_query → retrieve → grade_docs → 
      ↓ (good)        ↓ (poor)
  generate         rewrite_query → retrieve (retry once)
"""

import os
from typing import List, TypedDict, Annotated, Optional
import operator

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

GROQ_MODEL  = os.getenv("GROQ_MODEL", "llama3-8b-8192")
GROQ_KEY    = os.getenv("GROQ_API_KEY", "")


# ════════════════════════════════════════════════════════════
#  STATE — shared across all graph nodes
# ════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    # Input
    question        : str
    filename_filter : Optional[str]

    # Detected
    detected_label  : str
    rewritten_query : str

    # Retrieved
    retrieved_chunks: List[dict]
    retry_count     : int

    # Grading
    docs_relevant   : bool

    # Output
    answer          : str
    sources         : List[dict]

    # Memory — conversation history
    history         : Annotated[List[dict], operator.add]


# ════════════════════════════════════════════════════════════
#  NODE FUNCTIONS
# ════════════════════════════════════════════════════════════

def node_classify_query(state: RAGState, store, classifier) -> dict:
    """
    Node 1: Detect the label of the user's question.
    Uses the same LabelClassifier from Phase 3.
    """
    question = state["question"]
    label, conf = classifier.predict(question)

    # Low confidence → fall back to no label filter
    if conf < 0.30:
        label = None

    print(f"[Graph] Query label: {label} (conf: {conf:.2f})")
    return {
        "detected_label" : label or "General",
        "rewritten_query": question,   # first pass uses original
        "retry_count"    : 0,
    }


def node_retrieve(state: RAGState, store) -> dict:
    """
    Node 2: Retrieve top-K chunks from Qdrant using label filter.
    """
    query  = state["rewritten_query"]
    label  = state["detected_label"]
    fname  = state.get("filename_filter")

    # Don't filter by label if it's General or retry
    label_filter = label if (label != "General" and state["retry_count"] == 0) else None

    chunks = store.search(
        query=query,
        top_k=5,
        label_filter=label_filter,
        filename_filter=fname,
    )

    print(f"[Graph] Retrieved {len(chunks)} chunks "
          f"(label_filter={label_filter}, retry={state['retry_count']})")
    return {"retrieved_chunks": chunks}


def node_grade_docs(state: RAGState, llm: ChatGroq) -> dict:
    """
    Node 3: Grade retrieved docs for relevance.
    Uses LLM to decide if chunks actually answer the question.
    Falls back to score-based grading if LLM fails.
    """
    question = state["question"]
    chunks   = state["retrieved_chunks"]

    if not chunks:
        return {"docs_relevant": False}

    # Fast score-based check first (no LLM call needed if scores are strong)
    avg_score = sum(c["score"] for c in chunks) / len(chunks)
    if avg_score >= 0.55:
        print(f"[Graph] Docs graded RELEVANT by score (avg={avg_score:.2f})")
        return {"docs_relevant": True}

    # LLM-based grading for borderline cases
    context = "\n\n".join([c["text"][:300] for c in chunks[:3]])
    try:
        messages = [
            SystemMessage(content=(
                "You are a relevance grader. "
                "Answer ONLY with 'yes' or 'no'. "
                "Are the provided documents relevant to answer the question?"
            )),
            HumanMessage(content=(
                f"Question: {question}\n\n"
                f"Documents:\n{context}"
            )),
        ]
        response = llm.invoke(messages)
        relevant = "yes" in response.content.lower()
        print(f"[Graph] Docs graded by LLM: {'RELEVANT' if relevant else 'NOT RELEVANT'}")
        return {"docs_relevant": relevant}
    except Exception as e:
        print(f"[Graph] LLM grading failed ({e}), using score fallback")
        return {"docs_relevant": avg_score >= 0.35}


def node_rewrite_query(state: RAGState, llm: ChatGroq) -> dict:
    """
    Node 4: Rewrite the query to improve retrieval on retry.
    Called when retrieved docs are not relevant.
    """
    question = state["question"]
    try:
        messages = [
            SystemMessage(content=(
                "You are a query rewriter for a document retrieval system. "
                "Rewrite the question to be more specific and retrieval-friendly. "
                "Return ONLY the rewritten question, nothing else."
            )),
            HumanMessage(content=f"Original question: {question}"),
        ]
        response     = llm.invoke(messages)
        rewritten    = response.content.strip()
        print(f"[Graph] Query rewritten: {rewritten[:80]}...")
    except Exception as e:
        print(f"[Graph] Rewrite failed ({e}), using original")
        rewritten = question

    return {
        "rewritten_query": rewritten,
        "retry_count"    : state["retry_count"] + 1,
        "detected_label" : "General",   # broaden on retry
    }


def node_generate(state: RAGState, llm: ChatGroq) -> dict:
    """
    Node 5: Generate final answer from retrieved chunks.
    Builds citations and updates conversation history.
    """
    question = state["question"]
    chunks   = state["retrieved_chunks"]
    history  = state.get("history", [])

    if not chunks:
        return {
            "answer" : "I could not find relevant information in the document to answer your question.",
            "sources": [],
            "history": [{"role": "user",      "content": question},
                        {"role": "assistant",  "content": "No relevant chunks found."}],
        }

    # Build context from top chunks
    context_parts = []
    for i, c in enumerate(chunks[:4], 1):
        context_parts.append(
            f"[Source {i} | Page {c['page_hint']} | {c['label']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Build conversation history string for multi-turn memory
    history_str = ""
    if history:
        recent = history[-6:]   # last 3 turns (user + assistant pairs)
        history_str = "\n".join(
            f"{h['role'].upper()}: {h['content']}" for h in recent
        )

    try:
        messages = [
            SystemMessage(content=(
                "You are an accessibility-focused document assistant helping "
                "visually impaired users understand documents. "
                "Answer clearly and concisely using ONLY the provided context. "
                "If the context doesn't contain the answer, say so honestly. "
                "Always mention which page the information comes from."
            )),
            HumanMessage(content=(
                f"{f'Conversation so far:{chr(10)}{history_str}{chr(10)}{chr(10)}' if history_str else ''}"
                f"Context from document:\n{context}\n\n"
                f"Question: {question}"
            )),
        ]
        response = llm.invoke(messages)
        answer   = response.content.strip()
    except Exception as e:
        # Extractive fallback — no LLM needed
        print(f"[Graph] LLM generation failed ({e}), using extractive fallback")
        answer = chunks[0]["text"]

    # Build source citations
    sources = [
        {
            "chunk_id" : c["chunk_id"],
            "filename" : c["filename"],
            "page"     : c["page_hint"],
            "label"    : c["label"],
            "score"    : c["score"],
            "preview"  : c["text"][:120],
        }
        for c in chunks[:4]
    ]

    # Update memory
    new_history = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    print(f"[Graph] Answer generated ({len(answer)} chars), {len(sources)} sources")
    return {
        "answer" : answer,
        "sources": sources,
        "history": new_history,
    }


# ════════════════════════════════════════════════════════════
#  ROUTING — conditional edges
# ════════════════════════════════════════════════════════════

def route_after_grading(state: RAGState) -> str:
    """
    Route to generate if docs are relevant.
    Route to rewrite if not (max 1 retry to avoid infinite loop).
    """
    if state["docs_relevant"]:
        return "generate"
    if state["retry_count"] >= 1:
        print("[Graph] Max retries reached — generating with what we have")
        return "generate"
    return "rewrite_query"


# ════════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ════════════════════════════════════════════════════════════

def build_rag_graph(store, classifier):
    """
    Build and compile the LangGraph RAG pipeline.

    Args:
        store     : VectorStore instance (Phase 3)
        classifier: LabelClassifier instance (Phase 3)

    Returns:
        Compiled LangGraph app
    """
    llm = ChatGroq(
        api_key=GROQ_KEY,
        model_name=GROQ_MODEL,
        temperature=0.1,
        max_tokens=1024,
    )

    graph = StateGraph(RAGState)

    # ── Register nodes (bind dependencies via closures) ────────────
    graph.add_node("classify_query", lambda s: node_classify_query(s, store, classifier))
    graph.add_node("retrieve",       lambda s: node_retrieve(s, store))
    graph.add_node("grade_docs",     lambda s: node_grade_docs(s, llm))
    graph.add_node("rewrite_query",  lambda s: node_rewrite_query(s, llm))
    graph.add_node("generate",       lambda s: node_generate(s, llm))

    # ── Define edges ───────────────────────────────────────────────
    graph.set_entry_point("classify_query")
    graph.add_edge("classify_query", "retrieve")
    graph.add_edge("retrieve",       "grade_docs")
    graph.add_conditional_edges(
        "grade_docs",
        route_after_grading,
        {
            "generate"    : "generate",
            "rewrite_query": "rewrite_query",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate",      END)

    return graph.compile()


# ════════════════════════════════════════════════════════════
#  CONVENIENCE RUNNER
# ════════════════════════════════════════════════════════════

def run_query(
    app,
    question: str,
    history: List[dict] = None,
    filename_filter: str = None,
) -> dict:
    """
    Run a single query through the RAG graph.

    Args:
        app            : Compiled LangGraph app from build_rag_graph()
        question       : User's question string
        history        : Previous conversation turns (for multi-turn memory)
        filename_filter: Restrict search to one document

    Returns:
        Dict with 'answer', 'sources', 'history', 'detected_label'
    """
    initial_state: RAGState = {
        "question"        : question,
        "filename_filter" : filename_filter,
        "detected_label"  : "",
        "rewritten_query" : question,
        "retrieved_chunks": [],
        "retry_count"     : 0,
        "docs_relevant"   : False,
        "answer"          : "",
        "sources"         : [],
        "history"         : history or [],
    }

    final_state = app.invoke(initial_state)
    return {
        "answer"        : final_state["answer"],
        "sources"       : final_state["sources"],
        "history"       : final_state["history"],
        "detected_label": final_state["detected_label"],
    }


# ════════════════════════════════════════════════════════════
#  QUICK TEST
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    from ingestion import load_document, chunk_document
    from labelling.labeller import LabelClassifier
    from vectordb.store import VectorStore

    TEST_FILE = "data/uploads/1.8_Diligence_Summary.pdf"

    # Setup
    print("\n=== Setting up pipeline ===")
    raw_text, meta  = load_document(TEST_FILE)
    chunks          = chunk_document(raw_text, filename=meta["filename"])
    clf             = LabelClassifier()
    clf.train()
    chunks          = clf.label_chunks(chunks)
    store           = VectorStore(in_memory=True)
    store.upsert_chunks(chunks)

    # Build graph
    print("\n=== Building RAG graph ===")
    app = build_rag_graph(store, clf)

    # Turn 1
    print("\n=== Turn 1 ===")
    result  = run_query(app, "What is AI diligence?")
    history = result["history"]
    print(f"\nLabel   : {result['detected_label']}")
    print(f"Answer  : {result['answer'][:300]}")
    print(f"Sources : {len(result['sources'])} chunks")

    # Turn 2 — tests multi-turn memory
    print("\n=== Turn 2 (follow-up) ===")
    result2 = run_query(app, "Can you elaborate on that?", history=history)
    print(f"\nAnswer  : {result2['answer'][:300]}")
    print(f"History : {len(result2['history'])} turns stored")

    print("\nPhase 4 RAG graph OK.")