# rag_pipeline.py
from retriever import retrieve_top_k
from prompt import PROMPT_TEMPLATE
from generator import generate_answer

def run_rag(question, embedder, vector_store, llm, k=5):
    retrieved_chunks = retrieve_top_k(
        question, embedder, vector_store, k
    )

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    answer = generate_answer(llm, prompt)

    return answer, retrieved_chunks
