
from .deepseek_client import DeepSeekClient
from .db import search_similar
from .embedding_model import embed_texts
from typing import List, Optional
SYSTEM_PROMPT = """You are a helpful assistant answering questions based only on the provided context from a PDF. 
If the answer is not in the context, say you don't know."""

def answer_question(
    question: str,
    top_k: int = 5,
    doc_ids: Optional[List[str]] = None,
) -> str:
    client = DeepSeekClient()

    
    [q_emb] = embed_texts([question])

    
    contexts = search_similar(q_emb, top_k=top_k, doc_ids=doc_ids)

    if not contexts:
        return "I couldn't find any relevant context in the indexed documents."

    context_text = "\n\n".join(contexts)

    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
    ]
    return client.chat(messages)

if __name__ == "__main__":
    while True:
        q = input("Question (empty to quit): ").strip()
        if not q:
            break
        print("Answer:\n", answer_question(q))  
        print()
