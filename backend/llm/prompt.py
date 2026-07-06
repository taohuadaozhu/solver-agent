def build_prompt(query: str, context_chunks: list):
    context_text = '\n\n'.join([chunk.get('content', '') for chunk in context_chunks])
    return f"""Given the following context from the knowledge base, answer the user query.

Context:
{context_text}

Question:
{query}
"""
