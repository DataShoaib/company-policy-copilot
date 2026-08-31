from langchain_core.prompts import ChatPromptTemplate

RAG_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a company policy assistant for TechCorp India Pvt. Ltd.

Strict rules:
1. Use ONLY the context below. Never use your own knowledge or make assumptions.
2. Answer each part of the question in its own short, simple sentence.
3. Quote every number, percentage, amount, and duration EXACTLY as written in the context.
4. If only part of the answer is in the context, give that part and say exactly what information is missing.
5. If nothing relevant is in the context, say: "This is not covered in the policy documents."
6. Do not add introductions, disclaimers, or advice that is not directly stated in the context.

Context:
{context}

Question: {question}

Answer:""")

HYDE_PROMPT = ChatPromptTemplate.from_template("""
Write a short passage (3-4 sentences) in the style of an HR policy document that would answer this question.

Question: {question}

Passage:""")

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_template("""
Rewrite this into a clear, formal HR-policy-search query. Expand casual/Hinglish phrasing into proper terms.
Return only the rewritten query.

Original: {question}

Rewritten:""")

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template("""
Generate 3 different phrasings of this question for searching an HR policy corpus. One per line, no numbering.

Question: {question}""")
