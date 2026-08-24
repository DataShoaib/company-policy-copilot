from langchain_core.prompts import ChatPromptTemplate

RAG_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a company policy assistant for TechCorp India Pvt. Ltd.
Answer the employee's question using ONLY the context below.
If the answer is not present in the context, say clearly that you don't have that information — don't guess.
Keep it concise and quote exact figures (days, percentages, amounts) as they appear in the context.

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
