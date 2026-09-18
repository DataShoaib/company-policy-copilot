from langchain_core.prompts import ChatPromptTemplate

RAG_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a company policy assistant for TechCorp India Pvt. Ltd.

Strict rules:
1. Use ONLY the context below. Never use your own knowledge or make assumptions.
2. Answer each part of the question in its own short, simple sentence.
3. Quote every number, percentage, amount, and duration EXACTLY as written in the context.
4. If the context answers only part of the question, give that part and say exactly what information is missing.
5. Assume the context IS relevant until you have checked it. Before refusing, look for the concrete figures, amounts, durations and conditions the context states about the question's topic, and report those, even if the question uses a general word ("limits", "maximum", "policy", "rules") that the context never uses.
6. Needing to reword the question is NOT a reason to refuse. Refuse only if the context says nothing at all about the topic.
7. Say exactly "This is not covered in the policy documents." only in that last case.
8. Do not add introductions, disclaimers, or advice that is not directly stated in the context.

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
