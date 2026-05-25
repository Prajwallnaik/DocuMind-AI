from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def create_qa_chain(retriever):
    """Creates a RetrievalQA chain using Google Gemini (via OpenRouter or native API)."""
    import os
    
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="google/gemini-2.5-flash",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=2048,
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=os.environ.get("GOOGLE_API_KEY", "missing_key")
        )
    
    template = """Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise, professional, and grounded in the provided context.

CRITICAL: Do not use any emojis in your response. Keep the tone completely objective, clean, and professional.

Context:
{context}

Question: {question}
Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain
