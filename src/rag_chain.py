from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_MODEL, require_google_api_key


RAG_PROMPT = """
Você é um chatbot corporativo baseado em uma base de conhecimento em PDF.
Responda à pergunta do usuário utilizando exclusivamente as informações presentes no contexto fornecido.
Regras obrigatórias:
1. Não invente informações.
2. Não utilize conhecimento externo.
3. Se a resposta não estiver claramente presente no contexto, diga exatamente:
   "Não encontrei essa informação no documento fornecido."
4. Não faça deduções além do conteúdo do contexto.
5. Responda de forma clara, objetiva e profissional.
Contexto:
{context}
Pergunta:
{input}
Resposta:
"""


def create_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,  # Modelo utilizado
        temperature=0,  # Controla a criatividade do modelo
        google_api_key=require_google_api_key(),  # Chave da API
    )
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)  # Prompt para o modelo
    generation_chain = prompt | llm | StrOutputParser()  # Chain de geração de resposta

    class SimpleRagChain:
        def invoke(self, inputs):
            question = inputs["input"]  # Pergunta do usuário
            documents = retriever.invoke(question)  # Recupera documentos relevantes
            context = "\n\n".join(document.page_content for document in documents)  # Contexto
            answer = generation_chain.invoke({"context": context, "input": question})  # Gera a resposta

            return {
                "answer": answer,  # Resposta gerada pelo modelo
                "context": documents,  # Trechos utilizados como fonte
            }

    return SimpleRagChain()  # Retorna a chain de RAG
