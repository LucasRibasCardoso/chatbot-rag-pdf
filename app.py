from pathlib import Path

import streamlit as st

from src.config import DATA_DIR, VECTORSTORE_PATH, require_google_api_key
from src.pdf_loader import load_pdf, split_documents
from src.rag_chain import create_rag_chain
from src.vector_store import create_vector_store


st.set_page_config(
    page_title="Chatbot RAG PDF",
    layout="centered",
)


def initialize_session_state() -> None:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False


def save_uploaded_pdf(uploaded_file) -> Path:
    file_path = DATA_DIR / "documento_enviado.pdf"
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def show_sources(source_documents) -> None:
    with st.expander("Trechos utilizados como fonte"):
        for index, document in enumerate(source_documents, start=1):
            page = document.metadata.get("page")
            page_label = page + 1 if isinstance(page, int) else "não disponível"

            st.markdown(f"**Trecho {index} - Página: {page_label}**")
            st.write(document.page_content)


initialize_session_state()

st.title("Chatbot Corporativo Inteligente com Base de Conhecimento RAG")
st.write(
    "Envie um PDF, processe o documento e faça perguntas com respostas baseadas "
    "exclusivamente no conteúdo recuperado."
)

uploaded_file = st.file_uploader("Envie um arquivo PDF", type=["pdf"])

if st.button("Processar PDF", type="primary"):
    if uploaded_file is None:
        st.warning("Envie um arquivo PDF antes de processar.")
    else:
        try:
            require_google_api_key()

            with st.spinner("Processando PDF e criando índice vetorial..."):
                pdf_path = save_uploaded_pdf(uploaded_file)
                documents = load_pdf(str(pdf_path))
                chunks = split_documents(documents)
                vector_store = create_vector_store(chunks)
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                rag_chain = create_rag_chain(retriever)

            st.session_state.vector_store = vector_store
            st.session_state.rag_chain = rag_chain
            st.session_state.pdf_processed = True

            st.success(f"PDF processado com sucesso. Índice salvo em {VECTORSTORE_PATH}.")
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Não foi possível processar o PDF. Detalhes: {error}")

question = st.text_input("Digite sua pergunta")

if st.button("Perguntar"):
    if not st.session_state.pdf_processed or st.session_state.rag_chain is None:
        st.warning("É necessário carregar e processar um PDF antes de fazer perguntas.")
    elif not question.strip():
        st.warning("Digite uma pergunta antes de enviar.")
    else:
        try:
            with st.spinner("Buscando informações no documento..."):
                result = st.session_state.rag_chain.invoke({"input": question})

            st.subheader("Resposta")
            st.write(result.get("answer", "Não foi possível gerar uma resposta."))

            source_documents = result.get("context", [])
            if source_documents:
                show_sources(source_documents)
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Não foi possível responder à pergunta. Detalhes: {error}")
