from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL, VECTORSTORE_PATH


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_store(documents):
    if not documents:
        raise ValueError("Não há trechos do documento para indexar.")

    vector_store = FAISS.from_documents(documents, get_embeddings())
    vector_store.save_local(str(VECTORSTORE_PATH))

    return vector_store


def load_vector_store():
    if not VECTORSTORE_PATH.exists():
        raise ValueError("Nenhum índice FAISS local foi encontrado.")

    return FAISS.load_local(
        str(VECTORSTORE_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
