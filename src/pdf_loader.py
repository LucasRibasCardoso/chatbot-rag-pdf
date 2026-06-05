from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str):
    try:
        documents = PyPDFLoader(file_path).load()
    except Exception as error:
        raise ValueError(
            "Não foi possível ler o PDF. Verifique se o arquivo é válido."
        ) from error

    if not documents:
        raise ValueError("O PDF não possui páginas para processar.")

    documents_with_text = [
        document for document in documents if document.page_content.strip()
    ]

    if not documents_with_text:
        raise ValueError("O PDF não possui texto extraível.")

    return documents_with_text


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Número de caracteres em cada chunk
        chunk_overlap=150,  # Número de caracteres que se sobrepõem entre chunks
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Não foi possível dividir o texto do PDF em trechos.")

    return chunks
