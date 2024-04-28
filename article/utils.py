from pathlib import Path

# Загрузчики.
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

# Векторные базы данных.
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

# Разбивка текста на куски.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Базовые классы для типизации.
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# Распознование файлов.
import magic


def file_to_vector_store(
        file_path: Path,
        embedding: Embeddings,
        file_type: str=None,
        splitter_parameters: dict={"chunk_size": 1024, "chunk_overlap": 512},
    ) -> VectorStore:
    """
    Прочитать файл и конвертировать текстовые данные в векторную базу данных.

    Параметры
    ---------
    fila_path : Path
        Путь к файлу.
    embedding : embedding: langchain_core.embeddings.embeddings.Embeddings
        Модель для получения эмбеддингов.
    splitter_parameters : dict
        Настройки разбивки текста.

    Возвращает
    ----------
    vector_store : langchain_core.vectorstores.vectorstore
        Векторная база данных.
    """

    # Автоматическое определение типа файла по расширению.
    if file_type is None:
        file_type = magic.from_file(file_path, mime=True)

    implemented_file_types = ["application/pdf"]
    if not (file_type in implemented_file_types):
        raise NotImplementedError(
            "Векторизация реализована только для следующих документов: {str(implemented_file_types)}"
        )

    if file_type == "application/pdf":
        pdf_loader = PyPDFLoader(file_path)
        pages      = pdf_loader.load_and_split()

    # Разбивка текста.
    text_splitter = RecursiveCharacterTextSplitter(**splitter_parameters)
    documents     = text_splitter.split_documents(pages)

    # Векторная база данных.
    vector_store = Chroma.from_documents(
        documents,
        embedding,
        client_settings=Settings(anonymized_telemetry=False),
    )

    return vector_store

