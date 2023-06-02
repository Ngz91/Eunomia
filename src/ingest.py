import os
import glob
import concurrent.futures

from typing import List
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    PythonLoader,
    NotebookLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from src.constants import CHROMA_SETTINGS

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ipynb": (NotebookLoader, {}),
    ".py": (PythonLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".cpp": (TextLoader, {"encoding": "utf8"}),
    ".hpp": (TextLoader, {"encoding": "utf8"}),
    ".css": (TextLoader, {"encoding": "utf8"}),
}

LANG_MAPPINGS = {
    "py": Language.PYTHON,
    "cpp": Language.CPP,
    "hpp": Language.CPP,
    "js": Language.JS,
    "html": Language.HTML,
    "md": Language.MARKDOWN,
    "rb": Language.RUBY,
    "rst": Language.RUST,
}


class Ingestor:
    def __init__(
        self, cwd: str, db: str, emb_model: str, ignore_folders: List[str]
    ) -> None:
        self.cwd = cwd
        self.db = db
        self.emb_model = emb_model
        self.ignore_folders = ignore_folders

        self.chunk_size = 1000
        self.chunk_overlap = 4

        self.threshold = 5242880  # 5 MB in bytes

    def load_single_document(self, file_path: str) -> Document:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()[0]

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(self) -> List[Document]:
        all_files = []
        for ext in LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(self.cwd, f"**/*{ext}"), recursive=True)
            )
        filtered_files = []

        for file_path in all_files:
            if not any(
                ignore_folder in file_path for ignore_folder in self.ignore_folders
            ):
                filtered_files.append(file_path)

        results = []

        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for file_path in filtered_files:
                file_size = os.path.getsize(file_path)
                if file_size > self.threshold:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        future = executor.submit(self.load_single_document, file_path)
                        results.append(future.result())
                else:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self.load_single_document, file_path)
                        results.append(future.result())
                pbar.update()

        return results

    def split_docs(
        self, docs_list: List[Document], language: str = ""
    ) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        if language != "":
            text_splitter.from_language(language)
        texts = text_splitter.split_documents(docs_list)
        return texts

    def process_documents(self, ignored_files: List[str] = []) -> List[Document]:
        """
        Load documents and split in chunks
        """
        doc_dict = {}
        print(f"Loading documents from {self.cwd}")
        documents = self.load_documents()
        if not documents:
            print("No new documents to load")
            exit(0)
        print(f"Loaded {len(documents)} new documents from {self.cwd}")
        for doc in documents:
            ext = doc.metadata["source"].split(".")[-1]
            if ext not in doc_dict:
                doc_dict[ext] = []
            doc_dict[ext].append(doc)
        all_docs = []
        for ext, docs in doc_dict.items():
            split_docs = self.split_docs(docs, language=LANG_MAPPINGS[ext])
            all_docs.extend(split_docs)
        print(
            f"Split into {len(all_docs)} chunks of text (max. {self.chunk_size} tokens each)"
        )
        return all_docs

    def does_vectorstore_exist(self) -> bool:
        """
        Checks if vectorstore exists.
        """
        index_path = os.path.join(self.db, "index")
        if os.path.exists(index_path):
            collections_path = os.path.join(self.db, "chroma-collections.parquet")
            embeddings_path = os.path.join(self.db, "chroma-embeddings.parquet")
            if os.path.exists(collections_path) and os.path.exists(embeddings_path):
                index_files = glob.glob(os.path.join(index_path, "*.bin")) + glob.glob(
                    os.path.join(index_path, "*.pkl")
                )
                # At least 3 documents are needed in a working vectorstore.
                if len(index_files) > 3:
                    return True
        return False

    def ingest(self):
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=self.emb_model)

        if self.does_vectorstore_exist():
            # Update and store locally vectorstore
            print(f"Appending to existing vectorstore at {self.db}")
            db = Chroma(
                persist_directory=self.db,
                embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS,
            )
            collection = db.get()
            texts = self.process_documents(
                [metadata["source"] for metadata in collection["metadatas"]]
            )
            print(f"Creating embeddings. May take some minutes...")
            db.add_documents(texts)
        else:
            # Create and store locally vectorstore
            print("Creating new vectorstore")
            texts = self.process_documents()
            print(f"Creating embeddings. May take some minutes...")
            db = Chroma.from_documents(
                texts,
                embeddings,
                persist_directory=self.db,
                client_settings=CHROMA_SETTINGS,
            )
        db.persist()
        db = None

        print(
            f"Ingestion complete, you can now run 'eunomia start' to use the LLM to interact with your code!"
        )
