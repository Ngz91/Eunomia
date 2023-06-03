import os
import glob
import concurrent.futures

from typing import List
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from src.constants import CHROMA_SETTINGS, LOADER_MAPPING, LANG_MAPPINGS


class Ingestor:
    def __init__(
        self,
        cwd: str,
        db: str,
        emb_model: str,
        ignore_folders: List[str],
    ) -> None:
        self.cwd = cwd
        self.db = db
        self.emb_model = emb_model
        self.ignore_folders = ignore_folders

        self.chunk_size = 60
        self.chunk_overlap = 2

        self.threshold = 5242880  # 5 MB in bytes

    def load_single_document(self, file_path: str) -> Document:
        """
        Load a single document from a file path using a loader based on the file extension.

        :param file_path: A string representing the path to the file to be loaded.
        :type file_path: str
        :return: A Document object representing the loaded document.
        :rtype: Document
        :raises ValueError: If the file extension is not supported.
        """
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()[0]

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(self, ignored_files: List[str] = []) -> List[Document]:
        """
        Loads documents from files in the cwd directory and its subdirectories.
        Excludes files in specified ignore folders and only loads files that are above
        a specified file size threshold. Uses multi-processing if file size is above the
        threshold and multi-threading otherwise.

        Returns:
            A list of loaded documents.
        """
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

        # Remove files that are already in the vectorstore if it already exists
        if ignored_files:
            filtered_files = [
                file_path for file_path in all_files if file_path not in ignored_files
            ]

        results = []

        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for file_path in filtered_files:
                file_size = os.path.getsize(file_path)
                if file_size > self.threshold:
                    with ProcessPoolExecutor() as executor:
                        future = executor.submit(self.load_single_document, file_path)
                        results.append(future.result())
                else:
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(self.load_single_document, file_path)
                        results.append(future.result())
                pbar.update()

        return results

    def split_docs(self, docs_list: List[Document], language: str) -> List[Document]:
        """
        Splits a list of documents into smaller chunks using the RecursiveCharacterTextSplitter
        object. The function takes a list of Document objects and a language string as
        parameters. The function returns a List of Document objects.

        :param docs_list: The list of Document objects to be split.
        :type docs_list: List[Document]
        :param language: The language string to be used by the RecursiveCharacterTextSplitter
                        object.
        :type language: str
        :return: A List of Document objects.
        :rtype: List[Document]
        """
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            language=language,
        )
        texts = text_splitter.split_documents(docs_list)
        return texts

    def process_documents(self, ignored_files: List[str] = []) -> List[Document]:
        """
        Process documents and split them into smaller chunks of text.

        :param ignored_files: A list of files to ignore. Default is an empty list.
        :type ignored_files: List[str]
        :return: A list of Document objects after splitting them into smaller chunks.
        :rtype: List[Document]
        """
        doc_dict = {}
        print(f"Loading documents from {self.cwd}")
        documents = self.load_documents(ignored_files=ignored_files)
        if not documents:
            print("No new documents to load")
            exit(0)
        print(f"Loaded {len(documents)} new documents from {self.cwd}")
        with ThreadPoolExecutor() as executor:
            # Create list of futures for each document split
            futures = []
            for doc in documents:
                ext = doc.metadata["source"].split(".")[-1]
                if ext not in doc_dict:
                    doc_dict[ext] = []
                futures.append(
                    executor.submit(self.split_docs, [doc], language=LANG_MAPPINGS[ext])
                )
            # Gather results from futures as they complete
            all_docs = []
            for future in concurrent.futures.as_completed(futures):
                split_docs = future.result()
                all_docs.extend(split_docs)
        print(
            f"Split into {len(all_docs)} chunks of text (max. {self.chunk_size} tokens each)"
        )
        return all_docs

    def does_vectorstore_exist(self) -> bool:
        """
        Check if the vectorstore exists by verifying the existence of the index file and
        the collections and embeddings parquet files.

        :param self: An instance of the class.

        :return: A boolean indicating whether the vectorstore exists or not.
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

    def ingest(self) -> None:
        """
        Ingests documents to create embeddings and store them locally in a vectorstore using Chroma. If the vectorstore
        already exists, updates and appends to it. If it doesn't exist, creates a new vectorstore. Prints progress
        messages to the console.

        Parameters:
        None
        Returns:
        None
        """
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
            f"Vectorstore created, you can now run 'eunomia start' to use the LLM to interact with your code!"
        )
