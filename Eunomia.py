import os
import sys
import json

from dotenv import load_dotenv
from typing import Callable

from langchain.llms import GPT4All
from langchain.vectorstores import Chroma
from langchain.callbacks import StdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.vectorstores.base import VectorStoreRetriever

from src.ingest import Ingestor

from src.constants import CHROMA_SETTINGS


class Eunomia:
    def __init__(self) -> None:
        self.cwd = self.get_cwd()
        self.db = f"{self.cwd}\{os.environ.get('PERSIST_DIRECTORY')}".strip()
        self.llm = os.environ.get("LLM")
        self.backend = os.environ.get("BACKEND")
        self.embeddings_model = os.environ.get("EMBEDDINGS_MODEL")
        self.ignore_folders = json.loads(os.environ.get("IGNORE_FOLDERS"))

        self.model_n_ctx = int(os.environ.get("MODEL_N_CTX"))
        self.target_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS"))

    def get_cwd(self) -> str:
        current_working_dir = os.getcwd()
        return current_working_dir

    def ingest(self) -> None:
        """
        Initialize an Ingestor instance and creates/updates the vectostore.

        :return: None
        """
        ingestor = Ingestor(
            self.cwd,
            self.db,
            self.embeddings_model,
            self.ignore_folders,
        )

        ingestor.ingest()

    def start(self) -> None:
        """
        Initializes the necessary components for the LLM to start. This function uses the following components:

        - HuggingFace embeddings for NLP processing using the specified `model_name`
        - Chroma for retrieving and storing the embeddings
        - GPT4All language model for conversational responses
        - Conversational retrieval chain for generating responses

        Parameters:
        - self: The instance of the LLM class.

        Returns:
        - None
        """
        embeddings = self._initialize_embeddings(self.embeddings_model)
        chroma = self._initialize_chroma(self.db, embeddings)
        retriever = self._initialize_retriever(chroma, self.target_chunks)
        llm = self._initialize_llm(self.llm, self.model_n_ctx, self.backend)

        qa = self._initialize_qa(llm, retriever)

        print(
            r"""
     ______   __  __   __   __   ______   __    __   __   ______    
    /\  ___\ /\ \/\ \ /\ "-.\ \ /\  __ \ /\ "-./  \ /\ \ /\  __ \   
    \ \  __\ \ \ \_\ \\ \ \-.  \\ \ \/\ \\ \ \-./\ \\ \ \\ \  __ \  
     \ \_____\\ \_____\\ \_\\"\_\\ \_____\\ \_\ \ \_\\ \_\\ \_\ \_\ 
      \/_____/ \/_____/ \/_/ \/_/ \/_____/ \/_/  \/_/ \/_/ \/_/\/_/ 
            """
        )

        while True:
            query = input("\nEnter a query: ")
            if query in ["quit", "q"]:
                break

            response = qa({"question": query})

    def _initialize_embeddings(self, model_name: str) -> HuggingFaceEmbeddings:
        """
        Initializes HuggingFaceEmbeddings with the specified model name.

        :param model_name: A string representing the name of the model to use for embedding initialization.
        :type model_name: str

        :return: An instance of HuggingFaceEmbeddings initialized with the specified model name.
        :rtype: HuggingFaceEmbeddings
        """
        return HuggingFaceEmbeddings(model_name=model_name)

    def _initialize_chroma(
        self, persist_directory: str, embedding_function: Callable
    ) -> Chroma:
        """
        Initializes a Chroma object with the given persist directory, embedding function, and client settings.

        :param persist_directory: A string representing the path to the directory where Chroma's data will be persisted.
        :param embedding_function: A callable that takes a string as input and returns a numpy array representing the embedding of the input.

        :return: A Chroma object.
        """
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            client_settings=CHROMA_SETTINGS,
        )

    def _initialize_retriever(
        self, chroma: Chroma, target_chunks: int
    ) -> VectorStoreRetriever:
        """
        Initializes a retriever object to initialize the ConversationalRetrievealChain instance.

        :param chroma: A Chroma instance to be used to initialize the ConversationalRetrievalChain.
        :type chroma: Chroma

        :param target_chunks: The target number of chunks.
        :type target_chunks: int

        :return: A VectorStoreRetriever object.
        """
        return chroma.as_retriever(
            search_kwargs={
                "k": target_chunks,
                "fetch_k": 20,
                "maximal_marginal_relevance": True,
            }
        )

    def _initialize_llm(self, model: str, n_ctx: int, backend: str) -> GPT4All:
        """
        Initializes a GPT4All model with the given parameters.

        :param model: The name of the GPT4All model to use.
        :param n_ctx: The size of the input context for the model.
        :param backend: The backend to use for the model.

        :return: The initialized GPT4All model.
        """
        return GPT4All(
            model=model,
            n_ctx=n_ctx,
            backend=backend,
            verbose=True,
            callbacks=[StdOutCallbackHandler()],
            n_threads=8,
            temp=0.5,
        )

    def _initialize_qa(
        self, llm: GPT4All, retriever: ConversationalRetrievalChain
    ) -> ConversationalRetrievalChain:
        """
        Initializes a ConversationalRetrievalChain object.

        :param llm: A GPT4All model which will be used for conversational retrieval.
        :type llm: GPT4All
        :param retriever: A ConversationalRetrievalChain object which the new object will be based on.
        :type retriever: ConversationalRetrievalChain

        :return: A new ConversationalRetrievalChain object created from the GPT4All model and ConversationalRetrievalChain object.
        :rtype: ConversationalRetrievalChain
        """
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        return ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory
        )


if __name__ == "__main__":
    load_dotenv()

    eunomia = Eunomia()

    args = sys.argv

    if len(args) > 2:
        raise ValueError("Expected one argument and got more.")

    param = args[1].strip()

    if param == "-s" or param == "start":
        eunomia.start()

    elif param == "-i" or param == "ingest":
        eunomia.ingest()
