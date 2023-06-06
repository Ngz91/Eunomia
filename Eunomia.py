import os
import sys
import json

from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import StdOutCallbackHandler

from src.ingest import Ingestor

from src.constants import CHROMA_SETTINGS

load_dotenv()


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
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
        chroma = Chroma(
            persist_directory=self.db,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
        retriever = chroma.as_retriever(
            search_kwargs={
                "k": self.target_chunks,
                "fetch_k": 20,
                "maximal_marginal_relevance": True,
            }
        )
        llm = GPT4All(
            model=self.llm,
            n_ctx=self.model_n_ctx,
            backend=self.backend,
            verbose=True,
            callbacks=[StdOutCallbackHandler()],
            n_threads=8,  # Change this according to your cpu threads
            temp=0.5,
        )

        qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

        chat_history = []

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

            response = qa({"question": query, "chat_history": chat_history})
            chat_history.append((query, response["answer"]))


if __name__ == "__main__":
    eunomia = Eunomia()

    args = sys.argv

    if len(args) > 2:
        raise ValueError("Expected one argument and got more.")

    param = args[1].strip()

    if param == "-s" or param == "start":
        eunomia.start()

    elif param == "-i" or param == "ingest":
        eunomia.ingest()
