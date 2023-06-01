import os
import sys
import json

from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from src.ingest import Ingestor

from src.constants import CHROMA_SETTINGS

load_dotenv()


class Eunomia:
    def __init__(self):
        self.cwd = self.get_cwd()
        self.db = f"{self.cwd}\{os.environ.get('PERSIST_DIRECTORY')}".strip()
        self.llm = os.environ.get("LLM")
        self.embeddings_model = os.environ.get("EMBEDDINGS_MODEL")
        self.ignore_folders = json.loads(os.environ.get("IGNORE_FOLDERS"))

        self.model_n_ctx = os.environ.get("MODEL_N_CTX")
        self.target_chunks = os.environ.get("TARGET_SOURCE_CHUNKS")

    def get_cwd(self):
        current_working_dir = os.getcwd()
        return current_working_dir

    def ingest(self):
        ingestor = Ingestor(
            self.cwd, self.db, self.embeddings_model, self.ignore_folders
        )

        ingestor.ingest()

    def start(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
        db = Chroma(
            persist_directory=self.db,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
        retriever = db.as_retriever(search_kwargs={"k": int(self.target_chunks)})
        llm = GPT4All(
            model=self.llm,
            n_ctx=self.model_n_ctx,
            backend="gptj",
            callbacks=StreamingStdOutCallbackHandler,
        )

        qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

        chat_history = []

        while True:
            question = input("\nEnter a query: ")
            if question == "quit" or question == "q":
                break

            res = qa({"question": question, "chat_history": chat_history})
            chat_history.append((question, res["answer"]))


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
