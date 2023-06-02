import os
from dotenv import load_dotenv
from chromadb.config import Settings

from langchain.document_loaders import (
    TextLoader,
    PythonLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)

from langchain.text_splitter import Language

load_dotenv()

# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY")

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".py": (PythonLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".cpp": (TextLoader, {"encoding": "utf8"}),
    ".hpp": (TextLoader, {"encoding": "utf8"}),
    ".js": (TextLoader, {"encoding": "utf8"}),
    ".rb": (TextLoader, {"encoding": "utf8"}),
    ".rs": (TextLoader, {"encoding": "utf8"}),
    ".java": (TextLoader, {"encoding": "utf8"}),
    ".jar": (TextLoader, {"encoding": "utf8"}),
    ".go": (TextLoader, {"encoding": "utf8"}),
    ".scala": (TextLoader, {"encoding": "utf8"}),
    ".sc": (TextLoader, {"encoding": "utf8"}),
    ".swift": (TextLoader, {"encoding": "utf8"}),
}

# Map supported file extensions to langchain's Language dataclass
LANG_MAPPINGS = {
    "py": Language.PYTHON,
    "cpp": Language.CPP,
    "hpp": Language.CPP,
    "js": Language.JS,
    "html": Language.HTML,
    "md": Language.MARKDOWN,
    "rb": Language.RUBY,
    "rs": Language.RUST,
    "java": Language.JAVA,
    "jar": Language.JAVA,
    "go": Language.GO,
    "scala": Language.SCALA,
    "sc": Language.SCALA,
    "swift": Language.SWIFT,
}
