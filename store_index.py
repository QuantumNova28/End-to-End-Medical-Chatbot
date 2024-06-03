from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from langchain.vectorstores import Pinecone as LangchainPinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
# from pinecone import Pinecone, ServerlessSpec
# pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key = os.environ.get("PINECONE_API_KEY")
)

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=LangchainPinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)