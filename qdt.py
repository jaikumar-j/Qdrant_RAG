from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader= PyPDFLoader("/home/kiwitech/qdrant_vec_rag/gpt-4.pdf")

documents= loader.load()
print(documents[0])
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size= 100,
    chunk_overlap= 50
)
texts=text_splitter.split_documents(documents)
model_name= "BAAI/bge-large-en"
model_kwargs = {"device":"cpu"}
encode_kwargs= {"normalize_embeddings":False}

embeddings= HuggingFaceBgeEmbeddings(
    model_name= model_name,
    model_kwargs= model_kwargs,
    encode_kwargs= encode_kwargs     
)
print("Embedding Model loaded........")

url= "http://localhost:6333"
collection_name= "gpt_db"

qdrant= Qdrant.from_documents(
    documents,
    embeddings,
    url= url,
    prefer_grpc= False,
    collection_name= collection_name
)
print("Qdrant Index Created.........")