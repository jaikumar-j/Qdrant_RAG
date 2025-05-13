from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient


model_name= "BAAI/bge-large-en"
model_kwargs = {"device":"cpu"}
encode_kwargs= {"normalize_embeddings":False}

embeddings= HuggingFaceBgeEmbeddings(
    model_name= model_name,
    model_kwargs= model_kwargs,
    encode_kwargs= encode_kwargs     
)
url= "http://localhost:6333"
collection_name= "gpt_db"

client= QdrantClient(
    url= url,
    prefer_grpc= False
)
print(client)
print("################")

db= Qdrant(
    client= client,
    embeddings= embeddings,
    collection_name= collection_name
)
print(db)
print("################")

query="what are some of the limitations of GPT-4?"
docs= db.similarity_search_with_score(query= query, k=5)

for i in docs:
    doc, _ = i  # Unpack the tuple and ignore the score
    print(doc.page_content)
