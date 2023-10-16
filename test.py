import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def test(query: str, k: int = 4, filter = None, fetch_k: int = 20, **kwargs):
    print('query'+query)
    print('k'+str(k))
    print('kwargs',kwargs) 

embeddingmodel_path='/home/song/development/llm/embeddings/shibing624/text2vec-base-chinese'
embeddings = HuggingFaceEmbeddings(model_name=embeddingmodel_path, model_kwargs={'device': 'cuda'})


top_k=3
score_threshold1=0.5

# Text Test
loader = TextLoader("/home/song/development/llm/testdata/minfadian.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

db = FAISS.from_documents(docs, embeddings)

query = "自然人"

test(query,  k=top_k, score_threshold='0.01')

docs = db.similarity_search_with_score(query,  k=top_k, score_threshold='0.1')
print(docs[0].page_content)

# Pdf Test
pdf='/home/song/development/llm/testdata/GPTCanSolveMathematicalProblemsWithoutaCalculator.pdf'



