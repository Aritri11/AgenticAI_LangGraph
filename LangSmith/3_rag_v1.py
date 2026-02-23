import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
#setting up project name inside the code (method 2)
os.environ['LANGCHAIN_PROJECT'] = 'RAG Chatbot'

PDF_PATH = "Envinromental Studies_ebook.pdf"  # <-- change to your PDF filename

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index
emb =OllamaEmbeddings(
    model='qwen3-embedding:4b'
)
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a tutor. Use only the provided context to answer. If the answer is not explicitly in the context, say: \"I don't know.\""),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer clearly in one or two sentences.")
])


# 5) Chain
llm = ChatOllama(
    model='llama3.2:1b',
    temperature=0
)
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)
