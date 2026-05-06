import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1. Load multiple docs
def load_documents():
    docs = []

    for file in os.listdir("data"):
        path = os.path.join("data", file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue

        loaded_docs = loader.load()

        # Add metadata
        for d in loaded_docs:
            d.metadata["source"] = file

        docs.extend(loaded_docs)

    return docs


documents = load_documents()

# 2. Smart chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

# 3. Embeddings + Vector DB
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# 4. Retriever (improved)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 5. LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 6. Better Prompt
# 🔹 Prompt 1: Rewrite question
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite follow-up question as standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# 🔹 Prompt 2: Answer generation
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer ONLY from context.
If not found, say "I don't know".

Context:
{context}
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# 6.5. Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

# 7. Document chain (NEW WAY)
document_chain = create_stuff_documents_chain(llm, qa_prompt)

# 8. RAG chain
chain = create_retrieval_chain(history_aware_retriever, document_chain)

# 9. Test
# if __name__ == "__main__":
#     res = chain.invoke({"input": "What is mentioned in documents?"})
#     print(res["answer"])