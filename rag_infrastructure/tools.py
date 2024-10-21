from typing import BinaryIO 
from langchain_community.llms import Ollama  
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA

main_llm = Ollama(model="llama3"  )

 
embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2048, chunk_overlap=120, length_function=len, is_separator_regex=False
                                               )

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)


db_path= 'vector_database'

def save_vectors(pdf_file_path:str):
    
    loader = PDFPlumberLoader(pdf_file_path)
    docs = loader.load_and_split() 
    chunks = text_splitter.split_documents(docs) 
    vector_store = Chroma.from_documents(
        documents= chunks, embedding= embedding, persist_directory= db_path
    )

    vector_store.persist()
    
    response = {
        "status": "Successfully vectorized", 
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    
    return response



def get_answer(query):
    
    vector_store = Chroma(persist_directory=db_path, embedding_function=embedding)
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )
    
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
    )
    
    
    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Answer the question and provide additional helpful information,
    based on the pieces of information, if applicable. Be succinct.

    Responses should be properly formatted to be easily read.
    All the direct answer and additional information must be in only a single paragraph. 
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=main_llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": False},
      )
    
    answer= qa.invoke(query)
    
    return {'answer':answer['result'], 'question':query}
    
    # document_chain = create_stuff_documents_chain(main_llm, raw_prompt)
    
    # chain = create_retrieval_chain(retriever, document_chain)

    # result = chain.invoke({"input": query})
    
    # return result["answer"]

def get_response(request:str):
    return main_llm.invoke(request)
