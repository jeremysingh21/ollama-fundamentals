from langchain_community.document_loaders import SQLDatabaseLoader as SQLLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
from sqlalchemy import create_engine, text
import pyodbc
import os
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from urllib.parse import quote_plus
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

# Load environment variables
load_dotenv()

# Create connection string using pyodbc format
driver = "ODBC Driver 17 for SQL Server"  # Changed from 18 to 17
server = os.getenv('DB_SERVER')
database = os.getenv('DB_NAME')
username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')

# Create SQLAlchemy engine with pyodbc
connection_string = (
    f"mssql+pyodbc://{username}:{quote_plus(password)}@{server}/{database}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&TrustServerCertificate=yes"
    "&autocommit=True"
    "&timeout=30"
)
engine = create_engine(connection_string)

# Test connection
try:
    with engine.connect() as conn:
        print("Successfully connected to the database!")
except Exception as e:
    print(f"Failed to connect to database: {e}")
    exit(1)

# Connect to database and load data
print("Connecting to database...")
db = SQLDatabase(engine=engine)

# Example query - replace with your specific table/query
query = """
SELECT TOP (200) *  -- Use parentheses for TOP clause
FROM [listings].[listings]  -- Use square brackets for identifiers
WHERE geog_state IN ('NJ', 'NY', 'PA')
"""  # You can modify this query to target specific tables

# Try loading data with SQLDatabaseLoader first
try:
    loader = SQLLoader(
        db=db,
        query=query,
        include_rownum_into_metadata=False,
        metadata_columns=["*"]  # Include all columns as metadata
    )
    data = loader.load()
    print(f"Successfully loaded {len(data)} records using SQLDatabaseLoader")
    # Print first 3 records for verification
    print("\nSample records:")
    for i, record in enumerate(data[:3]):
        print(f"\nRecord {i+1}:")
        print(f"Content: {record.page_content}")
        print(f"Metadata: {record.metadata}")
except Exception as e:
    print(f"Warning: SQLDatabaseLoader failed with error: {e}")
    print("Falling back to direct SQL query...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            # Convert rows to Document objects with proper structure
            data = [
                Document(
                    page_content=str(row),  # Main content
                    metadata=dict(row)      # Preserve structured data as metadata
                )
                for row in result.mappings()  # Get rows as dictionaries
            ]
        print(f"Successfully loaded {len(data)} records using direct query")
        # Print first 3 records for verification
        print("\nSample records:")
        for i, record in enumerate(data[:3]):
            print(f"\nRecord {i+1}:")
            print(f"Content: {record.page_content}")
            print(f"Metadata: {record.metadata}")
    except Exception as e:
        print(f"Error: Both loading methods failed: {e}")
        exit(1)

# Process each SQL record into a document format
print("Loading SQL data...")

# Filter complex metadata before splitting
filtered_chunks = filter_complex_metadata(data)
print(f"Filtered {len(data) - len(filtered_chunks)} complex metadata values")

# Split records into chunks if needed
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
split_chunks = text_splitter.split_documents(filtered_chunks)
print(f"Done splitting records into {len(split_chunks)} chunks...")

# Set up embedding model
model = "deepseek-r1"
ollama.pull("nomic-embed-text")

# Create vector database from SQL records
vector_db = Chroma.from_documents(
    documents=split_chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="sql-rag",
)
print("Done adding to vector database...")
# Print sample vector data
print("\nSample vector data:")
sample_docs = vector_db.similarity_search("sample", k=3)
for i, doc in enumerate(sample_docs):
    print(f"\nVector Document {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")

# Set up retrieval system
llm = ChatOllama(model=model)

# Query prompt for generating multiple perspectives
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant records from
    a SQL database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

# RAG prompt template
template = """Answer the question based ONLY on the following database records:
{context}
Question: {question}

Provide a clear and concise answer based on the database information."""

prompt = ChatPromptTemplate.from_template(template)

# Set up the RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example usage
if __name__ == "__main__":
    # Test query to verify RAG setup
    test_question = "What are the different property types available in the database?"
    test_result = chain.invoke(input=test_question)
    print("\nTest Question:", test_question)
    print("\nTest Answer:", test_result)
    
    # Original example query
    question = "Give me a list of all properties with that is property type of retail in new york. Make it a bulleted list with numbers and show all the details."
    result = chain.invoke(input=question)
    print("\nQuestion:", question)
    print("\nAnswer:", result) 




