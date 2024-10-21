from importlib.metadata import metadata

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
import chromadb
from langchain_openai import OpenAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models.models import SubmitFinalAnswer,FirstNodeResponse,State
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.document_compressors import  EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from utils.prompts import  system_prompt1
from utils.tool_utils import chunk_transcript,save_chunks_to_json,get_best_page
import os
# Load environment variables
load_dotenv()

import openai
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize OpenAI API and sentence transformer for embeddings
openai.api_key = 'your-openai-api-key'  # Replace with your OpenAI API Key
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load metadata from the JSON file
def load_metadata(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# Load the transcript data
def load_transcript(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# Find the most relevant video based on the question
def find_relevant_video(meta_data, user_question):
    titles = [entry['title'] for entry in meta_data['entries']]
    video_ids = [entry['id'] for entry in meta_data['entries']]

    # Get embeddings for the user query and the video titles
    question_embedding = embedding_model.encode([user_question])[0]
    title_embeddings = embedding_model.encode(titles)

    # Calculate cosine similarity between the query and the titles
    similarities = np.dot(title_embeddings, question_embedding) / (
                np.linalg.norm(title_embeddings, axis=1) * np.linalg.norm(question_embedding))
    best_match_idx = np.argmax(similarities)

    return meta_data['entries'][best_match_idx]
# -------------------------------
# 1. LLM Management Setup
# -------------------------------

def get_llm(model_name: str, temperature: float = 0):
    """Initialize LLM (OpenAI, Anthropic, etc.) based on model_name and temperature."""
    if model_name == 'openai':
        print("get llm function: ",os.environ.get('OPENAI_API_MODEL'))
        return ChatOpenAI(model=os.environ.get('OPENAI_API_MODEL'), temperature=temperature)
    else:
        raise ValueError(f"Unknown model: {model_name}")



# Query Generation process from th LLM
def query_result_node(state: State, query_gen):
    """Generate the SQL query using an LLM prompt."""
    # Iterate through the messages to extract tool message content
    try:
        user_question = str(state['question'])
        message = query_gen.invoke({"qs": user_question,})
        print("FIRST NODE: ",message)
        return {"finalResponse": message['datasource']}
    except Exception as e:
        print("Exception in first    call:",str(e))
        return {"error": str(e)}

# Select relevant id and video from playlist Generation process from th LLM
def select_relevant_video_node(state: State):
    """Selection the best video suits from the user's question and return the id and title of th user's requested video"""
    # Iterate through the messages to extract tool message content
    try:
        # message = query_gen.invoke({"qs": qs,})
        user_question = str(state['question'])
        if state['finalResponse']=='pandas':
            print("INSIDE IF")
            meta_data = load_metadata('transcripts/playlists/PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS/metadata.json')
        else:
            meta_data=load_metadata('transcripts/playlists/PL-osiE80TeTvipOqomVEeZ1HRrcEvtZB_/metadata.json')

        best_video = find_relevant_video(meta_data, user_question)
        video_id = best_video['id']
        return {"best_video": best_video}
    except Exception as e:
        print("Exception in Second    call:",str(e))
        return {"error": str(e)}

# Select relevant id and video from playlist Generation process from th LLM
def building_vector_node(state: State):
    """Selection the best video suits from the user's question and return the id and title of th user's requested video"""
    # Iterate through the messages to extract tool message content
    try:

        transcript_file = f"transcripts/videos/{state['best_video']['id']}/transcript.json"
        output_file = f"chunks{state['best_video']['id']}.json"

        with open(transcript_file, 'r') as file:
            transcript_data = json.load(file)

        # Chunk the transcript
        chunks = chunk_transcript(transcript_data, chunk_size=300, overlap=50)
        save_chunks_to_json(chunks, output_file)

        # Initialize the Chroma client
        client = chromadb.Client()

        # Extract text from each chunk (ensure 'text' is present in each chunk)
        # Extract text from each chunk and create Document objects
        documents = [
            Document(page_content=chunk['text'], metadata={"chunk_id": chunk['chunk_id']}) for idx, chunk in enumerate(chunks)
        ]


        collections = client.list_collections()

        if not os.path.exists('some_data/chroma_db'):
            os.makedirs('some_data/chroma_db')  # Create the directory if it doesn't exist

        # Create the Chroma vector store from documents
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory="some_data/chroma_db",
        )
        vectordb.persist()
        return  state
    except Exception as e:
        print("Exception in vector  call:",str(e))
        return {"error": str(e)}

# Select relevant id and video from playlist Generation process from th LLM
def retrival_output_node(state: State,query_gen):
    """Selection the best video suits from the user's question and return the id and title of th user's requested video"""
    # Iterate through the messages to extract tool message content
    try:
        # message = query_gen.invoke({"qs": qs,})
        user_question = str(state['question'])

        vectordb = Chroma(embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
                          persist_directory='some_data/chroma_db')
        # base_retriver = vectordb.as_retriever()
        # retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        # print("rettivedd",retriever.get_relevant_documents(user_question))
        results = vectordb.similarity_search_with_relevance_scores(user_question, k=1)
        # print("RESULT",results)
        metadata_list = []
        content_list = []

        for doc_tuple in results:
            if isinstance(doc_tuple, tuple) and len(doc_tuple) == 2:
                doc, score = doc_tuple  # Unpack the tuple
                # Check if doc is an instance of Document
                if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                    metadata_list.append(doc.metadata)  # Store metadata
                    content_list.append(doc.page_content)  # Store page content
                else:
                    print("Retrieved object does not have the expected attributes:", doc)
            else:
                print("Retrieved data is not in expected tuple format:", doc_tuple)

        # print(metadata_list)
        # print(content_list)
        # splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10, separator=". ")
        # redundant_filter = EmbeddingsRedundantFilter(embeddings=OpenAIEmbeddings(model="text-embedding-3-small"))
        # relevant_filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        #                                    similarity_threshold=0)
        # pipeline_compressor = DocumentCompressorPipeline(
        #     transformers=[splitter, redundant_filter, relevant_filter])
        # compression_retriever = ContextualCompressionRetriever(
        #     base_compressor=pipeline_compressor, base_retriever=base_retriver)
        # retrive_data = compression_retriever.invoke(user_question)
        # print("retived",retrive_data)
        # Convert retrieved data to just the text (flattening the list if necessary)
        # docs_text = "\n\n".join([doc.page_content for doc in retrive_data])  # Use 'page_content' attribute

        # print(docs_text)  # Check if the text looks correct
        print(metadata_list[0]['chunk_id'])
        # prompt = PromptTemplate(
        #     input_variables=["query", "docs"],
        #     template="""
        #         You are a helpful assistant that that can answer questions about given docs.
        #
        #         Answer the following question: {query}
        #         By searching the following transcript: {docs}
        #
        #         Only use the relevent  information from the transcript to answer the question.
        #         Note that the asnwer should not be hallucinate , it should be excatly from the transcript only.
        #         try to check the relavant info also find from which time the video starts
        #         If you feel like you don't have enough information to answer the question, say "I don't know".
        #         """,
        # )
        # openLLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        # llm = ChatOpenAI(
        #     model="gpt-4o-mini",
        #     temperature=0,
        # )
        with open(f"chunks{state['best_video']['id']}.json", 'r') as json_file:
            data = json.load(json_file)
        for chunk in data:
            if chunk.get("chunk_id") == metadata_list[0]['chunk_id']:
                start_time= chunk.get("start_time")
        start_time_in_seconds = int(start_time)

        # Check if the URL already has parameters
        if '?' in state['best_video']['url']:
            # If it does, append the start time as an additional parameter
            formatted_url = f"{state['best_video']['url']}&t={start_time_in_seconds}"
        else:
            # If it doesn't, append the start time as the first parameter
            formatted_url = f"{state['best_video']['url']}?t={start_time_in_seconds}"
        # prompt_to_llm = prompt.format(query=state['question'], docs=retrive_data)
        # response = llm.invoke(prompt_to_llm)
        return {"response": formatted_url}
    except Exception as e:
        print("Exception in final  call:",str(e))
        return {"error": str(e)}


# -------------------------------
# 4. Main Workflow Setup
# -------------------------------

# Use an in-memory SQLite database
# memory = SqliteSaver.from_conn_string(":memory:")

def build_workflow():
    """Main function to build and connect the workflow nodes."""
    workflow = StateGraph(State)

    query_gen_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt1)]
    )
    select_playlist_gen_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt1)]
    )
    retrival_gen_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt1)]
    )

    query_gen = query_gen_prompt | get_llm('openai').with_structured_output(FirstNodeResponse)
    retrival_gen = retrival_gen_prompt | get_llm('openai').with_structured_output(SubmitFinalAnswer)

    workflow.add_node("query_generation_node",lambda state: query_result_node(state,query_gen))
    workflow.add_node("select_relevant_video", lambda state:select_relevant_video_node(state))
    workflow.add_node("building_vector", lambda state:building_vector_node(state))
    workflow.add_node("retrieve_data", lambda state:retrival_output_node(state,retrival_gen))
    # # Set node edges (flow)
    workflow.set_entry_point("query_generation_node")
    workflow.add_edge("query_generation_node", "select_relevant_video")
    workflow.add_edge("select_relevant_video", "building_vector")
    workflow.add_edge("building_vector", "retrieve_data")
    workflow.add_edge("retrieve_data", END)

    # Compile the graph with the in-memory checkpointing
    return workflow.compile()


workflow_app= build_workflow()
def visualize_workflow(workflow):
    """Compile and generate the workflow graph visualization."""
    workflow.get_graph().draw_mermaid_png(output_file_path="langgraph.png")
visualize_workflow(workflow_app)
