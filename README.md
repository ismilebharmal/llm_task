# RAG-based LLM Application for Corey Schafer's Pandas Tutorials

## Overview
This project implements a Retrieval-Augmented Generation (RAG)-based system to answer user questions related to Corey Schafer's Pandas Tutorials YouTube playlist. The system utilizes ChromaDB for vector storage and retrieval and an LLM (e.g., OpenAI's GPT) for generating final answers. This README outlines the process of implementation, key concepts, and tools used.

## Table of Contents
- [Step-by-Step Process](#step-by-step-process)
- [LangGraph Flow](#langgraph-flow)
- [Key Concepts and Tools](#key-concepts-and-tools)
- [Example Scenario](#example-scenario)
- [Conclusion](#conclusion)

## Step-by-Step Process

1. **User Question → Playlist Selection**
   - **Objective**: Identify the relevant playlist based on the user's question.
   - **Description**: Analyze the user's question to detect the topic (e.g., Pandas or Matplotlib) and select the corresponding playlist.
   - **LangGraph Node**: User Question → Playlist Classifier
   - **Tools**: Keyword matching, basic classifier model, or manual mapping.

2. **Playlist Selection → Video ID and Title**
   - **Objective**: Identify the specific video most relevant to the user's question.
   - **Description**: Perform a similarity search on video titles and descriptions to find the best match.
   - **LangGraph Node**: Playlist → Video ID & Title Retrieval
   - **Tools**: Similarity search using embeddings (e.g., SentenceTransformers).

3. **Video ID and Title → Transcript Chunking and Vector Storage**
   - **Objective**: Retrieve, chunk, and store the video transcript for future retrieval.
   - **Description**: Load the transcript, chunk it, embed it, and store it in ChromaDB.
   - **LangGraph Node**: Transcript → Chunking → Vector Storage
   - **Tools**: Python code (e.g., SentenceTransformers) for chunking and embedding.

4. **User Query → Retrieve Matching Transcript and Time**
   - **Objective**: Retrieve the most relevant chunk based on the user's question.
   - **Description**: Embed the user's question, perform a similarity search on the stored chunks, and extract the relevant metadata.
   - **LangGraph Node**: Vector Store → Chunk Retrieval (with Timestamp)
   - **Tools**: ChromaDB for vector storage and retrieval.

5. **LLM Answer Generation with Code and Video Link**
   - **Objective**: Provide a complete answer using vector-retrieved data and LLM-generated output.
   - **Description**: Pass the user's query and retrieved chunk to the LLM to generate a detailed response, including a link to the relevant video portion.
   - **LangGraph Node**: LLM → Final Answer Generation (with Video Link)
   - **Tools**: LLM (e.g., GPT) for generating the response.

## LangGraph Flow
```plaintext
+--------------------------+       +---------------------------+       +--------------------------+
|   User Question Input     |  -->  |   Playlist Classification  |  -->  |   Select Video (ID & Title)|
+--------------------------+       +---------------------------+       +--------------------------+
                                                                              |
                                                                              v
+--------------------------+       +---------------------------+       +--------------------------+
|   Load Transcript.json    |  -->  |     Transcript Chunking    |  -->  |   Embed & Store in ChromaDB|
+--------------------------+       +---------------------------+       +--------------------------+
                                                                              |
                                                                              v
+--------------------------+       +---------------------------+       +--------------------------+
|   Retrieve Transcript     |  -->  |   Retrieve Most Relevant   |  -->  |   LLM: Generate Answer    |
|   (with Timestamps)       |       |   Chunk (with Timestamp)   |       |   Provide Code & Video    |
+--------------------------+       +---------------------------+       +--------------------------+
   