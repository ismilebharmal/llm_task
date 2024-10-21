import json


def chunk_transcript(transcript_data, chunk_size=300, overlap=50):
    chunks = []
    current_chunk = []
    current_chunk_word_count = 0
    current_start_time = 0.0

    # Iterate over the words and form chunks
    for word_info in transcript_data['results']['channels'][0]['alternatives'][0]['words']:
        current_chunk.append(word_info['punctuated_word'])
        current_chunk_word_count += 1

        if current_chunk_word_count >= chunk_size:
            end_time = word_info['end']
            chunks.append({
                "chunk_id": f"chunk_{len(chunks) + 1}",
                "text": " ".join(current_chunk),
                "start_time": current_start_time,
                "end_time": end_time
            })
            # Reset the chunk
            current_chunk = current_chunk[-overlap:]  # Keep the overlap words
            current_chunk_word_count = len(current_chunk)
            current_start_time = word_info['start']

    # If there are remaining words, add them as the last chunk
    if current_chunk:
        chunks.append({
            "chunk_id": f"chunk_{len(chunks) + 1}",
            "text": " ".join(current_chunk),
            "start_time": current_start_time,
            "end_time": word_info['end']
        })

    return chunks


def save_chunks_to_json(chunks, output_file):
    with open(output_file, 'w') as outfile:
        json.dump(chunks, outfile, indent=4)



def get_best_page(response):
    max_score = -1
    best_page = None

    for doc in response:
        score = doc.state["query_similarity_score"]
        if score > max_score:
            max_score = score
            best_page = doc

    if best_page:
        highest_page_content = best_page.page_content
        highest_page_metadata = best_page.metadata
        highest_query_similarity_score = best_page.state["query_similarity_score"]
        return highest_page_content
    else:
        return None