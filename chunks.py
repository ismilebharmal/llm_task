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


if __name__ == "__main__":
    transcript_file = "app/transcripts/videos/HQ6XO9eT-fc/transcript.json"
    output_file = "chunks2.json"

    with open(transcript_file, 'r') as file:
        transcript_data = json.load(file)

    # Chunk the transcript
    chunks = chunk_transcript(transcript_data, chunk_size=300, overlap=50)

    # Save the chunks to a JSON file
    save_chunks_to_json(chunks, output_file)
