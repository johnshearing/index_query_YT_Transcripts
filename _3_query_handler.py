# Enter your query near the bottom of the script. I entered "Positive expectations" but it could be any question or natural language query.
# Then just run the script at the bash terminal with "python3 _query_handler01.py"
# The script will search the index created from video transcripts and return the most relevant sentences along with metadata about the videos.


import psycopg2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname="ah_db",
    user="js",
    password="js",
    host="localhost"
)
cur = conn.cursor()

# Step 2: Load the pre-created FAISS indexes
hicks_index = faiss.read_index("hicks_index.faiss")
all_index = faiss.read_index("all_index.faiss")

# Step 3: Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Function to fetch all sentences from the database (needed for mapping indices back to sentences)
def fetch_all_sentences():
    cur.execute("""
        SELECT id, transcription_id, start_time, text, speaker
        FROM segments
        ORDER BY transcription_id, start_time
    """)
    segments = cur.fetchall()
    return [{'id': s[0], 'transcription_id': s[1], 'start_time': s[2], 'text': s[3], 'speaker': s[4]} for s in segments]

# Step 5: Fetch all sentences and separate them
all_segments = fetch_all_sentences()
hicks_sentences = [s for s in all_segments if s['speaker'] == 'Hicks']
all_sentences = [s for s in all_segments]

# Step 6: Function to handle user queries
def handle_query(query, k=3, max_transcripts=5):
    """
    Retrieve the top k relevant Hicks sentences from up to max_transcripts transcripts,
    ranked by relevance to the query.
    Args:
        query (str): The user's question or input.
        k (int): Number of sentences to retrieve per transcript (default: 5).
        max_transcripts (int): Maximum number of transcripts to consider (default: 5).
    """
    # Generate embedding for the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Step 1: Find the closest sentences to the query across all transcripts
    distances, indices = all_index.search(query_embedding, k * max_transcripts)  # Search for more candidates
    
    # Group results by transcription_id
    transcript_relevance = {}  # Dictionary to store relevance score for each transcript
    for dist, idx in zip(distances[0], indices[0]):
        sentence = all_sentences[idx]
        transcription_id = sentence['transcription_id']
        if transcription_id not in transcript_relevance:
            transcript_relevance[transcription_id] = []
        transcript_relevance[transcription_id].append((dist, idx))  # Store distance and sentence index
    
    # Sort transcripts by their best (lowest) distance (most relevant first)
    sorted_transcripts = sorted(transcript_relevance.items(), key=lambda x: min(d for d, _ in x[1]))
    
    # Step 2: Collect top sentences from the most relevant transcripts
    top_sentences = []
    transcripts_processed = 0
    
    for transcription_id, relevance_list in sorted_transcripts:
        if transcripts_processed >= max_transcripts:
            break
            
        # Get all Hicks sentences from this transcript
        transcript_hicks_sentences = [s for s in hicks_sentences if s['transcription_id'] == transcription_id]
        if not transcript_hicks_sentences:
            continue
            
        # Embed these Hicks sentences
        transcript_hicks_texts = [s['text'] for s in transcript_hicks_sentences]
        transcript_hicks_embeddings = model.encode(transcript_hicks_texts, convert_to_numpy=True)
        
        # Create a temporary FAISS index for this transcript
        embedding_dim = transcript_hicks_embeddings.shape[1]
        transcript_index = faiss.IndexFlatL2(embedding_dim)
        transcript_index.add(transcript_hicks_embeddings)
        
        # Search for the top k most relevant sentences in this transcript
        _, transcript_indices = transcript_index.search(query_embedding, k)
        
        # Add the top sentences to our results
        for i in transcript_indices[0]:
            if len(top_sentences) < k * max_transcripts:  # Limit total sentences
                top_sentences.append(transcript_hicks_sentences[i])
        
        transcripts_processed += 1
    
    # Step 3: Fetch metadata for each transcription in the results
    results_with_metadata = []
    seen_transcriptions = set()
    
    for sentence in top_sentences:
        if sentence['transcription_id'] not in seen_transcriptions:
            cur.execute("SELECT metadata FROM transcriptions WHERE id = %s", (sentence['transcription_id'],))
            metadata = cur.fetchone()
            if metadata:
                metadata = metadata[0]
            else:
                metadata = {}
            seen_transcriptions.add(sentence['transcription_id'])
        else:
            metadata = {}  # Reuse previous metadata if already fetched
            
        results_with_metadata.append({
            'text': sentence['text'],
            'start_time': sentence['start_time'],
            'transcription_id': sentence['transcription_id'],
            'video_title': metadata.get('videoTitle', 'N/A'),
            'url': metadata.get('url', 'N/A')
        })
    
    # Step 4: Print the results, sorted by relevance (earliest transcripts first)
    print(f"\nTesting query: '{query}'")
    for i, result in enumerate(results_with_metadata[:k * max_transcripts]):  # Limit to k sentences total
        print(f"Result {i+1} (Transcript ID: {result['transcription_id']}, Start Time: {result['start_time']}):")
        print(f"Text: {result['text']}")
        print(f"Video Title: {result['video_title']}")
        print(f"URL: {result['url']}")
        print()

# Step 7: Test the system with a sample query
if __name__ == "__main__":
    sample_query = "Positive expectations"
    handle_query(sample_query)

    # Step 8: Close the database connection
    cur.close()
    conn.close()
    print("Database connection closed.")