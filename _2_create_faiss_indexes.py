# This script reads the postgreSQL database, chunks the data by sentences, and embeds the data into a FAISS index.
# An LLM will use the index to return relevant sentences from the videos based on a query.


import json
import psycopg2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os

# Suppress warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Step 1: Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname="ah_db",
    user="js",
    password="js",
    host="localhost"
)
cur = conn.cursor()

# Step 2: Function to merge segments into sentences
def merge_segments_into_sentences(segments):
    sentences = []
    current_sentence = []  # List of text parts
    current_segment_ids = []  # List of corresponding segment IDs
    current_speaker = None
    
    for seg in segments:
        if seg['speaker'] != current_speaker and current_sentence:
            sentences.append({
                'text': ' '.join(current_sentence),
                'speaker': current_speaker,
                'transcription_id': seg['transcription_id'],
                'start_time': seg['start_time'],
                'segment_ids': current_segment_ids
            })
            current_sentence = []
            current_segment_ids = []
        
        current_speaker = seg['speaker']
        text_sentences = re.split(r'(?<=[.!?]) +', seg['text'])
        
        for sentence in text_sentences:
            if sentence:
                current_sentence.append(sentence)
                current_segment_ids.append(seg['id'])  # Track the segment ID
                if sentence[-1] in '.!?':
                    sentences.append({
                        'text': ' '.join(current_sentence),
                        'speaker': current_speaker,
                        'transcription_id': seg['transcription_id'],
                        'start_time': seg['start_time'],
                        'segment_ids': current_segment_ids
                    })
                    current_sentence = []
                    current_segment_ids = []
    
    if current_sentence:
        sentences.append({
            'text': ' '.join(current_sentence),
            'speaker': current_speaker,
            'transcription_id': seg['transcription_id'],
            'start_time': seg['start_time'],
            'segment_ids': current_segment_ids
        })
    
    return sentences

# Step 3: Fetch all segments from the database, ordered by transcription_id and start_time
cur.execute("""
    SELECT id, transcription_id, start_time, text, speaker
    FROM segments
    ORDER BY transcription_id, start_time
""")
segments = cur.fetchall()
segment_data = [{'id': s[0], 'transcription_id': s[1], 'start_time': s[2], 'text': s[3], 'speaker': s[4]} for s in segments]

# Step 4: Merge segments into sentences
sentences = merge_segments_into_sentences(segment_data)

# Step 5: Separate sentences by speaker: Hicks (answers) and Non-Hicks (questions)
hicks_sentences = [s for s in sentences if s['speaker'] == 'Hicks']
all_sentences = [s for s in sentences]

# Step 6: Generate embeddings using SBERT (all-MiniLM-L6-v2)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed Hicks sentences (answers)
hicks_texts = [s['text'] for s in hicks_sentences]
hicks_embeddings = model.encode(hicks_texts, convert_to_numpy=True)

# Embed all sentences (questions)
all_texts = [s['text'] for s in all_sentences]
all_embeddings = model.encode(all_texts, convert_to_numpy=True)

print(f"Generated embeddings for {len(hicks_sentences)} Hicks sentences and {len(all_sentences)} Non-Hicks sentences.")

# Step 7: Create separate FAISS indexes
embedding_dim = hicks_embeddings.shape[1]  # Dimension of the embeddings

# Hicks Index (for answers)
hicks_index = faiss.IndexFlatL2(embedding_dim)
hicks_index = faiss.IndexIDMap(hicks_index)
hicks_index.add_with_ids(hicks_embeddings, np.array([i for i in range(len(hicks_sentences))]))  # Use sentence index as ID

# Non-Hicks Index (for audience questions)
all_index = faiss.IndexFlatL2(embedding_dim)
all_index = faiss.IndexIDMap(all_index)
all_index.add_with_ids(all_embeddings, np.array([i for i in range(len(all_sentences))]))  # Use sentence index as ID

# Step 8: Save both FAISS indexes to disk
faiss.write_index(hicks_index, "hicks_index.faiss")
faiss.write_index(all_index, "all_index.faiss")
print("FAISS indexes created and saved.")

# Step 9: Close the database connection
cur.close()
conn.close()
print("Database connection closed.")