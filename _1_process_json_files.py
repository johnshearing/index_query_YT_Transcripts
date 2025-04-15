# This takes a directory full of YouTube transcripts and metadata in json format and loads them into a postgreSQL database.
# See this repository to see how the json files are created: https://github.com/johnshearing/scrape_yt_mk_transcripts
# The postgreSQL database is a preliminary step toward creating a FAISS index that an LLM can read and answer questions about the videos.


import json
import psycopg2
from glob import glob
from urllib.parse import urlparse, parse_qs

# Function to extract video_id from YouTube URL
def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.netloc == 'www.youtube.com' and parsed_url.path == '/watch':
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]
    return None

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="ah_db",
    user="js",
    password="js",
    host="localhost"
)
cur = conn.cursor()

# Process each JSON file
for json_file in glob("../yascrape/temp/*.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
        # Extract video_id from metadata['url']
        if "metadata" in data and "url" in data["metadata"]:
            video_id = get_video_id(data["metadata"]["url"])
            if video_id is None:
                print(f"Could not extract video_id from URL in file: {json_file}")
                continue
        else:
            print(f"Missing metadata or url in file: {json_file}")
            continue
        
        # Insert into transcriptions table
        cur.execute(
            """
            INSERT INTO transcriptions (video_id, language, metadata, full_text)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            (video_id, data["language"], json.dumps(data["metadata"]), data["text"])
        )
        transcription_id = cur.fetchone()[0]
        
        # Insert each segment into segments table
        for segment in data["segments"]:
            cur.execute(
                """
                INSERT INTO segments (transcription_id, start_time, end_time, text, speaker)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (transcription_id, segment["start"], segment["end"], segment["text"], segment["speaker"])
            )

# Commit changes and close
conn.commit()
cur.close()
conn.close()