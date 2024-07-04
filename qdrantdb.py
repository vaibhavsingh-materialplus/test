from qdrant_client import QdrantClient
#import qdrant_client
from qdrant_client.models import Distance, VectorParams
#qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance, for testing, CI/CD
# OR
#client = QdrantClient(path="path/to/db")  # Persists changes to disk, fast prototyping

client = QdrantClient("http://localhost:6333") # Connect to existing Qdrant instance

collection_name = "SpeakerRecognition"

# Check if collection exists
if client.collection_exists(collection_name):
    pass
    #print(f"Collection '{collection_name}' already exists.")
else:
    # Create the collection
    client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=192, distance=Distance.COSINE),)
    #print(f"Collection '{collection_name}' created.")
#client = QdrantClient(":memory:")
