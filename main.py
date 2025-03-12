from pinecone.grpc import PineconeGRPC, GRPCClientConfig
import time
from pinecone import Pinecone, PodSpec, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
def main():
    model_name = 'distilbert-base-nli-stsb-mean-tokens'
    model = SentenceTransformer(model_name)


    # Initialize a client.
    # API key is required, but the value does not matter.
    # Host and port of the Pinecone Local instance
    # is required when starting without indexes.
    api_key = os.getenv('PINECONE_KEY')
    pc = Pinecone(api_key=api_key)
    indexes = pc.list_indexes()
    print(f"index {indexes}")

    # Create an index
    index_name = "example-index"

    if not pc.has_index(index_name):
        pc.create_index(
            # The name of the index. This is a user-defined name that can be used to refer to the index later when performing operations on it.
            name=index_name,
            # The dimensionality of the vectors that will be stored in the index.
            # This should match the dimensionality of the vectors that will be inserted into the index.
            # We have specified 768 here because that is the embedding dimension returned by the SentenceTransformer model.
            dimension=768,
            # The distance metric used to calculate the similarity between vectors.
            # In this case, euclidean is used, which means that the Euclidean distance will be used as the similarity metric.
            metric="euclidean",
            # A PodSpec object that specifies the environment in which the index will be created.
            # In this example, the index is created in a GCP (Google Cloud Platform) environment named gcp-starter.
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        print(f"Waiting for index {index_name} to be ready...")
        time.sleep(1)
    print(f"Index {index_name} is ready")

    indexes = pc.list_indexes()
    print(f"index {indexes}")
    # Now that we have created an index, we can push vector embeddings.
    sentences_to_index = [
        {"id": "vector1",  "text": "I love using vector databases"},
        {"id": "vector2",  "text": "Vector databases are great for storing and retrieving vectors"},
        {"id": "vector3",  "text": "Using vector databases makes my life easier"},
        {"id": "vector4",  "text": "Vector databases are efficient for storing vectors"},
        {"id": "vector5",  "text": "I enjoy working with vector databases"},
        {"id": "vector6",  "text": "Vector databases are useful for many applications"},
        {"id": "vector7",  "text": "I find vector databases very helpful"},
        {"id": "vector8",  "text": "Vector databases can handle large amounts of data"},
        {"id": "vector9",  "text": "I think vector databases are the future of data storage"},
        {"id": "vector10", "text": "Using vector databases has improved my workflow"}
    ]

    vector_data = []
    # Create embeddings for each sentence
    for sentence in sentences_to_index:
        # Create the embedding
        embedding = model.encode(sentence["text"])
        print(f"embedding:{embedding}")
        # Store the embedding in a dictionary with the sentence ID and metadata
        vector_info = {"id": sentence["id"], "values": embedding.tolist(), "metadata": {"sentence": sentence["text"]}}
        print(f"vector_info:{vector_info}")
        # Add the dictionary to the list of vectors
        vector_data.append(vector_info)
    print(f"vector_data:{vector_data}")

    # Get index
    index = pc.Index(index_name)
    # Save the embeddings (with metadata) in the index
    upsert_result =index.upsert(vectors=vector_data, namespace="example-namespace")
    print(f"upsert_result:{upsert_result}")

    stats = index.describe_index_stats()
    print(f"stats:{stats}")

    # Now that we have stored the vectors in the above index, let's run a similarity search to see the obtained results
    # We can do this using the query() method of the index object we created earlier.
    # First, we define a search text and generate its embedding:
    search_text = "Vector database are totally not useless"

    search_embedding = model.encode(search_text).tolist()

    # Note that include_values=True, include_metadata=True make the query more expensive and here is used only to get the values and metadata
    query_result = index.query(vector=search_embedding, top_k=3, namespace="example-namespace", include_values=True, include_metadata=True)
    print(f"query_result:{query_result}")
    # The output is something like the following:
    # query_result:{'matches': [
    #              {'id': 'vector1', 'score': 146.098938, 'values': []},
    #              {'id': 'vector7', 'score': 149.657135, 'values': []},
    #              {'id': 'vector4', 'score': 152.011017, 'values': []}],
    #               'namespace': '',
    #               'usage': {'read_units': 5}}
    
    # Print only metadata of each resulting vector
    print("\nMetadata of matching vectors:")
    for i, match in enumerate(query_result['matches']):
        print(f"{i+1}. ID: {match['id']}, Score: {match['score']}")
        if 'metadata' in match:
            print(f"   Metadata: {match['metadata']}")
            if 'sentence' in match['metadata']:
                print(f"   Text: \"{match['metadata']['sentence']}\"")
        else:
            print("   No metadata available")
    print()
    
    # * matches: A list of dictionaries, where each dictionary contains information about a matching vector.
    #            Each dictionary includes:
    #            * the id of the matching vector,
    #            * the score indicating the similarity between
    #            * the query vector and the matching vector.
    #            As we specified euclidean as our metric while creating this index, a higher score indicates more distance or similarity.
    # * namespace: The namespace of the index where the query was performed.
    # * usage: A dictionary containing information about the usage of resources during the query operation.
    #          In this case, read_units indicates the number of read units consumed by the query operation, which is 5.
    #          However, we originally appended 10 vectors to this index, which shows that it did look through all the vectors to find the nearest neighbors.


    # Delete the index (cleanup index created for this example)
    pc.delete_index(index_name)

if __name__ == "__main__":
    main()