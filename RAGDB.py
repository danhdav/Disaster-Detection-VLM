import chromadb

# 1. Initialize a Persistent Chroma Client
# This creates a folder named 'xview2_vector_db' in your current directory 
# to save your embeddings so you don't lose them when the script ends.
client = chromadb.PersistentClient(path="./xview2_vector_db")

# 2. Create or load a collection
# Collections are like tables in a traditional database.
collection = client.get_or_create_collection(name="disaster_assessments")

# 3. Prepare your data (Using the output from your previous parsing script)
# For this example, let's assume `parsed_docs` contains the dictionaries we made earlier.
documents = []
metadatas = []
ids = []

for doc in parsed_docs: # parsed_docs from the previous step
    documents.append(doc["text"])
    ids.append(doc["id"])
    
    # We extract the metadata from the text or the raw JSON to enable filtering
    # (Assuming we extracted 'disaster_type' in the previous step)
    disaster_type = "hurricane" if "hurricane" in doc["text"].lower() else "wildfire" # Simplified for example
    
    metadatas.append({
        "disaster_type": disaster_type,
        "source": "xview2_train"
    })

# 4. Add data to Chroma
# Chroma will automatically download a small embedding model the first time 
# and convert your 'documents' into vectors in the background.
print("Embedding and saving to local database...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
print("Data saved successfully!\n")

# 5. Test your RAG Retrieval with a Query and a Metadata Filter
query_text = "Show me areas with total devastation and destroyed buildings."

results = collection.query(
    query_texts=[query_text],
    n_results=2, # Return the top 2 matches
    where={"disaster_type": "hurricane"} # ONLY search within hurricane data!
)

print("--- SEARCH RESULTS ---")
for i in range(len(results['documents'][0])):
    print(f"Match {i+1} (ID: {results['ids'][0][i]}):")
    print(results['documents'][0][i])
    print("-" * 20)