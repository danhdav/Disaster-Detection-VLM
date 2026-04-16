import chromadb
import json
import re
import boto3

# ==========================================
# 1. AWS S3 CONFIGURATION
# ==========================================
s3_client = boto3.client('s3')

# CHANGE THESE TO MATCH YOUR AWS SETUP!
BUCKET_NAME = 'disaster-detection-group12' 
# If your files are in a folder like 'train/labels/', put that here. 
# If they are just sitting in the main bucket, leave it as ''
PREFIX = 'train/labels/' 

# ==========================================
# 2. CHROMADB SETUP
# ==========================================
client = chromadb.PersistentClient(path="./xview2_vector_db")

# Reset the database so we can build it fresh from the cloud
try:
    client.delete_collection(name="disaster_assessments")
except:
    pass

collection = client.create_collection(name="disaster_assessments")

# ==========================================
# 3. S3 PAGINATION (Finding all the files)
# ==========================================
print(f"Scanning S3 Bucket: '{BUCKET_NAME}' for post-disaster JSONs...")

paginator = s3_client.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX)

json_keys = []
for page in pages:
    if 'Contents' in page:
        for obj in page['Contents']:
            key = obj['Key']
            # We only care about post-disaster labels
            if key.endswith('_post_disaster.json'):
                json_keys.append(key)

print(f"Found {len(json_keys)} post-disaster files in S3. Beginning extraction...\n")

# ==========================================
# 4. CLOUD EXTRACTION & EMBEDDING
# ==========================================
documents = []
metadatas = []
ids = []

for idx, key in enumerate(json_keys):
    filename = key.split('/')[-1] # Drops the folder path, keeps just the filename
    
    if idx % 100 == 0:
        print(f"Processing file {idx} of {len(json_keys)}...")
    
    # Read the file directly from S3 memory!
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
    file_content = response['Body'].read().decode('utf-8')
    data = json.loads(file_content)
    
    # 1. Extract Damage Counts
    damage_counts = {
        "destroyed": 0,
        "major-damage": 0,
        "minor-damage": 0,
        "no-damage": 0,
        "un-classified": 0
    }
    
    buildings = data['features']['xy']
    for building in buildings:
        subtype = building['properties'].get('subtype', 'un-classified')
        if subtype in damage_counts:
            damage_counts[subtype] += 1

    # 2. Extract Center Coordinates (Averaging the first building's roof)
    center_lat, center_lon = 0.0, 0.0
    try:
        wkt_string = data['features']['lng_lat'][0]['wkt']
        coords = re.findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)', wkt_string)
        lons = [float(c[0]) for c in coords]
        lats = [float(c[1]) for c in coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)
    except (KeyError, IndexError):
        pass # Skip if the image has zero buildings
        
    # 3. Format the English Summary for Llama 3.1
    rag_text_chunk = (
        f"Filename: {filename}\n"
        f"Location: Latitude {center_lat:.4f}, Longitude {center_lon:.4f}\n"
        f"Building Damage Assessment:\n"
        f"- {damage_counts['destroyed']} destroyed buildings\n"
        f"- {damage_counts['major-damage']} buildings with major damage\n"
        f"- {damage_counts['minor-damage']} buildings with minor damage\n"
        f"- {damage_counts['no-damage']} buildings with no damage\n"
    )
    
    documents.append(rag_text_chunk)
    metadatas.append({"filename": filename, "lat": center_lat, "lon": center_lon})
    ids.append(filename)

# 5. Push to ChromaDB
print("\nPushing data to local ChromaDB Vector Store...")
# ChromaDB handles chunking automatically under the hood, but adding in batches is safer for memory
batch_size = 500
for i in range(0, len(documents), batch_size):
    collection.add(
        documents=documents[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size],
        ids=ids[i:i+batch_size]
    )

print("✅ Cloud Ingestion Complete! Your Vector DB is locked and loaded.")