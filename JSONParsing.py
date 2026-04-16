import os
import json
from glob import glob

# Using a raw string (r"") is best practice for Windows file paths
labels_dir = r"C:\Users\Proshun Saha\Downloads\New folder (7)\train\labels"

def process_local_xview2_jsons(directory_path, limit=5):
    # Filter strictly for post-disaster JSONs
    search_pattern = os.path.join(directory_path, "*_post_disaster.json")
    json_files = glob(search_pattern)
    
    print(f"Found {len(json_files)} post-disaster JSON files. Processing first {limit}...\n")
    
    documents = []
    
    for file_path in json_files[:limit]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract core metadata
        metadata = data['metadata']
        disaster_type = metadata['disaster_type']
        image_id = metadata['id']
        
        # Initialize damage counters
        damage_counts = {
            "no-damage": 0,
            "minor-damage": 0,
            "major-damage": 0,
            "destroyed": 0,
            "un-classified": 0
        }
        
        # Tally up the damage from each building polygon
        if 'features' in data and 'xy' in data['features']:
            for feature in data['features']['xy']:
                properties = feature.get('properties', {})
                if 'subtype' in properties:
                    damage_level = properties['subtype']
                    if damage_level in damage_counts:
                        damage_counts[damage_level] += 1
        
        # Compile the text chunk for your RAG Embedding
        rag_text_chunk = (
            f"Image ID {image_id} captures a {disaster_type} event. "
            f"Building damage assessment: {damage_counts['destroyed']} destroyed, "
            f"{damage_counts['major-damage']} with major damage, "
            f"{damage_counts['minor-damage']} with minor damage, and "
            f"{damage_counts['no-damage']} with no damage."
        )
        
        # Store both the text and the raw ID for future reference
        documents.append({"id": image_id, "text": rag_text_chunk})
        print(f"Processed: {image_id} | {disaster_type} | Destroyed: {damage_counts['destroyed']}")
        
    return documents

# Run the function locally (set limit=None when you are ready to process all of them)
parsed_docs = process_local_xview2_jsons(labels_dir, limit=5)

# Example of how you will access the text later:
# print(parsed_docs[0]['text'])