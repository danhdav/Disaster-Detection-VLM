import chromadb
import ollama
import re
import boto3
import webbrowser # NEW: Replaces 'os' for opening web links
from botocore.config import Config

# ==========================================
# 1. SETUP & CLOUD CONFIGURATION
# ==========================================
client = chromadb.PersistentClient(path="./xview2_vector_db")
collection = client.get_collection(name="disaster_assessments")

# AWS S3 Setup
s3_client = boto3.client(
    's3',
    region_name='us-east-2', # <--- Make sure this matches your bucket's region!
    config=Config(signature_version='s3v4')
)

BUCKET_NAME = 'disaster-detection-group12' 
IMAGES_PREFIX = 'train/images/' 

# The chatbot's short-term memory
last_queried_filename = None
last_found_files = []  

# Helper function to generate and open S3 links
def open_s3_image(filename):
    png_filename = filename.replace('.json', '.png')
    object_key = f"{IMAGES_PREFIX}{png_filename}"
    
    try:
        # Generate a link valid for 5 minutes (300 seconds)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': object_key},
            ExpiresIn=300 
        )
        webbrowser.open(presigned_url)
        return True
    except Exception as e:
        print(f"   [Error generating link: {e}]")
        return False

print("\n" + "="*50)
print("xView2 Cloud Disaster Response Terminal [ONLINE]")
print("Type 'exit' or 'quit' to end the session.")
print("="*50 + "\n")

while True:
    user_query = input("\n👤 You: ")
    
    if user_query.strip().lower() in ['exit', 'quit']:
        print("Shutting down the chatbot...")
        break
        
    if not user_query.strip():
        continue

    user_lower = user_query.lower()

    # 1. Regex Matchers & Memory
    match_exact = re.search(r'([a-zA-Z0-9_-]+\.json)', user_query)
    coord_match = re.search(r'lat.*?(-?\d+\.\d+).*?lon.*?(-?\d+\.\d+)', user_lower)
    
    is_plural_pronoun = any(word in user_lower for word in ["these", "those", "them", "all", "they"])
    is_singular_pronoun = any(word in user_lower for word in ["this", "it", "that", "the file", "the image"])
    
    target_filename = None
    target_group = []
    
    if match_exact:
        target_filename = match_exact.group(1)
        last_queried_filename = target_filename 
    elif is_plural_pronoun and last_found_files:
        target_group = last_found_files
    elif is_singular_pronoun and last_queried_filename:
        target_filename = last_queried_filename

    # 2. THE CLOUD IMAGE OPENER ROUTER
    open_keywords = ["open", "show", "display"]
    is_open_command = any(word in user_lower for word in open_keywords)
    is_plural_open = any(word in user_lower for word in ["images", "pictures", "surrounding"]) or is_plural_pronoun
    
    if is_open_command and not "around" in user_lower:
        if is_plural_open and last_found_files:
            print(f"   [System: Requesting secure AWS links for {len(last_found_files)} images...]")
            opened_count = 0
            for f_name in last_found_files:
                if open_s3_image(f_name):
                    opened_count += 1
            print(f"\n🤖 Chatbot: I have opened {opened_count} images in your browser!")
            continue
        elif target_filename:
            print(f"   [System: Requesting secure AWS link for {target_filename.replace('.json', '.png')}...]")
            if open_s3_image(target_filename):
                print("\n🤖 Chatbot: Image opened in your web browser!")
            else:
                print("\n🤖 Chatbot: Something went wrong fetching the image from AWS.")
            continue
        else:
            print("\n🤖 Chatbot: I don't know which file(s) you mean.")
            continue

    # 3. THE SPATIAL "AROUND" ROUTER
    is_around_query = "around" in user_lower
    
    if is_around_query and target_filename:
        print(f"   [System: 'Around' command detected. Finding coordinates for {target_filename}...]")
        target_result = collection.query(query_texts=["dummy"], n_results=1, where={"filename": target_filename})
        
        if not target_result['metadatas'][0]:
            print("\n🤖 Chatbot: Couldn't find target file.")
            continue
            
        target_meta = target_result['metadatas'][0][0]
        center_lat, center_lon = target_meta.get('lat', 0.0), target_meta.get('lon', 0.0)
        radius = 0.02
        
        print(f"   [System: Searching radius +/- {radius} degrees from Lat {center_lat:.4f}, Lon {center_lon:.4f}]")
        results = collection.query(
            query_texts=[user_query], n_results=5, 
            where={"$and": [{"lat": {"$gte": center_lat - radius}}, {"lat": {"$lte": center_lat + radius}},
                            {"lon": {"$gte": center_lon - radius}}, {"lon": {"$lte": center_lon + radius}}]}
        )
        
        if results['metadatas'][0]:
            found_files = [meta['filename'] for meta in results['metadatas'][0]]
            last_found_files = found_files 
            print(f"   [System: Found {len(found_files)} files: {', '.join(found_files)}]")
            
            if is_open_command:
                print(f"   [System: Requesting secure AWS links for {len(last_found_files)} images...]")
                for f_name in last_found_files:
                    open_s3_image(f_name)
                print("\n🤖 Chatbot: Opened surrounding images in your browser!")
                continue 
            
            user_query = f"{user_query}\n\n[System Directive: Base your answer on ALL of these retrieved files: {', '.join(found_files)}. If the user asks for coordinates, provide the Latitude and Longitude for all of them.]"

    # 4. COORDINATE SEARCH ROUTER 
    elif coord_match:
        search_lat = float(coord_match.group(1))
        search_lon = float(coord_match.group(2))
        print(f"   [System: Raw coordinates detected. Searching near Lat {search_lat}, Lon {search_lon}...]")
        
        radius = 0.05 
        results = collection.query(
            query_texts=[user_query], n_results=3, 
            where={"$and": [{"lat": {"$gte": search_lat - radius}}, {"lat": {"$lte": search_lat + radius}},
                            {"lon": {"$gte": search_lon - radius}}, {"lon": {"$lte": search_lon + radius}}]}
        )
        
        if results['metadatas'] and results['metadatas'][0]:
            found_files = [meta['filename'] for meta in results['metadatas'][0]]
            last_found_files = found_files 
            
            user_query = f"{user_query}\n\n[System Directive: OVERRIDE STRICT MATCHING. The user is searching for a location. You successfully used a spatial bounding box to find the closest nearby files. Do NOT say you lack data. Instead, enthusiastically tell the user that while you don't have that exact pinpoint, you found these closest matches in the radius: {', '.join(found_files)}. List them and their exact coordinates.]"

    # 5. GROUP MATCH ROUTER (PLURAL MEMORY)
    elif target_group:
        print(f"   [System: Plural memory detected. Applying filter for {len(target_group)} recent files...]")
        results = collection.query(
            query_texts=[user_query], 
            n_results=len(target_group), 
            where={"filename": {"$in": target_group}} 
        )
        user_query = f"{user_query}\n\n[System Directive: OVERRIDE STRICT MATCHING. The user is asking about a specific group of files you recently retrieved. Answer the user's question using strictly the context provided for these exact files: {', '.join(target_group)}.]"

    # 6. EXACT MATCH ROUTER (SINGULAR MEMORY)
    elif target_filename:
        print(f"   [System: Target file identified. Applying strict filter for {target_filename}]")
        results = collection.query(query_texts=[user_query], n_results=1, where={"filename": target_filename})
        
    # 7. SEMANTIC SEARCH ROUTER
    else:
        results = collection.query(query_texts=[user_query], n_results=3)

    # 8. LLM GENERATION
    if not results or not results['documents'] or not results['documents'][0]:
        print("\n🤖 Chatbot: I could not find any matching files in the database.")
        continue

    retrieved_context = "\n".join(results['documents'][0])
    
    system_prompt = f"""You are a disaster response AI assistant. 
    Answer the user's question using strictly the provided context from the xView2 satellite dataset. 
    If the exact numbers or answer are not in the context, say "I do not have enough data to answer that." 
    Do not make up information.

    CONTEXT:
    {retrieved_context}
    """

    response = ollama.chat(model='llama3.1', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_query} 
    ])

    print("\n🤖 Chatbot:", response['message']['content'])