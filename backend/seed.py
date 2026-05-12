import json
import os
from pymongo import MongoClient

# Connect to your database using the URL from your .env file
client = MongoClient("mongodb://localhost:27017/")
db = client["disaster_assessment"] # Name of your database
collection = db["images"]          # Name of your collection

# Point this to the folder where all your downloaded JSONs are
folder_path = r"C:\Users\Proshun Saha\Downloads\test images tar\test_images_labels_targets\test\labels" 

# This loop does the clicking for you
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            data = json.load(file)
            collection.insert_one(data)
            print(f"Uploaded: {filename}")

print("Done! All files are in MongoDB.")