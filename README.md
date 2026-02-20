# Disaster-Detection-VLM

## Misc. notes + diagrams

### Data pipeline DAG flow 
```mermaid
flowchart LR
    PARAM["DAG Run Param: folder_path <br/><i>(path to folder with dataset images + JSON metadata)</i>"]

    T1["Task: load_source_data <i>(discovers images + JSON metadata from folder)</i>"]
    T2["Task: segment_structures <i>(creates segmented images for each structure in each image)</i>"]
    T3["Task: assess_structure <i>(uses VLM to assess building subtype, ex: destroyed, minor-damage, etc))</i>"]

    OBJ(["Blob Object Storage for images - S3 or compatible solution")]
    DB[("MongoDB database collections: <br/><br/>source_records <i>(stores metadata from source JSON)</i><br/>structures <i>(stores building features + original assessment result from source JSON)</i><br/>assessments <i>(stores VLM disaster assessment)</i>")]

    PARAM --> T1
    T1 -->|"XCom: source jobs"| T2
    T2 -->|"XCom: structure IDs"| T3

    T1 -->|"writes source_records"| DB
    T2 -->|"uploads images"| OBJ
    T2 -->|"writes structures"| DB
    T3 -->|"reads image"| OBJ
    T3 -->|"reads metadata"| DB
    T3 -->|"writes assessments"| DB
```