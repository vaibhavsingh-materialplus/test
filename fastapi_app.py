# fastapi_app.py

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm
import numpy as np
from pydub import AudioSegment
import csv
import os
#import qdrant_client
#docker run -p 6333:6333 qdrant/qdrant:latest
#uvicorn fastapi_app:app --reload
def read_value_from_csv(key):
    filename = "metadata.csv"
    value = None
    
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            if row['key'] == str(key):
                value = row['value']
                break
    
    return value


def write_dict_to_csv(metadata):
    filename = "metadata.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['key', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()  # Write header only if the file is newly created
        
        for key, value in metadata.items():
            writer.writerow({'key': key, 'value': value}) 
    

#client = QdrantClient(
  #  url="https://34db88e0-2ec1-4ae1-832e-acfce9dc4a6d.europe-west3-0.gcp.cloud.qdrant.io",
   # api_key="NpscFSq5HO6eiWqOWY6ZQxsRv64Lcwtbr6WkzLni66ovUtVZj-xoBg",)
#client = QdrantClient(":memory:")
client = QdrantClient("http://localhost:6333") # Connect to existing Qdrant instance
#client = QdrantClient(":memory:")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
#,run_opts={"device":"cuda"}

app = FastAPI()


class InputData(BaseModel):
    folder: list
    text: str

class InputAudio(BaseModel):
   text:str



def convert_audio(input_file, output_file, bitrate=256000, channels=1, sample_rate=16000):
  # Load the audio file
  audio = AudioSegment.from_file(input_file)

  # Set parameters
  audio = audio.set_frame_rate(sample_rate)
  audio = audio.set_channels(channels)

  # Set bit rate
  audio.export(output_file, format="wav", bitrate=f"{bitrate}bps")

def search(path:str):
    input_file = f"{path}"
    output_file = 'final_converted.wav'
    convert_audio(input_file, output_file, bitrate=256000, channels=1, sample_rate=16000)
    tmp, fs = torchaudio.load(output_file)
    embedding= classifier.encode_batch(tmp)
    embedding = embedding.cpu()  
    # Specify collection name, limit (number of results), and optional filters
    results = client.search(collection_name="SpeakerRecognition",query_vector=embedding[0][0].numpy(),limit=3)
    #return results
    id_score_list = []
    for point in results:
        #print(point.payload)
        key_to_find = point.id
        found_value = read_value_from_csv(key_to_find)
    
        if found_value!="":
          id_score_list.append({'Speaker':found_value,'id':point.id,'score':point.score})
        else:
          id_score_list.append({'Speaker':"None",'id':point.id,'score':point.score})
         

        #id_score_list.append({'id': point.id, 'score': point.score})

    

    return id_score_list

speaker=dict()


#Cluster_voxtrain1
def process_data(folder,text):
  a = client.count(
      collection_name="SpeakerRecognition")
  
  # Declare `my_global_variable` as global
 # global labels
  embeddings = []
  for i,x in tqdm(enumerate(folder),total=len(folder)):
    if x.endswith(".wav"):
        tmp, fs = torchaudio.load(x)
        e = classifier.encode_batch(tmp)
        e=e.cpu()
        #labels[i+a.count+1]=text
        
        embedding_dict = {"id": i+a.count+1, "vector": e[0, 0].numpy()}
        embeddings.append(embedding_dict)
        speaker[i+a.count+1]=text
     
  client.upsert('SpeakerRecognition', embeddings)
  #count=count.append(a)
  write_dict_to_csv(speaker)#return f"{len(labels)}+dict_keys:{embedding_dict.keys}+labels[0]:{labels[0]}+spk_label:{spk_label}"
  #return f"done upserting of: {text} and embedding size={e.shape} and total audio samples={len(folder)}"  
  #return speaker
  return a.count


@app.post("/publish/")
async def publish_endpoint(input_data: InputData):
    return process_data(input_data.folder, input_data.text)


@app.post("/predict/")
async def predict(input_audio:InputAudio):
    return search(input_audio.text)
   
    #output_text = input_data.text.upper()  # Example: Convert text to uppercase
    #return {"result": output_text}
