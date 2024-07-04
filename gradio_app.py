# gradio_app.py

import gradio as gr
import requests

def predict(text):
    
    url = "http://localhost:8000/predict/"  # Replace with your FastAPI server URL
    data = {"text": text}
    response = requests.post(url, json=data)
    
    #return response
    
    results = response.json()
   
    return results


def publish(folder,text):
    url = "http://localhost:8000/publish/"  # Replace with your FastAPI server URL
    print(type(folder))
    data = {"folder": folder, "text": text}
    
    response = requests.post(url, json=data)
   
    #result = response.json()
    return response.json()
    #return result['result']



#iface2=gr.Interface(fn=predict, inputs=[gr.Textbox(label="Enter Text")], outputs="text", title="Add Speaker")
# Create interfaces for each tab
iface1 = gr.Interface(fn=publish, inputs=[gr.File(label="Upload Folder", file_count="directory"), gr.Textbox(label="Enter Text")], outputs="text", title="Add Speaker")
iface2 = gr.Interface(fn=predict, inputs=gr.File(label="Upload Audio File",file_count="single", type="filepath"), outputs="json", title="Find Speaker")

demo = gr.TabbedInterface([iface1, iface2], ["Add Speaker", "Detect Speaker"])
if __name__ == "__main__":
    demo.launch()
