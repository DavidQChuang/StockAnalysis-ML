import os

import random 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import gradio as gr

runs = [ 'asdf', 'bf', 'sd' ]

i = 0
def load_runs():
  runs.append("a" + i)
  i += 1

config = {
    "loss" :'mean_squared_error',
    "optimizer":'adam',
    "test_split":0.1,
    "validation_split":0.2,
    "batch_size":64,
    "epochs":16,
    "hidden_layer_size":256,
    "dropout_rate":0.3,
    "seq_len":72,
    "out_seq_len":1
}

data = []
time = pd.Timestamp('2023-06-13T12')
for i in range(395*5):
    price = random.random() + random.randint(27,33)
    time += pd.Timedelta(minutes=5)

    data.append({time: time, price: price})

data = pd.DataFrame(data)

def get_window(df, idx, seq_len):
  r = df.iloc[idx: idx+seq_len]
  return "asfasfasf"

with gr.Blocks() as app:
    gr.Markdown("## Stock Analysis")

    run_config = {}
    with gr.Tab("Runs"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    run_file = gr.Dropdown(runs, label="Run File")
                    run_selection = gr.Dropdown(runs, label="Run")
                refresh_button = gr.Button("Refresh 🔄")
                refresh_button.click(load_runs)
                load_button = gr.Button("Load Model 📥") #💾
                load_button.click(load_runs)
            with gr.Column(scale=3):
                gr.Markdown("""
                #### Model: GatedMLP
                Model config:
                """)
                for key, value in config.items():
                    run_config[key] = gr.Textbox(value=value, label=key)
    with gr.Tab("Model"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                #### Model: GatedMLP

                Epochs: __12__  
                Loss: __0.0012323__  
                Validation Loss: __0.0012323__  
                """)
            with gr.Column(scale=3):
                gr.Markdown(
                """
                ## Inference
                """)

                seq_len = 5
                print(len(data) - seq_len)
                slider = gr.Slider(0, len(data) - seq_len, step=1, label="Count", info="Select an input")
                #out = gr.JSON()
                out2 = gr.Textbox()
                
                slider.change(lambda i: get_window(data, i, seq_len), outputs=[out2])


    app.launch()
