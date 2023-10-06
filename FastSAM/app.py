import os
import numpy as np
from PIL import Image
import gradio as gr
import torch
import matplotlib.pyplot as plt
from fastsam import FastSAM, FastSAMPrompt

def gradio_fn(pil_input_img):
    # load model
    model = FastSAM('./weights/FastSAM.pt')
    input = pil_input_img
    input = input.convert("RGB")
    everything_results = model(
        input,
        device="cpu",
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9    
        )
    bboxes = None
    points = None
    point_label = None
    prompt_process = FastSAMPrompt(input, everything_results, device="cpu")
    ann = prompt_process.everything_prompt()
    prompt_process.plot(
        annotations=ann,
        output_path="./output.jpg",
        bboxes = bboxes,
        points = points,
        point_label = point_label,
        withContours=False,
        better_quality=False,
    )
    pil_image_output = Image.open('./output.jpg')
    np_img_array = np.array(pil_image_output)
    return np_img_array

example1 = './thanjavur.jpg'
example2 = './jeep.jpeg'
examples = [[example1, 0.5, -1], [example2, 0.5, -1]]

demo = gr.Interface(fn=gradio_fn, 
                    inputs=[gr.Image(type="pil",label="Input Image")], 
                    outputs="image", 
                    title="FAST-SAM Segment Everything",
                    description="- **FastSAM** model that returns segmented RGB image of given input image. \
                                   **Credits** : \
                                    https://huggingface.co/An-619 & \
                                    https://github.com/CASIA-IVA-Lab/FastSAM",
                    examples=examples)

demo.launch(share=True)