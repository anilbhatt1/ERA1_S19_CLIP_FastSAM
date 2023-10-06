import os
import cv2
import numpy as np
from PIL import Image
import json
import gradio as gr
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

class CFG:
    image_path = './images'
    captions_path = './captions'
    batch_size = 64
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        images_similarity = image_embeddings @ text_embeddings.T / self.temperature
        texts_similarity = images_similarity.T
        labels = torch.arange(batch["image"].shape[0]).long().to(CFG.device)

        total_loss = (
            F.cross_entropy(images_similarity, labels) +
            F.cross_entropy(texts_similarity, labels)
        ) / 2

        return total_loss
    
def find_matches_cpu(model, image_embeddings, query, image_filenames, n=4):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to('cpu')
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    return matches

def rle_decode(img_rle_array, img_name, img_size):
    encoded_image = img_rle_array
    # Initialize variables for decoding
    decoded_image = []
    for i in range(0, len(encoded_image), 2):
        pixel_value = encoded_image[i]
        run_length = encoded_image[i + 1]
        decoded_image.extend([pixel_value] * run_length)

    # Convert the decoded image back to a NumPy array
    decoded_array = np.array(decoded_image, dtype=np.uint8)

    # Reshape the decoded array to the original image shape (224, 224)
    decoded_image = decoded_array.reshape(img_size)  # Use original shape

    # Create a PIL Image from the decoded array
    decoded_image = Image.fromarray(decoded_image)

    decoded_image_save_path = './' + str(img_name)
    # Save or display the decoded image
    decoded_image.save(decoded_image_save_path)  # Save the decoded image to a file
    return decoded_image_save_path

def get_matched_image(matches, val_file_dict_loaded):
    img_size = (112, 112)
    match_img_list = []
    for img_name in matches:
        img_rle_array = val_file_dict_loaded[img_name]
        decoded_image_path = rle_decode(img_rle_array, img_name, img_size)
        match_img_list.append(decoded_image_path)
    return match_img_list

def get_grayscale_image(text_query):
    model_inf = CLIPModel().to('cpu')
    model_inf.load_state_dict(torch.load('best_clip_model_cpu.pt', map_location='cpu'))

    clip_image_embeddings_np_inf = np.load('clip_image_embeddings.npy')
    image_embeddings_inf = torch.tensor(clip_image_embeddings_np_inf)

    img_file_names = np.load('val_img_file_names.npy',allow_pickle=True)

    with open("val_imgs_rle_encode.json", "r") as json_file:
        val_file_dict_loaded = json.load(json_file)

    matches = find_matches_cpu(model_inf,
                 image_embeddings_inf,
                 query=text_query,
                 image_filenames=img_file_names,
                 n=1)

    matched_images = get_matched_image(matches, val_file_dict_loaded)
    return matched_images

def gradio_fn(text):
    text_query = str(text)
    match_img_list = get_grayscale_image(text_query)
    pil_img = Image.open(match_img_list[0])
    pil_img = pil_img.resize((224, 224))
    np_img_array = np.array(pil_img)
    return np_img_array

demo = gr.Interface(fn=gradio_fn, 
                    inputs=gr.Textbox(info="Enter the description of image you wish to search, CLIP will give the best image available in corpus that matches your search"), 
                    outputs=gr.Image(height=224, width=224),
                    title="CLIP Image Search")

demo.launch(share=True)