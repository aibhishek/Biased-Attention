import torch
import clip
from PIL import Image
import os
import numpy as np
import pandas as pd
import glob


def get_images(image_dir, model, preprocess):
    
    original_images = []
    images = []

    for name in glob.glob(image_dir):
        if name.endswith(".jpg") or name.endswith(".png")or name.endswith(".jfif"):
            image = Image.open(name).convert("RGB")

            original_images.append(image)
            images.append(preprocess(image))
    
    return original_images, images

def get_probs(original_images, images, labels, model, preprocess):
    
    image_input = torch.tensor(np.stack(images)).cuda()
    
    text_descriptions = [f"This is a photo of a {label}" for label in labels]
    text_tokens = clip.tokenize(text_descriptions).cuda()
    
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    print(text_features.shape)    
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
    print(text_probs.shape)
    return top_probs, top_labels

def aggregate_probs(image_dir, labels, model, preprocess):
    
    #Man
    dir_man = image_dir + 'man/*'
    original_images, images = get_images(dir_man, model, preprocess)
    top_probs, top_labels = get_probs(original_images, images, labels, model, preprocess) 
    
    # 1st probability
    first_label_man = []
    first_prob_man = []
    for i, image in enumerate(original_images):
        index = top_labels[i].numpy()
        probs = top_probs[i].numpy()
        first_label_man.append(labels[index[0]])
        first_prob_man.append(probs[0])
        
    #Woman
    dir_woman = image_dir + 'woman/*'
    original_images, images = get_images(dir_woman, model, preprocess)
    top_probs, top_labels = get_probs(original_images, images, labels, model, preprocess) 
    
    # 1st probability
    first_label_woman = []
    first_prob_woman = []
    for i, image in enumerate(original_images):
        index = top_labels[i].numpy()
        probs = top_probs[i].numpy()
        first_label_woman.append(labels[index[0]])
        first_prob_woman.append(probs[0])
     
    #Generate dataframe
    result_df = pd.DataFrame(list(zip(first_label_man,first_prob_man,first_label_woman,first_prob_woman)),
                                     columns = ['1st Label Man', 'Prob Man 1','1st Label Woman', 'Prob Woman 1'])
    
    return result_df

def get_zero_shot_classifications(model_name, image_dir, keywords):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load(model_name)
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    df = aggregate_probs(image_dir, keywords, model, preprocess)
    df.to_csv('./results/'+model_name+'_df.csv', index=False)

