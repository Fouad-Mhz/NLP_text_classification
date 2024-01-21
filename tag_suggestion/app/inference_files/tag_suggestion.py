import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import re, nltk, zipfile, os, pickle, sys
from bs4 import BeautifulSoup
from xgboost import XGBClassifier
from nltk.tokenize import MWETokenizer
from sklearn.preprocessing import MultiLabelBinarizer

folder_path = './tag_suggestion/app/inference_files/'  #'/content/drive/MyDrive/Datasets/'
models_folder_path= folder_path + 'xgboost_models/'
mlb_filename = folder_path + 'mlb_model.sav'
meta_model_filename = folder_path + 'meta_model_use.json'

def download_nltk_resources():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')

# Function to load all models from a folder
def load_base_models(folder_path = models_folder_path, nb_folds=20):
    loaded_models = []

    for i in range(nb_folds):
        model_filename = f'{folder_path}model_fold_{i}.json'
        loaded_model = XGBClassifier()
        loaded_model.load_model(model_filename)
        loaded_models.append(loaded_model)

    return loaded_models

def load_meta_model(meta_model_filename = meta_model_filename):

    # Create an instance of XGBClassifier
    meta_model = XGBClassifier()

    # Load the saved meta model
    meta_model.load_model(meta_model_filename)

    return meta_model

def import_resources():
    print("Downloading NLTK resources...")
    # Download NLTK resources
    download_nltk_resources()
    print("NLTK resources downloaded successfully.")

    print("Loading MultiLabelBinarizer...")
    # Load the MultiLabelBinarizer from the saved file
    mlb = pickle.load(open(mlb_filename, 'rb'))
    print("MultiLabelBinarizer loaded successfully.")

    print("Loading meta model...")
    # Load the meta model
    meta_model = load_meta_model()
    print("Meta model loaded successfully.")
    
    print("Loading base models...")
    # Load the base models
    loaded_models = load_base_models()
    print("Base models loaded successfully.")

    print("Loading Universal Sentence Encoder...")
    # Load Universal Sentence Encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Universal Sentence Encoder loaded successfully.")

    return embed, mlb, loaded_models, meta_model

embed, mlb, loaded_models, meta_model = import_resources()


# Preprocessing, Tokenization and Embedding functions

def initialize_tokenizer(mlb):
    tokenizer = MWETokenizer()
    mwe_tuple = tuple(mlb.classes_)
    tokenizer.add_mwe(mwe_tuple)
    return tokenizer

def replace_contractions(text):
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    return text

def tokenizer_fct(sentence, mlb):
    text_without_tags = BeautifulSoup(sentence, 'html.parser').get_text()
    text_without_tags = replace_contractions(text_without_tags)
    cleaned_text = re.sub(r'([,\?])|\s+|[^a-zA-Z,?#+]', lambda match: ' ' + match.group(1) + ' ' if match.group(1) else ' ', text_without_tags)
    tokenizer = initialize_tokenizer(mlb)
    word_tokens = tokenizer.tokenize(cleaned_text.split())
    return word_tokens

def transform_dl_fct(desc_text, mlb):
    word_tokens = tokenizer_fct(desc_text, mlb)
    lw = [w.lower() for w in word_tokens]
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

def multi_predict_tags(embedded_sentence, mlb, base_models):
    all_predicted_tags = set()

    for base_model in base_models:
        base_pred_new_sentence = base_model.predict(embedded_sentence)
        predicted_labels = mlb.inverse_transform(base_pred_new_sentence)[0]
        all_predicted_tags.update(predicted_labels)

    multi_models_suggested_tags = list(all_predicted_tags)
    return multi_models_suggested_tags

def meta_predict_tags(embedded_sentence, meta_model, base_models, mlb):
    base_preds = [base_model.predict(embedded_sentence) for base_model in base_models]
    stacked_X = np.column_stack(base_preds)

    # Use the meta model to make predictions on the stacked base model outputs
    y_pred_prob_meta = meta_model.predict(stacked_X)

    # Apply a threshold (you may adjust this based on your needs)
    threshold = 0.5
    y_pred_meta = (y_pred_prob_meta > threshold).astype(int)

    # Inverse transform to get the predicted labels
    predicted_labels = mlb.inverse_transform(y_pred_meta)[0]
  
    return predicted_labels

def inference(sentence, embed=embed, mlb=mlb, base_models=loaded_models, meta_model=meta_model):
    embedded_sentence = embed([transform_dl_fct(sentence, mlb)])
    
    # Make predictions
    multi_result = multi_predict_tags(embedded_sentence, mlb, base_models)
    meta_result = meta_predict_tags(embedded_sentence, meta_model, base_models, mlb)

    return multi_result, meta_result
