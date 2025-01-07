import os
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
import pandas as pd
import numpy as np
import nltk

def preprocessor(folder_path_of_files):
    """
    Reads all text files in a given folder, tokenizes them into sentences, 
    preprocesses using gensim's simple_preprocess, and returns a list of tokenized sentences.

    Args:
        folder_path (str): Path to the folder containing text files.

    Returns:
        list: A list of preprocessed sentences, where each sentence is a list of tokens.
    """
    
    story= []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open (file_path, encoding= 'unicode_escape') as f:
            corpus = f.read()
        raw_sent = sent_tokenize(corpus)
        # preprocess using simple_preprocess and store in story list
        for sent in raw_sent:
            story.append(simple_preprocess(sent))
            
    return story




if __name__ == "__main__":
    pass
    