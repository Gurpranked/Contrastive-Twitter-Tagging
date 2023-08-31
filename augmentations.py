import numpy as np
import torch
import scipy
import random
import nltk
import string
import re
import inflect
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#NLPAUG requirements
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action

p = inflect.engine()

path = './NLPAUG_Model_file'

# Downloading Models for the NLPAUG functions
from nlpaug.util.file.download import DownloadUtil

def download_word2vec():
    DownloadUtil.download_word2vec(dest_dir = path)

def gen_aug(sample: str, ssh_type: str) -> str | None:
    if ssh_type == 'na':
        return sample
    elif ssh_type == 'txt_rm_stopwords':
        return text_remove_stopwords(sample)
    elif ssh_type == 'txt_lwr':
        return text_lowercase(sample)
    elif ssh_type == 'txt_convert_num':
        return text_convert_numbers(sample)
    elif ssh_type == 'txt_expand_contractions':
        return text_expand_contractions(sample)
    elif ssh_type == 'txt_cnxt_word_aug':
        return text_contextual_word_aug(sample)
    else:
        print('The task is not available!\n')

# DONE
def text_lowercase(sample: str) -> str:
    """
    Given a sample string, return the same string with all lowercase characters
    
    Args:
        sample (str): string to be made lowercased
    
    Returns:
        A lowercased version of the sample provided string
    """
    return sample.lower()

# DONE
def text_remove_numbers(sample: str) -> str:
    """
    Given a sample string, return the string with all numerica characters removed.

    Args:
        sample(str): The string to be converted

    Returns:
        The same string with all numeric characters removed
    """
    # Splitting the text into a list of words
    list_of_words = sample.split()
    # New Empty List
    new_string = []

    for word in list_of_words:
        # If the word is a digit, ignore it (remove it)
        if word.isdigit():
            continue
        else:
            new_string.append(word)
        
    list_of_words = ' '.join(new_string)
    
    return list_of_words

# DONE
def text_convert_numbers(sample: str) -> str:
    """
    Given a sample string, return the string with all numeric characters converted to their english written form.

    Args:
        sample (str): The string to be converted
    
    Returns:
        A number converted version of th sample provided string
    """
    # Splitting the text to a list of words
    list_of_words = sample.split()
    # New Empty list
    new_string = []

    for word in list_of_words:
        # IF the word is a digit, convert it 
        # Convert it and append it to the the new_string list
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)
        
        # Otherwise leave the word as is
        else:
            new_string.append(word)
        
    # Join the words back to form the converted string
    list_of_words = ' '.join(new_string)

    return list_of_words

# DONE
def text_expand_contractions(sample: str) -> str:
    """
    Given a sample string, take all contraction words and replace them with their full forms.
    
    IE:
        don't -> do not
        can't -> cannot

    Args:
        sample (str): The string from which contractions are to be expanded
    
    Returns:
        The same as the sample provided string with all contractions expanded.
    """
    return contractions.fix(sample)

# Depends if the dataset is english only or a multitude of languages, defaults to english
def text_remove_stopwords(sample: str, language: str = 'english') -> str:
    """
    Given a sample string and a language (default of english). Remove any unnessary stopwords.

    Args:
        sample (str): The string from which the stopwords are to be remove
        language (str) (default = 'english'): The language from which the string is, which influence which stopwords to remove
    
    Returns:
        Returns the sample string with stop words remove, as per the language.
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sample)
    processed_text = ' '.join(word for word in word_tokens if word not in stop_words)
    return processed_text

# DONE
def text_contextual_word_aug(sample: str, aug_max: int = 3) -> str:
    """
    Given a sample string and maximum number of augmentations, return a string with synonyms substituted.

    Args:
        sample (str): The sample string from which the synonyms are to be substituted.
        aug_max (int) (default = 3): The maximum number of synonyms to subtitute within the string

    Returns:
        An augmented version of the provided sample string with aug_max number of synonyms subtituted.
    """
    aug = naw.ContextualWordEmbsAug(model_path = 'bert-base-uncased', model_type = 'bert', action = 'substitute', aug_max = aug_max, device = DEVICE, batch_size = 64)
    return ''.join([word.replace(".", "") for word in (text_lowercase(aug.augment(sample, num_thread=os.cpu_count())[0]))])