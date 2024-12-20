from sklearn import metrics
import numpy as np
import argparse
import torch
from transformers import AutoTokenizer
import json
from sentence_transformers import SentenceTransformer, util
from simcse import SimCSE
from nltk.tokenize import word_tokenize
import string
import re

import nltk
nltk.download('punkt_tab')

model_cards = {
    'sent_t5_large': 'sentence-t5-large',
    'sent_t5_base': 'sentence-t5-base',
    'sent_t5_xl': 'sentence-t5-xl',
    'sent_t5_xxl': 'sentence-t5-xxl',
    'mpnet': 'all-mpnet-base-v1',
    'sent_roberta': 'all-roberta-large-v1',
    'simcse_bert': 'princeton-nlp/sup-simcse-bert-large-uncased',
    'simcse_roberta': 'princeton-nlp/sup-simcse-roberta-large',
    'gpt2_large': 'microsoft/DialoGPT-large',
    'gpt2_medium': 'microsoft/DialoGPT-medium',
    'llama_3_1B': 'meta-llama/Llama-3.2-1B',
    'llama_3_3B': 'meta-llama/Llama-3.2-3B'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vectorize(sent_list, tokenizer):
    turn_ending = tokenizer.encode(tokenizer.eos_token)
    token_num = len(tokenizer)
    
    dial_tokens = [tokenizer.encode(item) + turn_ending for item in sent_list]
    
    max_len = max(len(tokens) for tokens in dial_tokens) 
    dial_tokens_padded = [tokens + [0] * (max_len - len(tokens)) for tokens in dial_tokens] 
    
    dial_tokens_np = np.array(dial_tokens_padded)
    
    input_labels = []
    
    for i in dial_tokens_np:
        temp_i = np.zeros(token_num)
        for token in i:  
            if token < token_num: 
                temp_i[token] = 1
        input_labels.append(temp_i)
    
    input_labels = np.array(input_labels)

    return input_labels


def report_score(y_true,y_pred):
    # micro result should be reported
    precision = metrics.precision_score(y_true, y_pred, average='micro')
    recall = metrics.recall_score(y_true, y_pred, average='micro')
    f1 = metrics.f1_score(y_true, y_pred, average='micro')
    print(f"micro precision_score on token level: {str(precision)}")
    print(f"micro recall_score on token level: {str(recall)}")
    print(f"micro f1_score on token level: {str(f1)}")


def embed_simcse(y_true,y_pred):
    model = SimCSE("princeton-nlp/sup-simcse-roberta-large",device=device)
    similarities = model.similarity(y_true, y_pred) # numpy array of N*N
    pair_scores = similarities.diagonal()
    for i,score in enumerate(pair_scores):
        assert pair_scores[i] == similarities[i][i]
    avg_score = np.mean(pair_scores)
    print(f'Evaluation on simcse-roberta with similarity score {avg_score}')


def embed_sbert(y_true,y_pred):
    model = SentenceTransformer('all-roberta-large-v1',device=device)       # has dim 768
    embeddings_true = model.encode(y_true,convert_to_tensor = True)
    embeddings_pred = model.encode(y_pred,convert_to_tensor = True)
    cosine_scores = util.cos_sim(embeddings_true, embeddings_pred)
    pair_scores = torch.diagonal(cosine_scores, 0)
    for i,score in enumerate(pair_scores):
        assert pair_scores[i] == cosine_scores[i][i]
    avg_score = torch.mean(pair_scores)
    print(f'Evaluation on Sentence-bert with similarity score {avg_score}')
    return avg_score


def report_embedding_similarity(y_true,y_pred):
    embed_sbert(y_true,y_pred)
    embed_simcse(y_true,y_pred)


def main(log_path):
    with open(log_path, 'r') as f:
        sent_dict = json.load(f)
    y_true = sent_dict['gt']     # list of sentences
    y_pred = sent_dict['pred']   # list of sentences   
    report_embedding_similarity(y_true,y_pred)


# remove punctuation from list of sentences 
def punctuation_remove(sent_list):
    removed_list = []
    for sent in sent_list:
        word_list = []
        for word in sent.split():
            word_strip = word.strip(string.punctuation)
            if word_strip:  # cases for not empty string
                word_list.append(word_strip)
        removed_sent = ' '.join(word_list)
        removed_list.append(removed_sent)
    return removed_list

# remove space before punctuation from list of sentences 
def space_remove(sent_list):
    removed_list = []
    for sent in sent_list:
        sent_remove = re.sub(r'\s([?.!"](?:\s|$))', r'\1', sent)
        removed_list.append(sent_remove)
    return removed_list

def metrics_word_level(token_true,token_pred):
    len_pred = len(token_pred)
    len_ture = len(token_true)
    recover_pred = 0
    recover_true = 0
    for p in token_pred:
        if p in token_true:
            recover_pred += 1
    for t in token_true:
        if t in token_pred:
            recover_true += 1
    ### return for precision recall calculation        
    return len_pred,recover_pred,len_ture,recover_true
            
    
def word_level_metrics(y_true,y_pred):
    assert len(y_true) == len(y_pred)
    recover_pred_all = 0
    recover_true_all = 0
    len_pred_all = 0
    len_ture_all = 0
    for i in range(len(y_true)):
        sent_true = y_true[i]
        sent_pred = y_pred[i]
        token_true = word_tokenize(sent_true)
        token_pred = word_tokenize(sent_pred)
        len_pred,recover_pred,len_ture,recover_true = metrics_word_level(token_true,token_pred)
        len_pred_all += len_pred
        recover_pred_all += recover_pred
        len_ture_all += len_ture
        recover_true_all += recover_true
        
        
    ### precision and recall are based on micro (but not exactly)
    precision = recover_pred_all/len_pred_all
    recall = recover_true_all/len_ture_all
    f1 = 2*precision*recall/(precision+recall)
    return precision,recall,f1

def remove_eos(sent_list):
    for i,s in enumerate(sent_list):
        sent_list[i] = s.replace('<|endoftext|>','')

def metric_token(log_path, model_name):
    with open(log_path, 'r') as f:
        sent_dict = json.load(f)
    y_true = sent_dict['gt']     # list of sentences
    y_pred = sent_dict['pred']   # list of sentences   
    
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    y_true_token = vectorize(y_true,tokenizer)
    y_pred_token = vectorize(y_pred,tokenizer)

    ### token-level metrics are reported
    report_score(y_true_token,y_pred_token)
    remove_eos(y_pred)           # make sure to remove <eos>
    ### scores for word level
    y_true_removed_p = punctuation_remove(y_true)       
    y_pred_removed_p = punctuation_remove(y_pred)  
    y_true_removed_s = space_remove(y_true)       
    y_pred_removed_s = space_remove(y_pred)  
    precision,recall,f1 = word_level_metrics(y_true_removed_s,y_pred_removed_s)
    print(f'word level precision: {str(precision)}')
    print(f'word level recall: {str(recall)}')
    print(f'word level f1: {str(f1)}')
    
    precision,recall,f1 = word_level_metrics(y_true_removed_p,y_pred_removed_p)
    print(f'word level precision without punctuation: {str(precision)}')
    print(f'word level recall without punctuation: {str(recall)}')
    print(f'word level f1 without punctuation: {str(f1)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate generation')
    parser.add_argument('--attack_model', type=str, default='gpt2_medium', help='Name of the attacker model')
    parser.add_argument('--embed_model', type=str, default='sent_t5_large', help='Name of embedding model')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--dataset', type=str, default='personachat', help='Name of dataset: PersonaChat or QNLI')
    parser.add_argument('--beam', type=bool, default=True, help='Toggle beam decoding method (sampling/beam)')

    args = parser.parse_args()

    config = {
        'attack_model': args.attack_model,
        'embed_model': args.embed_model,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'path': f'logs/{args.dataset}/{args.attack_model}/output_{args.embed_model}{"_beam" if args.beam else ""}.log',
        'hf_path': f'oliverneut/{args.dataset}-{args.attack_model}-{args.embed_model}-attacker'
    }

    path_list = [config['path']]

    for p in path_list:
        print(f'====={p}=====')
        metric_token(p, model_cards[config['attack_model']])