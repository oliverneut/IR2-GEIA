import argparse
import json
import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from attacker_models import SequenceCrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from evaluate import load
from attacker_ import setup_optimizer, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer('sentence-t5-base').to(device)
model.eval()

rouge = load('rouge')

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

class text_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):  
        return self.data[index]
        
    def collate(self, unpacked_data):
        return unpacked_data


def read_logs(path):
    with open(path) as f:
        data = json.load(f)
    return data


def get_rouge(data):
    gt = data["gt"]
    pred = data["pred"]
    results = rouge.compute(predictions=pred,references=gt)
    print(results)


def get_bleu(data):
    gt = data['gt']
    pred = data["pred"]
    cands_list_bleu = [sentence.split() for sentence in pred]
    refs_list_bleu = [[sentence.split()] for sentence in gt]
    bleu_score = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu)
    bleu_score_1 = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu,weights=(1, 0, 0, 0)) 
    bleu_score_2 = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu,weights=(0.5, 0.5, 0, 0)) 
    print(f'bleu1 : {bleu_score_1}')
    print(f'bleu2 : {bleu_score_2}')
    print(f'bleu : {bleu_score}')


def batch(iterable, n):
    iterable=iter(iterable)
    while True:
        chunk=[]
        for i in range(n):
            try:
                chunk.append(next(iterable))
            except StopIteration:
                yield chunk
                return
        yield chunk


def embed_similarity(data,batch_size=16):
    gt = data['gt']
    pred = data["pred"]
    
    gt_batch = list(batch(gt, batch_size))
    pred_batch = list(batch(pred, batch_size))
    cosine_scores_all = []
    for gt, pred in tqdm(zip(gt_batch, pred_batch), desc='Embedding similarity'):
        gt_emb = model.encode(gt, convert_to_tensor=True)
        pred_emb = model.encode(pred, convert_to_tensor=True)
        cosine_scores = util.cos_sim(gt_emb, pred_emb)
        assert cosine_scores.size()[0] == cosine_scores.size()[1]
        cosine_scores_all.extend(cosine_scores.diag().tolist())

    avg_score = np.mean(cosine_scores_all)
    print(f'Avg embed similarity: {avg_score}')


def calculate_ppl(data, config):
    dataset = text_dataset(data)
    dataloader = DataLoader(dataset, config['batch_size'], True, collate_fn=dataset.collate)
    
    attack_model = AutoModelForCausalLM.from_pretrained(model_cards[config['attack_model']]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_cards[config['attack_model']])
    tokenizer.pad_token = tokenizer.eos_token
    criterion = SequenceCrossEntropyLoss()

    attack_model.eval()
    
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        running_ppl = []
        for batch_text in tqdm(dataloader, desc='Calculating PPL'):
            inputs = tokenizer(batch_text, return_tensors='pt', padding='max_length', truncation=True, max_length=40)
            input_ids = inputs['input_ids'].to(device) # tensors of input ids
            labels = input_ids.clone()
            
            logits, _ = attack_model(input_ids, past_key_values=None, return_dict=False)
            logits = logits[:, :-1].contiguous()
            target = labels[:, 1:].contiguous()

            target_mask = torch.ones_like(target).float()

            loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")   

            perplexity = np.exp(loss.item())
            
            running_ppl.append(perplexity)

        print(f'Validate ppl: {np.mean(running_ppl)}')


def report_metrics(data, config):
    # get_rouge(data)
    # get_bleu(data)
    # embed_similarity(data)
    calculate_ppl(data['pred'], config)

    # exact_match(data)
    # get_edit_dist(data)
    # embed_similarity(data)


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
        'path': f'logs/{args.dataset}/{args.attack_model}/output_{args.embed_model}{"_beam" if args.beam else ""}.log'
    }

    path_list = [config['path']]
    for p in path_list:
        print(f'==={p}===')
        data = read_logs(p)
        report_metrics(data, config)



