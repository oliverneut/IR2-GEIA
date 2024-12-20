import argparse
import stanza
import json
from tqdm import tqdm

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

def calculate_nerr(file_path):
    with open(file_path) as f:
        result = json.load(f)
        gt_total_entity = 0
        correct_pred_count = 0

        for gt, pred in tqdm(zip(result['gt'], result['pred']), total=len(result['gt']), desc='Calculating NERR'):
            doc_gt, doc_pred = nlp(gt), nlp(pred)

            gt_entity_list = [token.text for ent in doc_gt.ents for token in ent.tokens]
            gt_total_entity += len(gt_entity_list)

            gt_entity_set = set(gt_entity_list)
            pred_entity_set = set([token.text for ent in doc_pred.ents for token in ent.tokens])

            correct_set = gt_entity_set & pred_entity_set
            correct_pred_count += len(correct_set)

        ner = correct_pred_count/gt_total_entity

        print(f"correct_pred_count: {correct_pred_count}")
        print(f"ner: {ner}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate generation')
    parser.add_argument('--attack_model', type=str, default='gpt2_medium', help='Name of the attacker model')
    parser.add_argument('--embed_model', type=str, default='sent_t5_large', help='Name of embedding model')
    parser.add_argument('--dataset', type=str, default='personachat', help='Name of dataset: PersonaChat or QNLI')
    parser.add_argument('--beam', type=bool, default=True, help='Toggle beam decoding method (sampling/beam)')

    args = parser.parse_args()

    config = {
        'path': f'logs/{args.dataset}/{args.attack_model}/output_{args.embed_model}{"_beam" if args.beam else ""}.log',
        'hf_path': f'oliverneut/{args.dataset}-{args.attack_model}-{args.embed_model}-attacker'
    }

    file_path = config['path']
    print(f'==={file_path}===')
    calculate_nerr(file_path)