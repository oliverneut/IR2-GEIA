import argparse
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

stopwords_list = stopwords.words('english')
additional_stopwords = ['!',',','.','?','-s','-ly','</s>','s']
stopwords_list.extend(additional_stopwords)
stopwords_set  = set(stopwords_list)


def calculate_swr(sentence):
    tokens = word_tokenize(sentence.lower())
    stop_word_count = sum(1 for token in tokens if token in stopwords_set)

    return stop_word_count, len(tokens)


def calculate_mean_swr(file_path):
    gt_total, pred_total = 0, 0
    gt_total_sw, pred_total_sw = 0, 0

    with open(file_path) as f:
        result = json.load(f)
        for gt, pred in zip(result['gt'], result['pred']):
            gt_sw_count, gt_total_count = calculate_swr(gt)
            pred_sw_count, pred_total_count = calculate_swr(pred)

            gt_total += gt_total_count
            pred_total += pred_total_count

            gt_total_sw += gt_sw_count
            pred_total_sw += pred_sw_count
    
    print(f'GT stop word ratio (SWR): {gt_total_sw/gt_total}')
    print(f'Pred stop word ratio (SWR): {pred_total_sw/pred_total}')


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
    calculate_mean_swr(file_path)