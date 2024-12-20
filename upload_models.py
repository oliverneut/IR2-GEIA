from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# Load your model and tokenizer
parser = argparse.ArgumentParser(description='Upload embed-attacker model')
parser.add_argument('--attack_model', type=str, default='gpt2_medium', help='Name of the attacker model')
parser.add_argument('--embed_model', type=str, default='sent_t5_large', help='Name of embedding model')
parser.add_argument('--dataset', type=str, default='personachat', help='Name of dataset: PersonaChat or QNLI')
args = parser.parse_args()

path = f'models/{args.dataset}/{args.attack_model}/attacker_{args.embed_model}'
model_name = f'{args.dataset}-{args.attack_model}-{args.embed_model}-attacker'
model = AutoModelForCausalLM.from_pretrained(path)

# Upload the model and tokenizer to Hugging Face
model.push_to_hub(model_name)