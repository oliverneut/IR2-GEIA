from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# Load your model and tokenizer
parser = argparse.ArgumentParser(description='Upload embed-attacker model')
parser.add_argument('--attack_model', type=str, default='gpt2_medium', help='Name of the attacker model')
parser.add_argument('--embed_model', type=str, default='sent_t5_large', help='Name of embedding model')
parser.add_argument('--model_name', type=str, default='gpt2_medium', help='Name of the embed-attacker model')
args = parser.parse_args()

model_name = args.model_name
path = f'models/{args.dataset}/{args.attack_model}/attacker_{args.embed_model}'
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

# Upload the model and tokenizer to Hugging Face
model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)