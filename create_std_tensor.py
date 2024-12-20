import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import argparse


model_cards = {
    'sent_t5_large': 'sentence-t5-large',
    'sent_t5_base': 'sentence-t5-base',
    'sent_t5_xl': 'sentence-t5-xl',
    'sent_t5_xxl': 'sentence-t5-xxl',
    'mpnet': 'all-mpnet-base-v1',
    'sent_roberta': 'all-roberta-large-v1',
    'simcse_bert': 'princeton-nlp/sup-simcse-bert-large-uncased',
    'simcse_roberta': 'princeton-nlp/sup-simcse-roberta-large'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embedding_model(model_name):
    if model_name in ['simcse_bert', 'simcse_roberta']:
        embed_model = AutoModel.from_pretrained(model_cards[model_name]).to(device)
        embed_tokenizer = AutoTokenizer.from_pretrained(model_cards[model_name])
    else:
        embed_model = SentenceTransformer(model_cards[model_name], device=device)
        embed_tokenizer = None
    return embed_model, embed_tokenizer


def get_embeddings(embed_model, sentences, embed_tokenizer=None):
    if embed_tokenizer:
        inputs = embed_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        embeddings = embed_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    else:
        embeddings = embed_model.encode(sentences, convert_to_tensor=True).to(device)
    return embeddings


def calculate_absolute_differences(embeddings):
    diffs = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
    abs_diffs = torch.abs(diffs)
    return abs_diffs


def calculate_average_differences(abs_differences):
    avg_diffs_per_entry = torch.mean(abs_differences, dim=1)
    avg_of_avg_diffs = torch.mean(avg_diffs_per_entry, dim=0)
    return avg_diffs_per_entry, avg_of_avg_diffs


def calculate_topic_similarity(embeddings_topic1, embeddings_topic2):
    similarity = torch.nn.functional.cosine_similarity(embeddings_topic1.unsqueeze(1), embeddings_topic2.unsqueeze(0), dim=2)
    avg_similarity = torch.mean(similarity)
    return avg_similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creating the standard deviation tensor of an embedding model')
    parser.add_argument('--embed_model', type=str, default='sent_t5_large', help='Name of embedding model')
    args = parser.parse_args()

    model_name = args.embed_model

    topics = {
        "colors": [
            "The sky is blue.",
            "The grass is green.",
            "The apple is red.",
            "The sun is yellow."
        ],
        "animals": [
            "The cat is sleeping.",
            "The dog is barking.",
            "A bird is flying.",
            "A horse is running."
        ],
        "emotions": [
            "He is happy.",
            "She feels sad.",
            "They are excited.",
            "I am nervous."
        ],
        "genders": [
            "He is a boy.",
            "She is a girl.",
            "They are non-binary.",
            "He is a man."
        ],
        "race": [
            "He is Caucasian.",
            "She is Asian.",
            "They are Black.",
            "He is Hispanic."
        ],
        "ethnicity": [
            "He is from the Netherlands.",
            "She is from Spain.",
            "They are from Mexico.",
            "He is from Japan."
        ]
    }


    embed_model, embed_tokenizer = load_embedding_model(model_name)

    avg_diffs_per_topic = {}
    avg_diffs_embeddings = []
    for topic, sentences in topics.items():
        embeddings = get_embeddings(embed_model, sentences, embed_tokenizer)
        abs_differences = calculate_absolute_differences(embeddings)
        avg_diffs_per_entry, avg_of_avg_diffs = calculate_average_differences(abs_differences)
        avg_diffs_per_topic[topic] = avg_of_avg_diffs
        avg_diffs_embeddings.append(avg_diffs_per_entry)
        print(f"Average absolute difference for topic '{topic}': {avg_of_avg_diffs}")

   
    avg_diffs_embeddings = torch.cat(avg_diffs_embeddings)

    topic_names = list(topics.keys())
    for i in range(len(topic_names)):
        for j in range(i + 1, len(topic_names)):
            topic1, topic2 = topic_names[i], topic_names[j]
            embeddings_topic1 = get_embeddings(embed_model, topics[topic1], embed_tokenizer)
            embeddings_topic2 = get_embeddings(embed_model, topics[topic2], embed_tokenizer)
            avg_similarity = calculate_topic_similarity(embeddings_topic1, embeddings_topic2)
            print(f"Average similarity between topics '{topic1}' and '{topic2}': {avg_similarity.item()}")

    overall_avg_diff = torch.mean(torch.stack(list(avg_diffs_per_topic.values())), dim=0)
    print(f"Overall average of average differences for all topics (shape {overall_avg_diff.shape}): {overall_avg_diff}")

    torch.save(overall_avg_diff, f'overall_avg_diff_{model_name}.pt')

    std_tensor = torch.sqrt(overall_avg_diff)
    batch_size = 64
    noise = torch.normal(mean=0.0, std=std_tensor.unsqueeze(0).expand(batch_size, -1))
    
