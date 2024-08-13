import sys
import os

# Project path
ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))
from discretizers import *
from import_packages import *
from gpt_semantics import *

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

openai.api_key = "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"
model_id = "gpt-3.5-turbo"

ref_bins = [0, 18, 35, 65, 100]
strategies = [
    [0, 19, 45, 65, 85, 100], # gold standard
    [0, 20, 40, 60, 80, 100],
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    [0, 19, 40, 65, 100],
    [0.0, 22.0, 25.0, 29.0, 35.0, 42.0, 59.0],
    [21.0, 24.0, 30.0, 42.0, 43.0, 48.0, 54.0, 59.0],
    [21.0, 30.0, 59.0],
]

sentences = []
for i in range(len(strategies)):
    ref_bins = strategies[i]
    context = "human age"
    messages = []
    prompt1 = f"You are given a set of bins for {context}. Can you describe the semantic meaning of the bins?\nBins: " + str(ref_bins)
    messages.append({"role": "user", "content": prompt1})
    msgresponse = call_gpt(messages, model=model_id)
    print(msgresponse)
    sentences.append(msgresponse)

embeddings = bert_model.encode(sentences)
print(embeddings)
print(embeddings.shape)

similarity_scores = cosine_similarity([embeddings[0]], embeddings[1:])
print(similarity_scores)

print("Done!")