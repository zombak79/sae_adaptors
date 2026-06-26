from utils import *

checkpoints=build_checkpoints(CONFIG)

sbert_name = "Qwen/Qwen3-Embedding-0.6B"
print(sbert_name)
sbert = SentenceTransformer(sbert_name, device="cuda")
sbert.max_seq_length=1024
for k in checkpoints.keys():
    print(k, end=": ")
    print(train_and_eval_sbert(checkpoints[k],sbert_name,sbert))

sbert_name = "nomic-ai/nomic-embed-text-v1.5"
print(sbert_name)
sbert = SentenceTransformer(sbert_name, device="cuda")
sbert.max_seq_length=1024
for k in checkpoints.keys():
    print(k, end=": ")
    print(train_and_eval_sbert(checkpoints[k],sbert_name,sbert))


sbert_name = "sentence-transformers/all-mpnet-base-v2"
sbert = SentenceTransformer(sbert_name, device="cuda")
print(sbert_name)
for k in checkpoints.keys():
    print(k, end=": ")
    print(train_and_eval_sbert(checkpoints[k],sbert_name,sbert))

sbert_name = "BAAI/bge-base-en-v1.5"
print(sbert_name)
sbert = SentenceTransformer(sbert_name, device="cuda")
for k in checkpoints.keys():
    print(k, end=": ")
    print(train_and_eval_sbert(checkpoints[k],sbert_name,sbert))


grid = build_grid(sae_config) 
for checkpoint in checkpoints.values():
    print("*"*30)
    print(checkpoint)
    train_and_eval_sae(checkpoint["path"], grid)

grid = build_grid(dae_config) 
for checkpoint in checkpoints.values():
    print("*"*30)
    print(checkpoint)
    train_and_eval_dae(checkpoint["path"], grid)
