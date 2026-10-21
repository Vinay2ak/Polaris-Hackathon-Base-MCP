import json
import time
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent.parent
data_path = ROOT / 'data/products.json'
out_path = ROOT / 'data/embeddings.npy'

print(f"Loading products from {data_path}...")
products = json.loads(data_path.read_text(encoding='utf-8'))
products = sorted(products, key=lambda p: p['id'])

texts = [
    f"{p.get('name') or ''} {p.get('product_type') or ''} {p.get('color') or ''} {p.get('category') or ''} {p.get('audience') or ''}"
    for p in products
]

print(f"Initializing BAAI/bge-small-en-v1.5 embedding model...")
model = TextEmbedding('BAAI/bge-small-en-v1.5')

print(f"Precomputing embeddings for {len(texts)} products with batch_size=512...")
t0 = time.time()
embeddings = []
# embed returns a generator
for i, emb in enumerate(model.embed(texts, batch_size=512)):
    embeddings.append(emb)
    if (i + 1) % 10000 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        print(f"Processed {i + 1}/{len(texts)} ({rate:.1f} items/s) - ETA: {(len(texts) - i - 1)/rate:.1f}s")

dt = time.time() - t0
print(f"Embedding completed in {dt:.2f}s ({len(texts)/dt:.1f} items/s). Converting to float16...")
arr = np.array(embeddings, dtype=np.float16)
# L2-normalize each vector for instant cosine similarity via dot product
norms = np.linalg.norm(arr.astype(np.float32), axis=1, keepdims=True)
norms[norms == 0] = 1.0
arr = (arr.astype(np.float32) / norms).astype(np.float16)

print(f"Saving float16 embeddings to {out_path} (shape: {arr.shape}, size: {arr.nbytes / (1024*1024):.2f} MB)...")
np.save(out_path, arr)
print("SUCCESS: Embeddings saved!")
