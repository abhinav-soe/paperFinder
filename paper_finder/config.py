DATA_PATH = "data/papers.csv"
INDEX_DIR = "index"
EVAL_DIR = "eval"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # fast; try all-mpnet-base-v2 for higher quality

# Seed searches used by fetch_openalex.py to assemble a varied corpus (edit freely).
TOPICS = [
    "fake news detection", "multimodal fake news detection", "misinformation on social media",
    "multimodal deep learning", "graph neural networks", "transformer language models",
    "medical image segmentation", "reinforcement learning", "federated learning privacy",
    "explainable artificial intelligence", "recommender systems", "time series forecasting",
    "speech recognition", "object detection", "large language model evaluation",
    "anomaly detection", "knowledge graph embedding", "generative adversarial networks",
    "sentiment analysis", "adversarial robustness", "autonomous driving perception",
]
