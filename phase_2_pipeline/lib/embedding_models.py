from .constants import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

def bi_encoder_model():
    bi_encoder_model = TextEmbedding(EMBEDDING_MODEL_NAME)
    return bi_encoder_model

def cross_encoder_model():
    cross_encoder_model = TextCrossEncoder(RERANKER_MODEL_NAME)
    return cross_encoder_model