from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase_2_pipeline.lib.constants import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME

def bi_encoder_model():
    bi_encoder_model = TextEmbedding(EMBEDDING_MODEL_NAME)
    return bi_encoder_model

def cross_encoder_model():
    cross_encoder_model = TextCrossEncoder(RERANKER_MODEL_NAME)
    return cross_encoder_model