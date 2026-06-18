"""Cross-encoder reranker via plain transformers (avoids FlagEmbedding's slow-tokenizer bug
with newer transformers). Same model as before: BAAI/bge-reranker-v2-m3."""


class Reranker:
    def __init__(self, model_id, device="cuda", max_length=512, batch_size=64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device).eval()

    def score(self, pairs):
        """pairs: list of [query, passage]. Returns list of floats (sigmoid of logit)."""
        torch = self.torch
        out = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i:i + self.batch_size]
            enc = self.tok(chunk, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.view(-1).float()
            out.extend(torch.sigmoid(logits).tolist())
        return out
