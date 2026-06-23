import torch

class TransformerLensHFAdapter(torch.nn.Module):
    def __init__(self, tl_model):
        super().__init__()
        self.model = tl_model
        self.tokenizer = tl_model.tokenizer

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_input_embeddings(self):
        return self.model.embed

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # TransformerLens expects tokens directly
        logits = self.model(input_ids)
        return type("HFOutput", (), {"logits": logits})
