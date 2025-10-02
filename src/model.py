import torch
import torch.nn as nn
from transformers import AutoModel

class HateSpeechClassifier(nn.Module):
    """
    DeBERTa-based model with a multi-layer classifier head and word-level attention,
    reflecting the architecture from the project paper.
    """
    def __init__(self, n_classes, model_name='microsoft/deberta-v3-base'):
        super(HateSpeechClassifier, self).__init__()
        # Load the pre-trained DeBERTa model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Word-level attention layer
        self.attention = nn.Linear(self.bert.config.hidden_size, 1)
        
        # Multi-layer classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, n_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Get the last hidden states from the BERT-based model
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state = outputs.last_hidden_state

        # --- Word-Level Attention Mechanism ---
        # 1. Calculate attention scores for each token
        attention_scores = self.attention(last_hidden_state).squeeze(-1)
        
        # 2. Apply softmax to get normalized attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # 3. Compute the context vector as a weighted sum of the hidden states
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * last_hidden_state, dim=1)
        
        # Pass the context vector through the classifier to get the final logits
        logits = self.classifier(context_vector)
        
        return logits

