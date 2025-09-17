import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

import config

class DNABERTSVClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2):
        super(DNABERTSVClassifier, self).__init__()
        
        model_config = AutoConfig.from_pretrained(model_name)
        self.dnabert = AutoModel.from_pretrained(model_name, config=model_config)
        
        # Add a classification head
        self.classifier = nn.Sequential(
            nn.Linear(model_config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        """
        # Get the embeddings from the base DNABERT model
        outputs = self.dnabert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # The pooler_output is designed for classification tasks.
        # It corresponds to the [CLS] token's embedding after further processing.
        pooled_output = outputs.pooler_output
        
        # Pass the embeddings through the classification head
        logits = self.classifier(pooled_output)
        
        return logits

if __name__ == '__main__':
    # Example of instantiating the model
    model = DNABERTSVClassifier(model_name=config.MODEL_NAME)
    
    # Create some dummy input
    dummy_input_ids = torch.randint(0, 10, (4, config.MAX_TOKEN_LENGTH)) # Batch size of 4
    dummy_attention_mask = torch.ones((4, config.MAX_TOKEN_LENGTH))
    
    # Forward pass
    logits = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    print("Model instantiated successfully.")
    print("Output logits shape:", logits.shape) # Should be (4, 2)