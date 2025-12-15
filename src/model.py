import torch.nn as nn
from transformers import AutoModelForAudioClassification

import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification


class DogIdentifier(nn.Module):
    def __init__(self, num_classes, model_id, freeze_encoder=True):
        super().__init__()

        print(f"Loading pre-trained model: {model_id}")

        # Load the model
        self.hf_model = AutoModelForAudioClassification.from_pretrained(
            model_id,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # --- Manually Freeze Layers ---
        if freeze_encoder:
            # We explicitly tell PyTorch not to update gradients for the feature projection
            # and the encoder layers, leaving only the classifier head trainable.

            # 1. Freeze the Feature Projection (CNNs)
            # The attribute name is usually 'wav2vec2_bert' or 'model' depending on the specific wrapper
            # We iterate through parameters to be safe
            for name, param in self.hf_model.named_parameters():
                # Freeze everything initially
                param.requires_grad = False

            # 2. Unfreeze the Classification Head
            # We want to train the last layers (classifier / projector)
            for name, param in self.hf_model.named_parameters():
                if "classifier" in name or "projector" in name or "intermediate_dense" in name:
                    param.requires_grad = True

    def forward(self, input_features, attention_mask=None):
        outputs = self.hf_model(
            input_features=input_features,
            attention_mask=attention_mask
        )
        return outputs.logits
