"""CA-AGFN: Clash-Aware Adaptive Gated Fusion Network.

A meme is rarely hateful in its text alone or its image alone. It is hateful in
the gap between them: a caption that is innocuous under one picture and vicious
under another. So the model builds that gap as an explicit feature rather than
hoping a classifier finds it in a concatenation.

Three pieces, in order.

`CrossModalAttention` lets each modality read the other, and returns the text
attention distribution, which the gate needs.

`SemanticClash` builds the gap itself from the two pooled representations, as
|t - v| concatenated with t * v: the first says how far apart they are, the
second where they agree. A linear layer turns that into the clash vector.

`AdaptiveGatedFusion` decides how much to trust each side. The novel term is the
language-conditioned entropy gate: the entropy of the text's attention over the
image says how sharply the text is pointing at anything. Text that commits to a
region is text worth weighting; text spread evenly over the whole image is text
that is not saying much, and the image should decide instead.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel

EPSILON = 1e-9


class CrossModalAttention(nn.Module):
    """Bidirectional attention, returning the text-to-vision distribution.

    The attention weights come back because the entropy gate is computed from
    them. That is the only reason this returns three things instead of two.
    """

    def __init__(self, hidden_size, n_heads=8, dropout=0.1):
        super().__init__()
        self.text_to_vision = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.vision_to_text = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.norm_text = nn.LayerNorm(hidden_size)
        self.norm_vision = nn.LayerNorm(hidden_size)

    def forward(self, text, vision, text_mask=None):
        padding = (text_mask == 0) if text_mask is not None else None
        if padding is not None:
            # A row that is entirely padding makes the softmax divide by zero
            # and returns NaN for the whole batch. Leaving its first position
            # visible costs nothing: the row has no content to contribute.
            padding = padding.clone()
            padding[padding.all(dim=1), 0] = False

        # Per head, not averaged across them. The default averages the eight
        # heads together before returning, and the mean of several sharp
        # distributions peaked in different places is a flat one: the entropy
        # of a mixture is never below the mean of its components' entropies.
        # Measured on the validation split, the averaged weights put the
        # entropy at 0.99 for every single meme, which is a constant, and a
        # constant cannot drive a gate.
        attended_text, weights = self.text_to_vision(
            query=text, key=vision, value=vision, average_attn_weights=False)
        text_out = self.norm_text(text + attended_text)

        attended_vision, _ = self.vision_to_text(
            query=vision, key=text, value=text, key_padding_mask=padding)
        vision_out = self.norm_vision(vision + attended_vision)
        return text_out, vision_out, weights


class SemanticClash(nn.Module):
    """The disagreement between the modalities, as a vector.

    |t - v| is magnitude of disagreement, t * v is agreement per dimension.
    Together they are what a concatenation of t and v does not make explicit.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.project = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, text_cls, vision_cls):
        difference = torch.abs(text_cls - vision_cls)
        product = text_cls * vision_cls
        return self.norm(self.activation(
            self.project(torch.cat([difference, product], dim=-1))))


class AdaptiveGatedFusion(nn.Module):
    """Two gates, blended by a learned scalar.

    The text gate reads the clash and the entropy of the text's attention. The
    vision gate reads the clash and the image. They are two different theories
    of which modality to trust, and `lambda` lets the model interpolate between
    them rather than committing to one at design time.

    Entropy is normalised by log(sequence length), so it lands in [0, 1] and
    does not change meaning when the image patch count changes.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.text_gate = nn.Linear(hidden_size * 2 + 1, 1)
        self.vision_gate = nn.Linear(hidden_size * 2, 1)
        # sigmoid(0) = 0.5: an even mix of the two strategies at init.
        self.blend = nn.Parameter(torch.zeros(1))

    @staticmethod
    def attention_entropy(weights):
        """Normalised entropy of the text's attention over the image.

        High means the text is looking everywhere, which is the same as looking
        nowhere. Low means it is pointing at something.

        Accepts (batch, tokens, patches) or (batch, heads, tokens, patches).
        With heads, the entropy is taken per head and then averaged, never the
        other way round: entropy is concave, so the entropy of the averaged
        distribution is at least the average of the entropies, and usually far
        above it. Eight sharp heads pointing at eight different patches average
        to something almost uniform, and measuring that gives 0.99 for every
        input. Averaging the entropies keeps the sharpness each head actually
        has.

        The weights are renormalised first. `nn.MultiheadAttention` applies
        dropout to the attention probabilities in training mode, so what comes
        back sums to somewhere near 0.9 rather than to 1. Without this the gate
        would mean a different thing in training than at evaluation.

        Dividing by log(n) puts the result in [0, 1] and keeps it comparable
        across image encoders with different patch counts.
        """
        weights = weights.clamp_min(0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPSILON)

        per_position = -(weights * torch.log(weights + EPSILON)).sum(dim=-1)
        ceiling = torch.log(torch.tensor(
            float(weights.size(-1)), device=weights.device)).clamp_min(EPSILON)
        normalised = per_position / ceiling

        # Average over every axis but the batch: tokens, and heads if present.
        return normalised.flatten(start_dim=1).mean(dim=1, keepdim=True)

    def forward(self, text_cls, vision_cls, clash, entropy):
        text_weight = torch.sigmoid(
            self.text_gate(torch.cat([text_cls, clash, entropy], dim=-1)))
        vision_weight = torch.sigmoid(
            self.vision_gate(torch.cat([vision_cls, clash], dim=-1)))

        blend = torch.sigmoid(self.blend)
        effective = blend * text_weight + (1 - blend) * (1 - vision_weight)
        return effective * text_cls + (1 - effective) * vision_cls, effective


class CAAGFN(nn.Module):
    """The whole network: two frozen-ish backbones and the fusion above."""

    CUSTOM_MODULES = (
        "vision_projection", "cross_attention", "clash", "fusion", "classifier",
    )

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_encoder = AutoModel.from_pretrained(config.text_model)
        self.vision_encoder = CLIPVisionModel.from_pretrained(config.vision_model)

        text_width = self.text_encoder.config.hidden_size
        vision_width = self.vision_encoder.config.hidden_size
        if text_width != config.hidden_size:
            raise ValueError(
                f"{config.text_model} has width {text_width}, but the config "
                f"says {config.hidden_size}."
            )

        self.vision_projection = nn.Linear(vision_width, config.hidden_size)
        self.cross_attention = CrossModalAttention(config.hidden_size)
        self.clash = SemanticClash(config.hidden_size)
        self.fusion = AdaptiveGatedFusion(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.freeze_backbones()

    def freeze_backbones(self):
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_top(self, n_layers=2):
        """Open the top `n_layers` transformer blocks of each backbone.

        Only blocks. Embeddings, relative-position tables and the final layer
        norms stay frozen: unfreezing those is what made the earlier version of
        this model diverge on the first optimiser step of phase two.
        """
        for name, parameter in self.text_encoder.named_parameters():
            parameter.requires_grad = self._in_top_block(
                name, "encoder.layer.", n_layers)
        for name, parameter in self.vision_encoder.named_parameters():
            parameter.requires_grad = self._in_top_block(
                name, "encoder.layers.", n_layers)

    @staticmethod
    def _in_top_block(name, marker, n_layers, total=12):
        if marker not in name:
            return False
        index = int(name.split(marker)[1].split(".")[0])
        return index >= total - n_layers

    def trainable_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def forward(self, input_ids, attention_mask, pixel_values,
                return_gate=False):
        text = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        vision = self.vision_projection(
            self.vision_encoder(pixel_values=pixel_values).last_hidden_state)

        text_out, vision_out, weights = self.cross_attention(
            text, vision, attention_mask)

        # Position 0 is the pooled token for both XLM-R and CLIP-ViT.
        text_cls, vision_cls = text_out[:, 0, :], vision_out[:, 0, :]
        clash = self.clash(text_cls, vision_cls)
        entropy = self.fusion.attention_entropy(weights)

        fused, gate = self.fusion(text_cls, vision_cls, clash, entropy)
        logits = self.classifier(self.dropout(fused)).squeeze(-1)

        if return_gate:
            return logits, {"gate": gate.squeeze(-1), "entropy": entropy.squeeze(-1)}
        return logits
