"""CA-AGFN: Clash-Aware Adaptive Gated Fusion Network.

A meme is rarely hateful in its text alone or its image alone. It is hateful in
the gap between them: a caption that is innocuous under one picture and vicious
under another. So the model builds that gap as an explicit feature rather than
hoping a classifier finds it in a concatenation.

Three pieces, in order.

`SemanticClash` builds the gap itself from the two pooled vectors, taken before
anything has mixed the modalities, as |t - v| concatenated with t * v. Both
vectors come out of CLIP's shared space, so the distance between them is the
distance CLIP was trained to make small for a caption and its own picture.

`CrossModalAttention` lets each modality read the other, and returns the text's
attention over the image, per head, which the gate needs.

`AdaptiveGatedFusion` decides how much to trust each side. The text gate reads
the entropy of the text's attention over the image: text that commits to a
region is text worth weighting; text spread evenly over the whole image is text
that is not saying much, and the image should decide instead.

The classifier then reads the fused vector *and* the clash. A convex mix of the
text and the image cannot express that the two disagree, so the clash has to
reach the classifier directly rather than only through a scalar gate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

EPSILON = 1e-9


def masked_mean(states, mask):
    """Average over the real tokens, never the padding.

    Captions are padded to a fixed length, and a short caption is mostly
    padding: "no need to panic" is 7 real tokens out of 128.
    """
    weights = mask.to(states.dtype).unsqueeze(-1)
    return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def attention_entropy(weights, mask=None):
    """Normalised entropy of the text's attention over the image, per meme.

    High means the text is looking everywhere, which is the same as looking
    nowhere. Low means it is pointing at something.

    Accepts (batch, tokens, patches) or (batch, heads, tokens, patches).

    Three details decide whether this measures anything.

    The weights are renormalised first. `nn.MultiheadAttention` applies
    dropout to the attention probabilities in training mode, so what comes
    back sums to about 0.9 rather than 1, and the gate would mean a different
    thing in training than at evaluation.

    The entropy is taken per head and averaged afterwards, never the other way
    round. Entropy is concave, so eight sharp heads pointing at eight different
    patches average to something almost uniform.

    Padding positions are excluded through `mask`. Every padded query still
    attends to the image, and before this mask existed they made up 85 to 95
    percent of the average: the entropy the first version reported was mostly
    the entropy of padding.

    Dividing by log(n) puts the result in [0, 1] and keeps it comparable
    across image encoders with different patch counts.
    """
    weights = weights.clamp_min(0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPSILON)

    per_position = -(weights * torch.log(weights + EPSILON)).sum(dim=-1)
    ceiling = torch.log(torch.tensor(
        float(weights.size(-1)), device=weights.device)).clamp_min(EPSILON)
    per_position = per_position / ceiling
    if per_position.dim() == 3:  # heads
        per_position = per_position.mean(dim=1)

    if mask is None:
        return per_position.mean(dim=1, keepdim=True)
    mask = mask.to(per_position.dtype)
    return ((per_position * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)).unsqueeze(-1)


class CrossModalAttention(nn.Module):
    """Bidirectional attention, returning the text-to-vision weights per head.

    The weights come back because the entropy gate is computed from them. That
    is the only reason this returns three things instead of two.
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

        # Per head: the default averages the heads before returning them, and
        # the entropy of that average is flat by construction.
        attended_text, weights = self.text_to_vision(
            query=text, key=vision, value=vision, average_attn_weights=False)
        text_out = self.norm_text(text + attended_text)

        attended_vision, _ = self.vision_to_text(
            query=vision, key=text, value=text, key_padding_mask=padding)
        vision_out = self.norm_vision(vision + attended_vision)
        return text_out, vision_out, weights


class SemanticClash(nn.Module):
    """The disagreement between the modalities, as a vector.

    Both inputs are L2-normalised first, so t * v sums to their cosine
    similarity, which is the quantity CLIP was trained on, and |t - v| is its
    complement spread across dimensions. The projection learns which
    dimensions of agreement and disagreement matter for hate.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.project = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, text, image):
        text = F.normalize(text, dim=-1)
        image = F.normalize(image, dim=-1)
        difference = torch.abs(text - image)
        product = text * image
        return self.norm(self.activation(
            self.project(torch.cat([difference, product], dim=-1))))


class AdaptiveGatedFusion(nn.Module):
    """Two gates, blended by a learned scalar.

    The text gate reads the text, the clash and the entropy of the text's
    attention. The vision gate reads the image and the clash. They are two
    theories of which modality to trust, and `blend` lets the model
    interpolate between them rather than committing to one at design time.
    Either input can be switched off for the ablations.
    """

    def __init__(self, hidden_size, use_clash=True, use_entropy=True):
        super().__init__()
        clash_width = hidden_size if use_clash else 0
        self.text_gate = nn.Linear(
            hidden_size + clash_width + int(use_entropy), 1)
        self.vision_gate = nn.Linear(hidden_size + clash_width, 1)
        # sigmoid(0) = 0.5: an even mix of the two strategies at init.
        self.blend = nn.Parameter(torch.zeros(1))

    def forward(self, text, image, clash=None, entropy=None):
        extra = [clash] if clash is not None else []
        text_weight = torch.sigmoid(self.text_gate(torch.cat(
            [text, *extra, *([entropy] if entropy is not None else [])],
            dim=-1)))
        vision_weight = torch.sigmoid(
            self.vision_gate(torch.cat([image, *extra], dim=-1)))

        blend = torch.sigmoid(self.blend)
        effective = blend * text_weight + (1 - blend) * (1 - vision_weight)
        return effective * text + (1 - effective) * image, effective


def _is_clip(name):
    return AutoConfig.from_pretrained(name).model_type in (
        "clip", "clip_text_model")


def _load_text_encoder(name):
    """The text transformer, and CLIP's projection for it when there is one."""
    if _is_clip(name):
        wrapped = CLIPTextModelWithProjection.from_pretrained(name)
        return wrapped.text_model, wrapped.text_projection.weight.detach().clone()
    return AutoModel.from_pretrained(name), None


def _load_vision_encoder(name):
    wrapped = CLIPVisionModelWithProjection.from_pretrained(name)
    return (wrapped.vision_model,
            wrapped.visual_projection.weight.detach().clone())


def _projection(width, hidden_size, initial=None):
    """A new projection, starting from CLIP's own when the shapes allow it."""
    layer = nn.Linear(width, hidden_size)
    if initial is not None and initial.shape == layer.weight.shape:
        with torch.no_grad():
            layer.weight.copy_(initial)
            layer.bias.zero_()
    return layer


def _blocks(encoder):
    """The stack of transformer blocks, whatever the encoder calls it."""
    stack = encoder.encoder
    return stack.layers if hasattr(stack, "layers") else stack.layer


class CAAGFN(nn.Module):
    """Two mostly frozen backbones and the fusion above.

    `config.fusion` picks the fusion, so every baseline in the results table is
    this class with a different setting and not separate code.
    """

    CUSTOM_MODULES = (
        "text_projection", "vision_projection", "cross_attention", "clash",
        "fusion", "head",
    )

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden = config.hidden_size

        if config.uses_text:
            self.text_encoder, initial = _load_text_encoder(config.text_model)
            self.text_is_clip = initial is not None
            self.text_projection = _projection(
                self.text_encoder.config.hidden_size, hidden, initial)
        if config.uses_image:
            self.vision_encoder, initial = _load_vision_encoder(
                config.vision_model)
            self.vision_projection = _projection(
                self.vision_encoder.config.hidden_size, hidden, initial)

        width = hidden
        if config.fusion == "concat":
            width = 2 * hidden
        elif config.fusion == "concat_clash":
            self.clash = SemanticClash(hidden)
            width = 3 * hidden
        elif config.fusion == "gated":
            self.cross_attention = CrossModalAttention(
                hidden, config.attention_heads)
            self.fusion = AdaptiveGatedFusion(
                hidden, config.use_clash, config.use_entropy)
            if config.use_clash:
                self.clash = SemanticClash(hidden)
                width = 2 * hidden

        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        self.freeze_backbones()

    def backbones(self):
        return [module for module in (getattr(self, "text_encoder", None),
                                      getattr(self, "vision_encoder", None))
                if module is not None]

    def freeze_backbones(self):
        for backbone in self.backbones():
            for parameter in backbone.parameters():
                parameter.requires_grad = False

    def unfreeze_top(self, n_layers=2):
        """Open the top `n_layers` transformer blocks of each backbone.

        Only blocks. Embeddings, position tables and the final layer norms stay
        frozen: opening those sends a large gradient into the parts of a
        pretrained encoder least able to absorb it.
        """
        self.freeze_backbones()
        for block in self._top_blocks(n_layers):
            for parameter in block.parameters():
                parameter.requires_grad = True

    def _top_blocks(self, n_layers):
        if n_layers <= 0:
            return []
        return [block for backbone in self.backbones()
                for block in _blocks(backbone)[-n_layers:]]

    def block_depths(self):
        """Parameter id to depth from the top (0 is the last block)."""
        depths = {}
        for backbone in self.backbones():
            blocks = _blocks(backbone)
            for index, block in enumerate(blocks):
                for parameter in block.parameters():
                    depths[id(parameter)] = len(blocks) - 1 - index
        return depths

    def is_custom(self, name):
        return name.split(".")[0] in self.CUSTOM_MODULES

    def trainable_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def tuned_state_dict(self):
        """Only the weights training can change.

        The new modules and the top blocks: about 100 MB instead of the full
        state dict, most of which is frozen CLIP that `from_pretrained` already
        restores.
        """
        top = {id(p) for block in self._top_blocks(self.config.unfreeze_top)
               for p in block.parameters()}
        return {name: parameter.detach().cpu().clone()
                for name, parameter in self.named_parameters()
                if self.is_custom(name) or id(parameter) in top}

    def load_tuned_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        absent = [name for name in missing if self.is_custom(name)]
        if unexpected or absent:
            raise RuntimeError(
                f"Checkpoint does not match this model: unexpected "
                f"{unexpected[:5]}, missing {absent[:5]}.")

    def encode_text(self, input_ids, attention_mask):
        output = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        tokens = output.last_hidden_state
        if self.text_is_clip:
            # The end-of-text token, which is what CLIP aligns with the image.
            pooled = output.pooler_output
        else:
            # RoBERTa's <s> was never trained as a summary of the sentence.
            pooled = masked_mean(tokens, attention_mask)
        return self.text_projection(tokens), self.text_projection(pooled)

    def encode_image(self, pixel_values):
        output = self.vision_encoder(pixel_values=pixel_values)
        # CLIP normalises its class token before projecting it; the same norm
        # over every patch keeps them on the scale the projection expects.
        tokens = self.vision_projection(
            self.vision_encoder.post_layernorm(output.last_hidden_state))
        return tokens, tokens[:, 0]

    def forward(self, input_ids, attention_mask, pixel_values,
                return_details=False):
        config = self.config
        details = {}
        if config.uses_text:
            text_tokens, text = self.encode_text(input_ids, attention_mask)
        if config.uses_image:
            image_tokens, image = self.encode_image(pixel_values)

        if config.fusion == "text":
            features = text
        elif config.fusion == "image":
            features = image
        elif config.fusion == "concat":
            features = torch.cat([text, image], dim=-1)
        elif config.fusion == "concat_clash":
            features = torch.cat([text, image, self.clash(text, image)], dim=-1)
        else:
            text_seq, image_seq, weights = self.cross_attention(
                text_tokens, image_tokens, attention_mask)
            entropy = attention_entropy(weights, attention_mask)
            clash = self.clash(text, image) if config.use_clash else None

            fused, gate = self.fusion(
                masked_mean(text_seq, attention_mask), image_seq[:, 0],
                clash, entropy if config.use_entropy else None)
            features = fused if clash is None else torch.cat(
                [fused, clash], dim=-1)
            details = {"gate": gate.squeeze(-1), "entropy": entropy.squeeze(-1)}

        logits = self.head(features).squeeze(-1)
        return (logits, details) if return_details else logits
