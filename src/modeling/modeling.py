"""
Transformer part of ClipBERT
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from .transformers import BertPreTrainedModel
from .transformers import (
    BertPreTrainingHeads, BertEmbeddings, BertEncoder, BertPooler)
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm


def get_random_sample_indices(
        seq_len, num_samples=100, device=torch.device("cpu")):
    """
    Args:
        seq_len: int, the sampled indices will be in the range [0, seq_len-1]
        num_samples: sample size
        device: torch.device

    Returns:
        1D torch.LongTensor consisting of sorted sample indices
        (sort should not affect the results as we use transformers)
    """
    if num_samples >= seq_len:
        # return all indices
        sample_indices = np.arange(seq_len)
    else:
        sample_indices = np.random.choice(
            seq_len, size=num_samples, replace=False)
        sample_indices = np.sort(sample_indices)
    return torch.from_numpy(sample_indices).long().to(device)


BertLayerNorm = LayerNorm


class VisualInputEmbedding(nn.Module):
    """
    Takes input of both image and video (multi-frame)
    """
    def __init__(self, config):
        super(VisualInputEmbedding, self).__init__()
        self.config = config

        # sequence embedding
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.row_position_embeddings = nn.Embedding(
            config.max_grid_row_position_embeddings,
            config.hidden_size)
        self.col_position_embeddings = nn.Embedding(
            config.max_grid_col_position_embeddings,
            config.hidden_size)
        self.token_type_embeddings = nn.Embedding(1, config.hidden_size)
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, grid):
        """
        Args:
            grid: (B, n_frm, H, W, C), note that #frm can be 1

        Returns:

        """
        bsz, _, _, _, hsz = grid.shape

        # temporal mean pooling
        grid = grid.mean(1)  # (B, H, W, d)
        grid = self.add_2d_positional_embeddings(grid)  # (B, H, W, d)
        # image token sequence
        visual_tokens = grid.view(bsz, -1, hsz)  # (B, H*W, d)

        # perform random sampling. It is only used in training phase
        # of pre-training, but not used in inference or downstream tasks.
        if hasattr(self.config, "pixel_random_sampling_size") and \
                self.config.pixel_random_sampling_size > 0 and self.training:
            sampled_indices = get_random_sample_indices(
                seq_len=visual_tokens.shape[1],
                num_samples=self.config.pixel_random_sampling_size,
                device=visual_tokens.device
            )
            visual_tokens = visual_tokens.index_select(
                dim=1, index=sampled_indices)  # (B, #samples, d)
        visual_tokens_shape = visual_tokens.shape[:-1]  # (B, H*W)
        device = visual_tokens.device

        # image token type embeddings.
        token_type_ids = torch.zeros(
            visual_tokens_shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = visual_tokens + position_embeddings + token_type_embeddings
        embeddings = visual_tokens + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (B, H*W, d)

    def add_temporal_postion_embeddings(self, grid):
        """
        Args:
            grid: (B, n_frms, H, W, d)

        Returns:
            (B, n_frms, H, W, d)
        """
        n_frms, height, width, hsz = grid.shape[-4:]

        # add row-wise position embeddings
        temporal_position_ids = torch.arange(
            n_frms, dtype=torch.long, device=grid.device)  # (n_frms, )
        t_position_embeddings = self.temporal_position_embeddings(
            temporal_position_ids)  # (n_frms, d)
        new_shape = (1, n_frms, 1, 1, hsz)  # (1, n_frms, 1, 1, d)
        grid = grid + t_position_embeddings.view(
            *new_shape)  # broadcast automatically

        return grid

    def add_2d_positional_embeddings(self, grid):
        """
        Args:
            grid: (B, *, H, W, d)

        Returns:
            (B, *, H, W, d)
        """
        height, width, hsz = grid.shape[-3:]

        # add row-wise position embeddings
        row_position_ids = torch.arange(
            height, dtype=torch.long, device=grid.device)  # (H, )
        row_position_embeddings = self.row_position_embeddings(
            row_position_ids)  # (H, d)
        row_shape = (1, ) * (len(grid.shape) - 3) + (
            height, 1, hsz)  # (1, *1, H, 1, d)
        grid = grid + row_position_embeddings.view(
            *row_shape)  # broadcast automatically

        # add column-wise position embeddings
        col_position_ids = torch.arange(
            width, dtype=torch.long, device=grid.device)  # (W, )
        col_position_embeddings = self.col_position_embeddings(
            col_position_ids)  # (W, d)
        col_shape = (1, ) * (len(grid.shape) - 3) + (
            1, width, hsz)  # (1, *1, 1, W, d)
        grid = grid + col_position_embeddings.view(
            *col_shape)  # broadcast automatically
        return grid


class ClipBertBaseModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    config keys:
        text_model: str, text model name, default "bert-based-uncased"
        pretrained: bool, use pre-trained vision_model, default True
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.visual_embeddings = VisualInputEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, text_input_ids, visual_inputs, attention_mask):
        r"""Modified from BertModel
        text_input_ids: (B, Lt)
        visual_inputs: (B, #frame, H, W, C)
        attention_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        """
        input_shape = text_input_ids.size()
        device = text_input_ids.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.

        text_embedding_output = self.embeddings(
            input_ids=text_input_ids)  # (B, Lt, D)
        visual_embedding_output = self.visual_embeddings(
            visual_inputs)  # (B, Lv, d)
        visual_attention_mask = attention_mask.new_ones(
            visual_embedding_output.shape[:2])  # (B, Lv)
        attention_mask = torch.cat(
            [attention_mask, visual_attention_mask], dim=-1)  # (B, lt+Lv, d)
        embedding_output = torch.cat(
            [text_embedding_output, visual_embedding_output],
            dim=1)  # (B, Lt+Lv, d)

        extended_attention_mask: torch.Tensor =\
            self.get_extended_attention_mask(
                attention_mask, input_shape, device)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=self.get_head_mask(
                None, self.config.num_hidden_layers)  # required input
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class ClipBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = ClipBertBaseModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        text_input_ids,
        visual_inputs,
        text_input_mask,
        mlm_labels=None,
        itm_labels=None,
    ):
        r"""
        text_input_ids: (B, Lt)
        visual_inputs: (B, #frame, H, W, C)
        text_input_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        mlm_labels: (B, Lt)
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        itm_label: (B, )  with 1 indicates positive pair, 0 indicates negative pair.
        """

        outputs = self.bert(
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )

        sequence_output, pooled_output = outputs[:2]
        # Only use the text part (which is the first `Lt` tokens) to save computation,
        # this won't cause any issue as cls only has linear layers.
        txt_len = text_input_mask.shape[1]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output[:, :txt_len], pooled_output)

        loss_fct = CrossEntropyLoss(reduction="none")
        if mlm_labels is not None:
            mlm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1))
        else:
            mlm_loss = 0
        if itm_labels is not None:
            itm_loss = loss_fct(
                seq_relationship_score.view(-1, 2), itm_labels.view(-1))
        else:
            itm_loss = 0

        return dict(
            mlm_scores=prediction_scores,  # (B, Lt, vocab_size),  only text part
            mlm_loss=mlm_loss,  # (B, )
            mlm_labels=mlm_labels,  # (B, Lt), with -100 indicates ignored positions
            itm_scores=seq_relationship_score,  # (B, 2)
            itm_loss=itm_loss,  # (B, )
            itm_labels=itm_labels  # (B, )
        )


def instance_bce_with_logits(logits, labels, reduction="mean"):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss


ClipBertForSequenceClassificationConfig = dict(
    cls_hidden_scale=2,   # mlp intermediate layer hidden size scaler
    classifier="mlp",  # classfied type, [mlp, linear]
    num_labels=3129,  # number of labels for classifier output
    loss_type="bce"  # [BCE, CE, KLDivLoss] only used when num_labels > 1
)


class ClipBertForSequenceClassification(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(ClipBertForSequenceClassification, self).__init__(config)
        self.config = config

        self.bert = ClipBertBaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, config.num_labels)
        )

        self.init_weights()

    def forward(self, text_input_ids, visual_inputs,
                text_input_mask, labels=None):
        outputs = self.bert(
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits, loss = self.calc_loss(logits, labels)
        return dict(
            logits=logits,
            loss=loss
        )

    def calc_loss(self, logits, labels):
        if labels is not None:
            if self.config.num_labels == 1:  # regression
                loss_fct = MSELoss(reduction="none")
                # labels = labels.to(torch.float)
                loss = loss_fct(
                    logits.view(-1), labels.view(-1))
            else:
                if self.config.loss_type == 'bce':  # [VQA]
                    loss = instance_bce_with_logits(
                        logits, labels, reduction="none")
                elif self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(reduction="none")
                    loss = loss_fct(
                        logits.view(-1, self.config.num_labels),
                        labels.view(-1))
                else:
                    raise ValueError("Invalid option for config.loss_type")
        else:
            loss = 0
        return logits, loss


class ClipBertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super(ClipBertForMultipleChoice, self).__init__(config)
        self.config = config

        self.bert = ClipBertBaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, 1)
        )
        self.init_weights()

    def forward(self, text_input_ids, visual_inputs,
                text_input_mask, labels=None):
        """
        Args:
            text_input_ids: (B * num_labels, Lt)
            visual_inputs: (B, Lv, d)
            text_input_mask: (B * num_labels, Lt)
            labels: (B, ), in [0, num_labels-1]

        Returns:

        """
        outputs = self.bert(
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits, loss = self.calc_loss(logits, labels)
        return dict(
            logits=logits,
            loss=loss
        )

    def calc_loss(self, logits, labels):
        if self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
            logits = logits.view(-1, self.config.num_labels)

        if labels is not None:
            if self.config.num_labels == 1:  # regression
                loss_fct = MSELoss(reduction="none")
                # labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.config.loss_type == 'bce':  # [VQA]
                    loss = instance_bce_with_logits(
                        logits, labels, reduction="none")
                elif self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(reduction="none")
                    # logits = logits.view(-1, self.config.num_labels)
                    loss = loss_fct(logits, labels.view(-1))
                else:
                    raise ValueError("Invalid option for config.loss_type")
        else:
            loss = 0
        return logits, loss


class ClipBertForRegression(BertPreTrainedModel):
    def __init__(self, config):
        super(ClipBertForRegression, self).__init__(config)
        self.config = config

        self.bert = ClipBertBaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ELU(),
            nn.BatchNorm1d(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1))

        self.init_weights()

    def forward(self, text_input_ids, visual_inputs,
                text_input_mask, labels=None):
        """
        Args:
            text_input_ids: (B * num_labels, Lt)
            visual_inputs: (B, Lv, d)
            text_input_mask: (B * num_labels, Lt)
            labels: (B, ), in [0, num_labels-1]

        Returns:

        """
        outputs = self.bert(
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        logits, loss = self.calc_loss(logits, labels)
        return dict(
            logits=logits,
            loss=loss
        )

    def calc_loss(self, logits, labels):
        if labels is not None:
            if self.config.loss_type == "mse":  # regression
                loss_fct = MSELoss(reduction="none")
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                raise ValueError(f"Invalid option {self.config.loss_type} for config.loss_type")
        else:
            loss = 0
        return logits, loss


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, hidden_states):
        return self.classifier(hidden_states)


class ClipBertForVideoTextRetrieval(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(ClipBertForVideoTextRetrieval, self).__init__(config)
        self.config = config

        self.bert = ClipBertBaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, config.num_labels)
        )
        self.margin = config.margin
        self.init_weights()

    def forward(self, text_input_ids, visual_inputs,
                text_input_mask, labels=None, sample_size=-1):
        outputs = self.bert(
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # rank (B, 1) or ce (B, 2)
        logits, loss = self.calc_loss(logits, labels, sample_size=sample_size)
        return dict(
            logits=logits,
            loss=loss
        )

    def calc_loss(self, logits, labels, sample_size=-1):
        if labels is not None:
            if self.config.loss_type == "ce":
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    logits.view(-1, self.config.num_labels),
                    labels.view(-1))
            elif self.config.loss_type == "rank":
                # triplet loss
                rank_scores_sigmoid = torch.sigmoid(logits).squeeze()  # (B * (#pos=1 + #neg), )
                assert sample_size > 0  # video batch size
                # wrong! scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)
                scores = rank_scores_sigmoid.contiguous().view(sample_size, -1)
                pos = scores[:, :1]  # (B, #pos=1)
                neg = scores[:, 1:]  # (B, #neg)
                loss = torch.clamp(self.margin + neg - pos, min=0)
            else:
                raise ValueError("Invalid option for config.loss_type")
        else:
            loss = 0
        return logits, loss
