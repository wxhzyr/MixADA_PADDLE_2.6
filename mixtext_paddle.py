# mixtext_paddle.py

import paddle
import paddle.nn as nn

# 导入标准的、公共的模型和组件
from paddlenlp.transformers import (
    BertPretrainedModel,
    RobertaPretrainedModel,
    BertModel,
    RobertaModel
)
from paddlenlp.transformers.roberta.modeling import RobertaClassificationHead


class SentMix(BertPretrainedModel):
    def __init__(self, config):
        super(SentMix, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, "hidden_dropout_prob") else 0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask, input_ids2=None, attention_mask2=None, l=None, mix_layer=1000, **kwargs):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
        extended_attention_mask2 = None
        if input_ids2 is not None:
            if attention_mask2 is None:
                attention_mask2 = paddle.ones_like(input_ids2)
            extended_attention_mask2 = self.get_extended_attention_mask(attention_mask2, input_ids2.shape)

        hidden_states = self.bert.embeddings(input_ids=input_ids)
        hidden_states2 = None
        if input_ids2 is not None:
            hidden_states2 = self.bert.embeddings(input_ids=input_ids2)
        
        if mix_layer == -1 and hidden_states2 is not None:
            hidden_states = l * hidden_states + (1 - l) * hidden_states2
        
        encoder_outputs = self.bert.encoder(hidden_states, extended_attention_mask)
        current_hidden_states = encoder_outputs[0] if isinstance(encoder_outputs, (tuple, list)) else encoder_outputs

        if hidden_states2 is not None:
            encoder_outputs2 = self.bert.encoder(hidden_states2, extended_attention_mask2)
            current_hidden_states2 = encoder_outputs2[0] if isinstance(encoder_outputs2, (tuple, list)) else encoder_outputs2
            if mix_layer >= 0: 
                current_hidden_states[:, 0, :] = l * current_hidden_states[:, 0, :] + (1-l) * current_hidden_states2[:, 0, :]

        pooled_output = self.bert.pooler(current_hidden_states) 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (current_hidden_states, pooled_output) 
        return logits, outputs


class RobertaSentMix(RobertaPretrainedModel):
    def __init__(self, config):
        super(RobertaSentMix, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask, input_ids2=None, attention_mask2=None, l=None, mix_layer=1000, **kwargs):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
        extended_attention_mask2 = None
        if input_ids2 is not None:
            if attention_mask2 is None:
                attention_mask2 = paddle.ones_like(input_ids2)
            extended_attention_mask2 = self.get_extended_attention_mask(attention_mask2, input_ids2.shape)

        hidden_states = self.roberta.embeddings(input_ids=input_ids)
        hidden_states2 = None
        if input_ids2 is not None:
            hidden_states2 = self.roberta.embeddings(input_ids=input_ids2)
        
        if mix_layer == -1 and hidden_states2 is not None:
            hidden_states = l * hidden_states + (1 - l) * hidden_states2
            
        encoder_outputs = self.roberta.encoder(hidden_states, extended_attention_mask)
        current_hidden_states = encoder_outputs[0] if isinstance(encoder_outputs, (tuple, list)) else encoder_outputs

        if hidden_states2 is not None:
            encoder_outputs2 = self.roberta.encoder(hidden_states2, extended_attention_mask2)
            current_hidden_states2 = encoder_outputs2[0] if isinstance(encoder_outputs2, (tuple, list)) else encoder_outputs2
            if mix_layer >= 0: 
                current_hidden_states[:, 0, :] = l * current_hidden_states[:, 0, :] + (1-l) * current_hidden_states2[:, 0, :]

        sequence_output = current_hidden_states
        logits = self.classifier(sequence_output)
        outputs = (sequence_output,) 
        return logits, outputs


class MixText(BertPretrainedModel): 
    def __init__(self, config):
        super(MixText, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, "hidden_dropout_prob") else 0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask, input_ids2=None, attention_mask2=None, l=None, mix_layer=1000, **kwargs):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
        extended_attention_mask2 = None
        if input_ids2 is not None:
            if attention_mask2 is None:
                attention_mask2 = paddle.ones_like(input_ids2)
            extended_attention_mask2 = self.get_extended_attention_mask(attention_mask2, input_ids2.shape)

        hidden_states = self.bert.embeddings(input_ids=input_ids)
        hidden_states2 = None
        if input_ids2 is not None:
            hidden_states2 = self.bert.embeddings(input_ids=input_ids2)
        
        if mix_layer == -1 and hidden_states2 is not None:
            hidden_states = l * hidden_states + (1 - l) * hidden_states2
        
        encoder_outputs = self.bert.encoder(hidden_states, extended_attention_mask)
        current_hidden_states = encoder_outputs[0] if isinstance(encoder_outputs, (tuple, list)) else encoder_outputs

        if hidden_states2 is not None:
            encoder_outputs2 = self.bert.encoder(hidden_states2, extended_attention_mask2)
            current_hidden_states2 = encoder_outputs2[0] if isinstance(encoder_outputs2, (tuple, list)) else encoder_outputs2
            if mix_layer >= 0: 
                current_hidden_states = l * current_hidden_states + (1-l) * current_hidden_states2

        pooled_output = self.bert.pooler(current_hidden_states)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (current_hidden_states, pooled_output)
        return logits, outputs


class RobertaMixText(RobertaPretrainedModel):
    def __init__(self, config):
        super(RobertaMixText, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    # --- 新增的方法 ---
    def get_extended_attention_mask(self, attention_mask, input_shape):
        """
        将 2D attention_mask 扩展为 4D，以便用于 self-attention 计算。
        将 padding 部分设为一个很大的负数，这样在 softmax 后会变为 0。
        """
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )
        
        dtype = self.roberta.embeddings.word_embeddings.weight.dtype
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e4
        return extended_attention_mask.astype(dtype)
    # --- 新增结束 ---

    def forward(self, input_ids, attention_mask, input_ids2=None, attention_mask2=None, l=None, mix_layer=1000, **kwargs):
        # 现在调用我们自己实现的方法
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
        extended_attention_mask2 = None
        if input_ids2 is not None:
            if attention_mask2 is None:
                attention_mask2 = paddle.ones_like(input_ids2)
            # 现在调用我们自己实现的方法
            extended_attention_mask2 = self.get_extended_attention_mask(attention_mask2, input_ids2.shape)

        hidden_states = self.roberta.embeddings(input_ids=input_ids)
        hidden_states2 = None
        if input_ids2 is not None:
            hidden_states2 = self.roberta.embeddings(input_ids=input_ids2)
        
        if mix_layer == -1 and hidden_states2 is not None:
            hidden_states = l * hidden_states + (1 - l) * hidden_states2
            
        encoder_outputs = self.roberta.encoder(hidden_states, extended_attention_mask)
        current_hidden_states = encoder_outputs[0] if isinstance(encoder_outputs, (tuple, list)) else encoder_outputs

        if hidden_states2 is not None:
            encoder_outputs2 = self.roberta.encoder(hidden_states2, extended_attention_mask2)
            current_hidden_states2 = encoder_outputs2[0] if isinstance(encoder_outputs2, (tuple, list)) else encoder_outputs2
            if mix_layer >= 0:
                current_hidden_states = l * current_hidden_states + (1-l) * current_hidden_states2
        
        sequence_output = current_hidden_states
        logits = self.classifier(sequence_output)
        outputs = (sequence_output,)
        return logits, outputs