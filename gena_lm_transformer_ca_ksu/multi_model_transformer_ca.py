import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


class SinusoidalPositionEmbedding(layers.Layer):
    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.positional_encoding = self._build_positional_encoding(self.max_len, self.d_model)

    @staticmethod
    def _build_positional_encoding(max_len: int, d_model: int) -> tf.Tensor:
        positions = np.arange(max_len)[:, np.newaxis]
        div_terms = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(positions * div_terms)
        pe[:, 1::2] = np.cos(positions * div_terms)
        return tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        return x + self.positional_encoding[:, :seq_len, :]

    def compute_mask(self, inputs, mask=None):
        return mask


class MaskedMeanPooling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        x, mask = inputs
        mask = tf.cast(mask, x.dtype)[..., None]
        summed = tf.reduce_sum(x * mask, axis=1)
        denom = tf.reduce_sum(mask, axis=1)
        denom = tf.maximum(denom, tf.ones_like(denom))
        return summed / denom


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        key_dim = max(1, d_model // num_heads)
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ]
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x: tf.Tensor, mask: tf.Tensor | None = None, training: bool | None = None) -> tf.Tensor:
        attention_mask = None
        if mask is not None:
            mask = tf.cast(mask, tf.bool)
            attention_mask = tf.logical_and(tf.expand_dims(mask, 1), tf.expand_dims(mask, 2))
        attn_output = self.att(
            query=x,
            value=x,
            key=x,
            attention_mask=attention_mask,
            training=training,
        )
        x = self.norm1(x + self.drop1(attn_output, training=training))
        ffn_output = self.ffn(x, training=training)
        return self.norm2(x + self.drop2(ffn_output, training=training))

    def compute_mask(self, inputs, mask=None):
        return mask


class CrossAttentionBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        key_dim = max(1, d_model // num_heads)
        self.cross_att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ]
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(
        self,
        query_x: tf.Tensor,
        key_value_x: tf.Tensor,
        query_mask: tf.Tensor | None = None,
        key_value_mask: tf.Tensor | None = None,
        training: bool | None = None,
    ) -> tf.Tensor:
        attention_mask = None
        if query_mask is not None and key_value_mask is not None:
            query_mask = tf.cast(query_mask, tf.bool)
            key_value_mask = tf.cast(key_value_mask, tf.bool)
            attention_mask = tf.logical_and(
                tf.expand_dims(query_mask, 2),
                tf.expand_dims(key_value_mask, 1),
            )
        attn_output = self.cross_att(
            query=query_x,
            value=key_value_x,
            key=key_value_x,
            attention_mask=attention_mask,
            training=training,
        )
        x = self.norm1(query_x + self.drop1(attn_output, training=training))
        ffn_output = self.ffn(x, training=training)
        return self.norm2(x + self.drop2(ffn_output, training=training))


def build_transformer_cross_attention_model(
    seq_len: int,
    seq_dim: int,
    struc_len: int,
    struc_dim: int,
    d_model: int = 256,
    num_heads: int = 8,
    ff_dim: int = 512,
    num_encoder_layers: int = 2,
    dropout: float = 0.2,
    classifier_dim: int = 256,
) -> Model:
    seq_input = layers.Input(shape=(seq_len, seq_dim), name="seq_input")
    struc_input = layers.Input(shape=(struc_len, struc_dim), name="struc_input")

    seq_mask = layers.Lambda(
        lambda x: tf.reduce_any(tf.not_equal(x, 0.0), axis=-1),
        name="seq_mask",
    )(seq_input)
    struc_mask = layers.Lambda(
        lambda x: tf.ones(tf.shape(x)[:2], dtype=tf.bool),
        name="struc_mask",
    )(struc_input)

    seq_x = layers.Dense(d_model, name="seq_projection")(seq_input)
    seq_x = SinusoidalPositionEmbedding(max_len=max(seq_len, 512), d_model=d_model, name="seq_pos")(seq_x)
    seq_x = layers.Dropout(dropout)(seq_x)

    struc_x = layers.Dense(d_model, name="struc_projection")(struc_input)
    struc_x = SinusoidalPositionEmbedding(
        max_len=max(struc_len, 512),
        d_model=d_model,
        name="struc_pos",
    )(struc_x)
    struc_x = layers.Dropout(dropout)(struc_x)

    for idx in range(num_encoder_layers):
        seq_x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"seq_encoder_{idx + 1}",
        )(seq_x, mask=seq_mask)
        struc_x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"struc_encoder_{idx + 1}",
        )(struc_x, mask=struc_mask)

    seq_x = CrossAttentionBlock(
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        name="seq_cross_attend_struc",
    )(seq_x, struc_x, query_mask=seq_mask, key_value_mask=struc_mask)

    struc_x = CrossAttentionBlock(
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        name="struc_cross_attend_seq",
    )(struc_x, seq_x, query_mask=struc_mask, key_value_mask=seq_mask)

    seq_vec = MaskedMeanPooling(name="seq_pool")([seq_x, seq_mask])
    struc_vec = MaskedMeanPooling(name="struc_pool")([struc_x, struc_mask])

    merged = layers.Concatenate(name="fusion")([seq_vec, struc_vec])
    merged = layers.Dense(classifier_dim, activation="gelu", name="cls_dense_1")(merged)
    merged = layers.Dropout(dropout, name="cls_dropout_1")(merged)
    merged = layers.Dense(classifier_dim // 2, activation="gelu", name="cls_dense_2")(merged)
    merged = layers.Dropout(dropout, name="cls_dropout_2")(merged)
    output = layers.Dense(2, activation="softmax", name="classifier")(merged)

    model = Model(inputs=[seq_input, struc_input], outputs=output, name="GENA_LM_TxCA")
    model.summary()
    return model
