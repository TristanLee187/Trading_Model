# Define the model architectures

from common import *
import tensorflow as tf
from keras import Model
from keras.api.layers import (
    LSTM, Conv1D, Dense, Input, MultiHeadAttention,
    Add, LayerNormalization, Layer, Concatenate
)
from keras.api.regularizers import l2


REG_FACTOR = 1e-3
DROPOUT_FACTOR = 0.2


def custom_categorical_crossentropy(y_true, y_pred):
    """
    Customer categorical-crossentropy loss function on 3 classes that uses weights to
    penalize different classificiations differently.

    Args:
        y_true (np.array): Ground truth one-hot encoding vector.
        y_pred (np.array): Prediction one-hot encoding vector.

    Returns:
        function with arguments (np.array, np.array): Function that computes the loss w.r.t.
            ground truth and prediction inputs.
    """
    # weights[i][j]: penalty for if the ground truth was i but the predicted was j.
    weights = tf.constant([
        [0.0, 2.0, 10.0],
        [3.0, 0.0, 3.0],
        [10.0, 2.0, 0.0],
    ])

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    weights_tensor = tf.reduce_sum(tf.expand_dims(weights, axis=0) * tf.expand_dims(y_true, axis=-1), axis=-2)
    weighted_loss = ce_loss * tf.reduce_sum(weights_tensor, axis=-1)
    return weighted_loss


# Attention pooling
class AttentionPooling(Layer):
    def __init__(self, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.attention = None
        
    def build(self, input_shape):
        # Simple attention
        self.attention = Dense(1)
        
    def call(self, inputs):
        attn_scores = self.attention(inputs)
        # Softmax
        attn_weights = tf.nn.softmax(attn_scores, axis=1)
        return tf.reduce_sum(inputs * attn_weights, axis=1)
    

# Single expert
class Expert(Layer):
    def __init__(self, dim_1, dim_2, **kwargs):
        super(Expert, self).__init__(**kwargs)
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.dense_1 = None
        self.dense_2 = None

    def build(self, input_shape):
        # 2 layer MLP
        self.dense_1 = Dense(self.dim_1, activation='gelu', 
                             kernel_regularizer=l2(REG_FACTOR), bias_regularizer=l2(REG_FACTOR))
        self.dense_2 = Dense(self.dim_2, activation='gelu', 
                             kernel_regularizer=l2(REG_FACTOR), bias_regularizer=l2(REG_FACTOR))
    
    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)
    

# Mixture of experts top k layer, attention pooling
class MoETopKLayer(Layer):
    def __init__(self, num_experts, dim_1, dim_2, top_k, **kwargs):
        super(MoETopKLayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.top_k = top_k

        self.experts = None
        self.attention_pooling = None
        self.gating_network = None

    def build(self, input_shape):
        self.experts = [Expert(self.dim_1, self.dim_2) for _ in range(self.num_experts)]
        self.attention_pooling = AttentionPooling()
        self.gating_network = Dense(self.num_experts, activation='softmax')

    def call(self, inputs):
        # Attention pooling
        attn_inputs = self.attention_pooling(inputs)
        # Get probs for each expert
        gate_outputs = self.gating_network(attn_inputs)
        # Get indices of top k experts
        top_k_values, top_k_indices = tf.nn.top_k(gate_outputs, k=self.top_k)
        # Mask bottom experts, normalize, expand
        mask = tf.reduce_sum(tf.one_hot(top_k_indices, depth=self.num_experts), axis=-2)
        gated_outputs = gate_outputs * mask
        gated_outputs /= tf.reduce_sum(gated_outputs, axis=-1, keepdims=True) + tf.constant(1e-9)
        gated_outputs = tf.expand_dims(tf.expand_dims(gated_outputs, axis=1), axis=1)
        # Weighted avg of top k
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts], axis=-1)
        weighted_expert_outputs = tf.reduce_sum(expert_outputs * gated_outputs, axis=-1)

        return weighted_expert_outputs


def last_layer(label: str):
    """
    Define the last layer of the architectrue depending on the task (given by the label).

    Args:
        label (str): The label type to train on.

    Returns:
        keras.layers.Layer: Keras layer to use for the model's output:
            - "price" (regression): A Dense layer with 1 unit and sigmoid activiation.
            - "signal" (classification): A Dense layer with 5 units and softmax activation.
    """
    if label == 'price':
        return Dense(units=1)
    elif label == 'signal':
        return Dense(units=3, activation='softmax')


def get_transformer_model(shape: tuple, meta_dim: int, label: str):
    """
    Define a transformer (with learnable positional encoding) and MoE based architecture.

    Args:
        shape (tuple[int, int]): shape of each input sequence.
        meta_dim (int): Dimension of the metadata vector.
        label (str):  The label type to train on.

    Returns:
        keras.Model: Model with a transformer-MoE architecture.
    """
    # Transformer block with learnable position encoding and MoE
    def transformer_block(x, num_heads, key_dim, ff_dim_1, ff_dim_2):
        # Positional encoding
        x = Add()([x, LSTM(units=x.shape[2], return_sequences=True, kernel_regularizer=l2(REG_FACTOR))(x)])
        # x = Add()([x, Conv1D(filters=x.shape[1], kernel_size=5, data_format='channels_first', 
        #                      padding='same', activation='gelu', bias_regularizer=l2(REG_FACTOR))(x)])
        
        # Attention!
        attn_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=DROPOUT_FACTOR, 
                                        kernel_regularizer=l2(REG_FACTOR), bias_regularizer=l2(REG_FACTOR))(x, x)
        x = Add()([x, attn_layer])
        x = LayerNormalization(epsilon=1e-8)(x)
        
        # MoE
        moe = MoETopKLayer(num_experts=5, dim_1=ff_dim_1, dim_2=ff_dim_2, top_k=2)(x)
        x = Add()([x, moe])
        x = LayerNormalization(epsilon=1e-8)(x)
        return x

    # Stack of Transformer blocks
    def transformer_stack(x, num_heads, key_dim, ff_dim_1, ff_dim_2, num_blocks):
        for _ in range(num_blocks):
            x = transformer_block(x, num_heads=num_heads, key_dim=key_dim, 
                                  ff_dim_1=ff_dim_1, ff_dim_2=ff_dim_2)
        return x

    # Define the inputs
    seq_input_layer = Input(shape=shape)
    meta_input_layer = Input(shape=meta_dim)

    # Encoder
    encoder = transformer_stack(seq_input_layer, num_heads=4, key_dim=8, 
                                ff_dim_1=shape[1], ff_dim_2=shape[1], num_blocks=2)
    encoder = Dense(4, activation='gelu')(encoder)

    # Decoder
    decoder = Dense(shape[1], activation='gelu')(encoder)
    decoder = transformer_stack(decoder, num_heads=4, key_dim=8,
                                ff_dim_1=shape[1], ff_dim_2=shape[1], num_blocks=2)
    
    # Simple Attention pooling
    pooling_layer = AttentionPooling()(decoder)

    # Output
    flat_layer = Concatenate()([pooling_layer, meta_input_layer])
    dense_layer_1 = Dense(units=128, activation='gelu')(flat_layer)
    dense_layer_2 = Dense(units=64, activation='gelu')(dense_layer_1)
    output_layer = last_layer(label)(dense_layer_2)
    model = Model(inputs=[seq_input_layer, meta_input_layer], outputs=output_layer)

    return model


CUSTOM_OBJECTS = {
    'custom_categorical_crossentropy': custom_categorical_crossentropy,
    'Expert': Expert,
    'MoETopKLayer': MoETopKLayer,
    'AttentionPooling': AttentionPooling
}
