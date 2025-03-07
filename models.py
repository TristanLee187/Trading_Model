# Define the model architectures

from common import *
import tensorflow as tf
from keras import Model
from keras.api.layers import (
    LSTM, Dense, Input, MultiHeadAttention,
    Add, LayerNormalization, Flatten, Layer, Concatenate)


REG_FACTOR = 1e-4


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
    weights_tensor = tf.reduce_sum(tf.expand_dims(
        weights, axis=0) * tf.expand_dims(y_true, axis=-1), axis=-2)
    weighted_loss = ce_loss * tf.reduce_sum(weights_tensor, axis=-1)
    return weighted_loss


# Single expert
class Expert(Layer):
    def __init__(self, units1, units2):
        super(Expert, self).__init__()
        self.units1 = units1
        self.units2 = units2
        self.dense1 = None
        self.dense2 = None

    def build(self, input_shape):
        self.dense1 = Dense(self.units1, activation='gelu')
        self.dense2 = Dense(self.units2, activation='gelu')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    

# Mixture of experts top k layer
class MoETopKLayer(Layer):
    def __init__(self, num_experts, expert_units_1, expert_units_2, top_k, **kwargs):
        super(MoETopKLayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_units_1 = expert_units_1
        self.expert_units_2 = expert_units_2
        self.top_k = top_k

        self.experts = None
        # Pool based on "attention"
        self.attention_weights = None
        self.gating_network = None

    def build(self, input_shape):
        self.experts = [Expert(self.expert_units_1, self.expert_units_2) for _ in range(self.num_experts)]
        self.attention_weights = Dense(1)
        self.gating_network = Dense(self.num_experts, activation='softmax')

    def call(self, inputs):
        # Attention pooling
        attn_scores = tf.nn.softmax(self.attention_weights(inputs), axis=1)
        attn_inputs = tf.reduce_sum(inputs * attn_scores, axis=1)
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
    Define a transformer, LSTM, and MoE based architecture.

    Args:
        shape (tuple[int, int]): shape of each input sequence.
        meta_dim (int): Dimension of the metadata vector.
        label (str):  The label type to train on.

    Returns:
        keras.Model: Model with a transformer-LSTM-MoE architecture.
    """
    # Transformer block with LSTM position encoding and MoE
    def transformer_block(x, num_heads, key_dim, ff_dim_1, ff_dim_2):
        x = Add()([x, LSTM(units=x.shape[2], return_sequences=True)(x)])
        attn_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.2)(x, x)
        x = Add()([x, attn_layer])
        x = LayerNormalization(epsilon=1e-8)(x)
        moe = MoETopKLayer(num_experts=5, expert_units_1=ff_dim_1, expert_units_2=ff_dim_2, top_k=2)(x)
        x = Add()([x, moe])
        x = LayerNormalization(epsilon=1e-8)(x)
        return x

    # Stack of Transformer blocks
    def transformer_stack(x, num_heads, key_dim, ff_dim_1, ff_dim_2, num_blocks):
        for _ in range(num_blocks):
            x = transformer_block(x, num_heads=num_heads, key_dim=key_dim, 
                                  ff_dim_1=ff_dim_1, ff_dim_2=ff_dim_2)
        return x

    # Define the Transformer model
    seq_input_layer = Input(shape=shape)
    meta_input_layer = Input(shape=meta_dim)

    # Transformers!
    transformer_blocks = transformer_stack(seq_input_layer, num_heads=4, key_dim=8, 
            ff_dim_1=shape[1], ff_dim_2=shape[1], num_blocks=4)
    
    # Pool by flattening
    pooling_layer = Flatten()(transformer_blocks)

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
    'MoETopKLayer': MoETopKLayer
}