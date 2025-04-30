# Define the model architectures

from common import *
import tensorflow as tf
from keras import Model
from keras.api.layers import (
    LSTM, Dense, Input, MultiHeadAttention,
    Add, LayerNormalization, Layer, Concatenate, Permute
)
from keras.api.regularizers import l2


REG_FACTOR = 1e-2
DROPOUT_FACTOR = 0.2


def custom_categorical_crossentropy(y_true, y_pred):
    """
    Custom categorical-crossentropy loss function on 3 classes that uses weights to
    penalize different classificiations differently.

    Args:
        y_true (np.array): Ground truth one-hot encoding vector.
        y_pred (np.array): Prediction one-hot encoding vector.

    Returns:
        function with arguments (np.array, np.array): Function that computes the loss w.r.t.
            ground truth and prediction inputs.
    """
    # Version 3!
    # weights[i][j]: penalty for if the ground truth was i but the predicted was j.
    weights = tf.constant([
        [0.0, 5.0, 15.0],
        [2.0, 0.0, 2.0],
        [10.0, 3.0, 0.0],
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
    

# Transformer block with LSTM positional encoding and MoE
class TransformerBlock(Layer):
    def __init__(self, num_heads, key_dim, ff_dim_1, ff_dim_2, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim_1 = ff_dim_1
        self.ff_dim_2 = ff_dim_2

        self.position_encoder = None
        self.attn_layer = None
        self.mixture_of_experts = None

        self.add_1 = None
        self.add_2 = None
        self.add_3 = None
        self.layernorm_1 = None
        self.layernorm_2 = None

    def build(self, input_shape):
        # LSTM position encoding
        self.position_encoder = LSTM(units=input_shape[2], return_sequences=True,
                                     kernel_regularizer=l2(REG_FACTOR),
                                     recurrent_regularizer=l2(REG_FACTOR), 
                                     bias_regularizer=l2(REG_FACTOR))
        # Multi-head attention
        self.attn_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, 
                                             dropout=DROPOUT_FACTOR, 
                                             kernel_regularizer=l2(REG_FACTOR), 
                                             bias_regularizer=l2(REG_FACTOR))
        # Mixture of experts
        self.mixture_of_experts = MoETopKLayer(num_experts=5, dim_1=self.ff_dim_1, dim_2=self.ff_dim_2, top_k=2)

        # Misc
        self.add_1 = Add()
        self.add_2 = Add()
        self.add_3 = Add()
        self.layernorm_1 = LayerNormalization(epsilon=1-8)
        self.layernorm_2 = LayerNormalization(epsilon=1-8)
    
    def call(self, inputs):
        # Encode positions
        positions = self.position_encoder(inputs)
        x = self.add_1([inputs, positions])

        # Attention!
        attention = self.attn_layer(x, x)
        x = self.add_2([x, attention])
        x = self.layernorm_1(x)

        # Mixture of experts
        moe = self.mixture_of_experts(x)
        x = self.add_3([x, moe])
        x = self.layernorm_2(x)
        
        return x


# Functional version of the above class to generate pictures of the transformer block
def transformer_block(shape, num_heads, key_dim, ff_dim_1, ff_dim_2):
    input_layer = Input(shape=shape)
    x = input_layer
    # Positional encoding
    position_encoder = LSTM(units=x.shape[2], return_sequences=True, 
                            kernel_regularizer=l2(REG_FACTOR), recurrent_regularizer=l2(REG_FACTOR), bias_regularizer=l2(REG_FACTOR))(x)
    x = Add()([x, position_encoder])
    
    # Attention!
    attn_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=DROPOUT_FACTOR, 
                                    kernel_regularizer=l2(REG_FACTOR), bias_regularizer=l2(REG_FACTOR))(x, x)
    x = Add()([x, attn_layer])
    x = LayerNormalization(epsilon=1e-8)(x)
    
    # MoE
    moe = MoETopKLayer(num_experts=5, dim_1=ff_dim_1, dim_2=ff_dim_2, top_k=2)(x)
    x = Add()([x, moe])
    x = LayerNormalization(epsilon=1e-8)(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


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
    # Stack of Transformer blocks
    def transformer_stack(x, num_heads, key_dim, ff_dim_1, ff_dim_2, num_blocks):
        for _ in range(num_blocks):
            x = TransformerBlock(num_heads=num_heads, key_dim=key_dim, 
                                 ff_dim_1=ff_dim_1, ff_dim_2=ff_dim_2)(x)
        return x

    # Define the inputs
    seq_input_layer = Input(shape=shape)
    meta_input_layer = Input(shape=meta_dim)

    # Transformer blocks
    temporal_stack = transformer_stack(seq_input_layer, num_heads=2, key_dim=8,
                              ff_dim_1=shape[1], ff_dim_2=shape[1], num_blocks=4)
    feature_input_layer = Permute((2, 1))(seq_input_layer)
    feature_stack = transformer_stack(feature_input_layer, num_heads=2, key_dim=8,
                              ff_dim_1=shape[0], ff_dim_2=shape[0], num_blocks=4)
    # Transpose back
    feature_stack = Permute((2, 1))(feature_stack)
    stack = Add()([temporal_stack, feature_stack])
 
    # Aggregation
    pooling_layer = LSTM(units=shape[1], kernel_regularizer=l2(REG_FACTOR), recurrent_regularizer=l2(REG_FACTOR), bias_regularizer=l2(REG_FACTOR))(stack)
    
    # Output
    flat_layer = Concatenate()([pooling_layer, meta_input_layer])
    dense_layer_1 = Dense(units=64, activation='gelu', kernel_regularizer=l2(REG_FACTOR), bias_regularizer=l2(REG_FACTOR))(flat_layer)
    dense_layer_2 = Dense(units=32, activation='gelu', kernel_regularizer=l2(REG_FACTOR), bias_regularizer=l2(REG_FACTOR))(dense_layer_1)
    dense_layer_3 = Dense(units=32, activation='gelu', kernel_regularizer=l2(REG_FACTOR), bias_regularizer=l2(REG_FACTOR))(dense_layer_2)
    output_layer = last_layer(label)(dense_layer_3)
    model = Model(inputs=[seq_input_layer, meta_input_layer], outputs=output_layer)

    return model


CUSTOM_OBJECTS = {
    'custom_categorical_crossentropy': custom_categorical_crossentropy,
    'Expert': Expert,
    'MoETopKLayer': MoETopKLayer,
    'AttentionPooling': AttentionPooling,
    'TransformerBlock': TransformerBlock
}
