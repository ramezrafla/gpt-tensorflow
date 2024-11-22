import tensorflow as tf
import keras
from keras import layers
import numpy as np

class GPT(keras.Model):
    def __init__(
        self, 
        num_blocks, 
        token_embed_dim, 
        num_attention_heads, 
        attention_dim, 
        feed_forward_dim, 
        context_size, 
        activation,
        dropout,
        vocab_size
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_embed_dim = token_embed_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dim = attention_dim
        self.feed_forward_dim = feed_forward_dim
        self.context_size = context_size
        self.activation = activation
        self.attention_mask = np.tril(np.ones((context_size, context_size)), 0)
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.metrics = [keras.metrics.Accuracy()]
        
        self.token_embed = layers.Embedding(
            input_dim = self.vocab_size, 
            output_dim = self.token_embed_dim
        )
        self.positional_embed = layers.Embedding(
            input_dim = self.context_size, 
            output_dim = self.token_embed_dim
        )
        self.attentions = [
            layers.MultiHeadAttention(
                num_heads = self.num_attention_heads,
                key_dim = self.attention_dim, 
                value_dim = self.attention_dim,
                dropout = self.dropout
            )    
            for _ in range(self.num_blocks)
        ]
        self.feed_forwards_1 = [
            layers.Dense(
                units = self.feed_forward_dim, 
                activation = self.activation
            ) 
            for _ in range(self.num_blocks)
        ]
        self.feed_forwards_2 = [
            layers.Dense(
                units = self.token_embed_dim, 
                activation = self.activation
            ) 
            for _ in range(self.num_blocks)
        ]
        self.dropouts = [
            layers.Dropout(self.dropout)
            for _ in range(self.num_blocks)
        ]
        self.layer_normalizations = [
            layers.LayerNormalization() 
            for _ in range(self.num_blocks * 2)
        ]
        self.final_dense = layers.Dense(
            units = self.vocab_size, 
            activation = 'softmax'
        )
       
    def call(self, x, training = False):
        x = self.token_embed(x) + self.positional_embed(tf.range(self.context_size))
        for i in range(self.num_blocks):
            att = self.attentions[i](x, x, attention_mask = self.attention_mask, training = training)
            x = x + att
            x = self.layer_normalizations[i * 2](x)
            ff = self.feed_forwards_1[i](x)
            ff = self.feed_forwards_2[i](ff)
            ff = self.dropouts[i](ff, training = training)
            x = x + ff
            x = self.layer_normalizations[i * 2 + 1](x)
        x = self.final_dense(x)
        return x

    # https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training = True) # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return { m.name: m.result() for m in self.metrics }
