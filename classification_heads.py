import tensorflow as tf


class MaskLayer(tf.keras.layers.Layer):
    """
    Used with EntityStartHead to implement the "Entity start state"
    head from the Matching the Blanks paper.
    https://www.aclweb.org/anthology/P19-1279.pdf

    Optionally applies a boolean mask to the time
    dimension of the NxK input tensor and returns
    an MxK tensor where N := the input sequence length,
    K := the embedding dimension (e.g. 768 for BERT-base),
    and M := the magnitude of the mask vector.

    input = [I, ate, dinner]
    mask = [True, False, True]
    onehot_input = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
    masked = MaskLayer()(onehot_input, mask)
    print(mask)

    [[1, 0, 1],
     [0, 0, 1]]# Elementwise sum of 'I' and 'dinner'

    If mask has only one True position, e.g. mask=[True, False, False],
    then M = 1 and the output is squeezed into a K-dim vector.

    masked = MaskLayer()(onehot_input, [True, False, False])
    print(mask)

    [1, 0, 0]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            masked = inputs
        else:
            masked = tf.ragged.boolean_mask(inputs, mask)
        return tf.squeeze(masked, axis=0)


class CLSHead(tf.keras.layers.Layer):
    """
    Inputs: [word_ids, token_type_ids]
    """
    def __init__(self, n_classes, out_activation, bias_initializer,
                 dropout_rate, *args, **kwargs):
        name = "cls_head"
        super().__init__(name=name, *args, **kwargs)
        self.n_classes = n_classes
        self.out_activation = out_activation
        self.bias_initializer = bias_initializer
        self.dropout_rate = dropout_rate

        # Layers
        self.drop_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.pred_layer = tf.keras.layers.Dense(
                self.n_classes, activation=self.out_activation,
                bias_initializer=self.bias_initializer,
                name="prediction")
        self.layers = [self.drop_layer, self.pred_layer]

    def call(self, bert_output):
        cls = bert_output[:, 0, :]  # [CLS] token
        dropped = self.drop_layer(cls)
        return self.pred_layer(dropped)


class EntityStartHead(tf.keras.layers.Layer):
    """
    Inputs: [
             [word_ids, token_type_ids],
             entity1_start_mask, entity2_start_mask
            ]
    """

    def __init__(self, n_classes, out_activation, bias_initializer,
                 dropout_rate, *args, **kwargs):
        name = "entity_start_head"
        super().__init__(name=name, *args, **kwargs)
        self.n_classes = n_classes
        self.out_activation = out_activation
        self.bias_initializer = bias_initializer
        self.dropout_rate = dropout_rate

        # Layers
        self.e1_mask_layer = MaskLayer(name="masked_e1")
        self.e2_mask_layer = MaskLayer(name="masked_e2")
        self.concat = tf.keras.layers.Concatenate()
        self.drop_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.pred_layer = tf.keras.layers.Dense(
                self.n_classes, activation=self.out_activation,
                bias_initializer=self.bias_initializer,
                name="prediction")
        self.layers = [self.e1_mask_layer, self.e2_mask_layer,
                       self.concat, self.drop_layer, self.pred_layer]

    def call(self, bert_output, e1_mask, e2_mask):
        e1_mask = tf.cast(e1_mask, tf.bool)
        e2_mask = tf.cast(e2_mask, tf.bool)
        e1 = self.e1_mask_layer(bert_output, e1_mask)
        e2 = self.e2_mask_layer(bert_output, e2_mask)
        dense_input = self.concat([e1, e2])
        dropped = self.drop_layer(dense_input)
        return self.pred_layer(dropped)
