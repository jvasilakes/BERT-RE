import tensorflow as tf


class MaskLayer(tf.keras.layers.Layer):
    """
    Used with EntityStartHead to implement the "Entity start state"
    or with EntityMentionHead to implement the "Entity mention pooling"
    heads from the Matching the Blanks paper:
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

    [[1, 0, 0],
     [0, 0, 1]]# Elementwise sum of 'I' and 'dinner'

    If mask has only one True position, e.g. mask=[True, False, False],
    then M = 1 and the output is squeezed into a K-dim vector.

    masked = MaskLayer()(onehot_input, [True, False, False])
    print(mask)

    [1, 0, 0]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, mask=None, keep_dims=True):
        if mask is None:
            masked = inputs
        else:
            # We're not actually operating over ragged tensors.
            # We need to use ragged.boolean_mask to preserve the input shape.
            masked = tf.ragged.boolean_mask(inputs, mask).to_tensor()
        if keep_dims is False:
            masked = tf.squeeze(masked)
        return masked


class MaxLayer(tf.keras.layers.Layer):
    """
    Used with EntityMentionHead to implement the "Entity mention pooling"
    head from the Matching the Blanks paper:
    https://www.aclweb.org/anthology/P19-1279.pdf

    This is essentially just a wrapper around tf.math.reduce_max.
    We use reduce_max rather than MaxPool2D because we are always pooling
    over token representations, which differ in length depending on the
    entity mention.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        return tf.math.reduce_max(inputs, axis=1)


class CLSHead(tf.keras.layers.Layer):
    """
    Inputs: [word_ids, token_type_ids]
    Get inputs using input_processors.StandardProcessor
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
    Get inputs using input_processors.EntityStartProcessor
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
        self.e1_mask_layer = MaskLayer(name="e1_start")
        self.e2_mask_layer = MaskLayer(name="e2_start")
        self.concat = tf.keras.layers.Concatenate()
        self.drop_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.pred_layer = tf.keras.layers.Dense(
                self.n_classes, activation=self.out_activation,
                bias_initializer=self.bias_initializer,
                name="prediction")
        self.layers = [self.e1_mask_layer, self.e2_mask_layer,
                       self.concat, self.drop_layer, self.pred_layer]

    def call(self, bert_output, e1_mask, e2_mask):
        e1 = self.e1_mask_layer(bert_output, e1_mask)
        e2 = self.e2_mask_layer(bert_output, e2_mask)
        dense_input = self.concat([e1, e2])
        dropped = self.drop_layer(dense_input)
        return self.pred_layer(dropped)


class EntityMentionPoolHead(tf.keras.layers.Layer):
    """
    Inputs: [
             [word_ids, token_type_ids],
             [entity1_mask, entity2_mask],
            ]
    Get inputs using input_processors.EntityMentionProcessor
    """
    def __init__(self, n_classes, out_activation, bias_initializer,
                 dropout_rate, *args, **kwargs):
        name = "mention_pool_head"
        super().__init__(name=name, *args, **kwargs)
        self.n_classes = n_classes
        self.out_activation = out_activation
        self.bias_initializer = bias_initializer
        self.dropout_rate = dropout_rate

        # Layers
        self.e1_mask_layer = MaskLayer(name="e1_mention")
        self.e2_mask_layer = MaskLayer(name="e2_mention")
        self.e1_max = MaxLayer(name="e1_max")
        self.e2_max = MaxLayer(name="e2_max")
        self.concat = tf.keras.layers.Concatenate()
        self.drop_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.pred_layer = tf.keras.layers.Dense(
                units=self.n_classes, activation=self.out_activation,
                bias_initializer=self.bias_initializer,
                name="prediction")

    def call(self, bert_output, e1_mask, e2_mask):
        e1_mention = self.e1_mask_layer(bert_output, e1_mask, keep_dims=True)
        e2_mention = self.e2_mask_layer(bert_output, e2_mask, keep_dims=True)
        e1_max = self.e1_max(e1_mention)
        e2_max = self.e2_max(e2_mention)
        dense_input = self.concat([e1_max, e2_max])
        dropped = self.drop_layer(dense_input)
        return self.pred_layer(dropped)
