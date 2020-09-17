import bert
import tensorflow as tf

from . import input_processors  # noqa : F401 '.input_processors' imported but unused
from . import heads  # noqa : F401 '.heads' imported but unused


def get_bert_classifier(inputs, bert_params, model_ckpt, classification_head,
                        logging_fn=print):
    if len(inputs) < 3:
        raise ValueError("BERT inputs must be of length 3")

    params_str = "Initializing BERT layer with params:"
    for (k, v) in bert_params.items():
        params_str += f"\n  {k}: {v}"
    logging_fn(params_str)

    # inputs[:3] are always [word_ids, token_mask, token_type_ids]
    bert_inputs = [inputs[0], inputs[2]]
    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")

    seq_output = bert_layer(bert_inputs)

    # inputs[3:] are any arguments to pass to the prediction layer.
    #   E.g. entity masks. Can be emtpy.
    args = [seq_output] + inputs[3:]
    predictions = classification_head(*args)

    model = tf.keras.Model(inputs=inputs, outputs=predictions,
                           name=f"bert_{classification_head.name}")
    input_shapes = [inp.shape for inp in inputs]
    model.build(input_shape=input_shapes)
    skipped = bert.load_bert_weights(bert_layer, model_ckpt)

    # Probably because we extended the vocabulary.
    if len(skipped) == 1:
        skipped_param, ckpt_value = skipped[0]
        emb_name = "bert/embeddings/word_embeddings/embeddings:0"
        if skipped_param.name == emb_name:
            old_vocab_size = ckpt_value.shape[0]
            new_vocab_size = bert_params["vocab_size"]
            logging_fn(f"Extending pretrained BERT embeddings: {old_vocab_size} -> {new_vocab_size}")  # noqa
            extended_embeddings = extend_ckpt_embeddings(
                    ckpt_value, new_vocab_size)
            tf.keras.backend.set_value(skipped_param, extended_embeddings)
        else:
            raise ValueError(f"Skipped loading params: {skipped}")
    elif len(skipped) > 1:
        raise ValueError(f"Skipped loading params: {skipped}")
    else:
        pass

    return model


def extend_ckpt_embeddings(ckpt_value, new_vocab_size):
    """
    Append vectors to ckpt_value up to new_vocab_size
    """
    num_to_add = new_vocab_size - ckpt_value.shape[0]
    # For reproducibility
    initializer = tf.keras.initializers.GlorotUniform(seed=0)
    new_emb_shape = (num_to_add, ckpt_value.shape[1])
    new_embeddings = initializer(shape=new_emb_shape)
    extended_embeddings = tf.concat([ckpt_value, new_embeddings], axis=0)
    assert extended_embeddings.shape[0] == new_vocab_size
    assert extended_embeddings.shape[1] == ckpt_value.shape[1]
    return extended_embeddings
