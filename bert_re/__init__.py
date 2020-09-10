import bert
import tensorflow as tf


def get_bert_classifier(inputs, bert_params, classification_head):
    if len(inputs) < 2:
        raise ValueError("Expected at inputs of len >= 2")

    print("Initializing BERT layer with params:")
    for (k, v) in bert_params.items():
        print(f"  {k}: {v}")

    # inputs[:2] are always [word_ids, token_type_ids]
    bert_inputs = inputs[:2]
    bert_output = bert.BertModelLayer.from_params(
            bert_params, name="bert")(bert_inputs)

    # inputs[2:] are any arguments to pass to the prediction layer.
    #   E.g. entity masks. Can be emtpy.
    args = [bert_output] + inputs[2:]
    predictions = classification_head(*args)

    model = tf.keras.Model(inputs=inputs, outputs=predictions,
                           name=f"bert_{classification_head.name}")
    return model
