import os
import argparse
import bert
import numpy as np
import tensorflow as tf

import input_processors
import classification_heads
import bert_models as models


tf.random.set_seed(0)

docs = {"standard": ["[CLS] He went to the store. [SEP]"],
        "entity": ["[CLS] [E1]He[\\E1] went to the [E2]store[\\E2]. [SEP]"],
        }

bert_model_dir = "/media/jav/JakeDrive/BERT_models/uncased_L-12_H-768_A-12/"
do_lower_case = True
n_classes = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=int, nargs="*")
    return parser.parse_args()


# Model 1
# Input: Standard (no entity markers)
# Head: Dense layer from [CLS] embedding
# Corresponds to Matching the Blanks Figure 3a.
def run_standard_cls():
    vocab_file = os.path.join(bert_model_dir, "vocab.txt")
    processor = input_processors.StandardProcessor(
            vocab_file=vocab_file, do_lower_case=do_lower_case,
            max_seq_length=128)

    bert_params = bert.params_from_pretrained_ckpt(bert_model_dir)
    # Make sure BERT takes any additional tokens into account
    bert_params["vocab_size"] = processor.vocab_size
    head = classification_heads.CLSHead(n_classes=1, out_activation="sigmoid",
                                        bias_initializer="zeros",
                                        dropout_rate=0.0)

    inputs = processor.get_input_placeholders()
    model = models.get_bert_classifier(inputs, bert_params, head)
    opt = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss_fn = "binary_crossentropy"
    model.compile(optimizer=opt, loss=loss_fn)

    word_ids, token_type_ids = processor.process(docs["standard"])
    loss = model.evaluate([word_ids, token_type_ids], np.array([1]))
    print()
    print(model.input_shape)
    print()

    print()
    print("=== Standard CLS ===")
    print(docs["standard"])
    print(processor._all_raw_tokens)
    print(word_ids)
    print(token_type_ids)
    print()
    model.summary()
    print(f"Loss: {loss}")


# Model 2
# Input: Entity
# Head: Dense layer from [CLS] embedding
# Corresponds to Matching the Blanks Figure 3d.
def run_entity_cls():
    vocab_file = os.path.join(bert_model_dir, "vocab.txt")
    processor = input_processors.EntityProcessor(
            vocab_file=vocab_file, do_lower_case=do_lower_case,
            max_seq_length=128)

    bert_params = bert.params_from_pretrained_ckpt(bert_model_dir)
    # Make sure BERT takes the additional entity tokens into account
    bert_params["vocab_size"] = processor.vocab_size
    head = classification_heads.CLSHead(n_classes=1, out_activation="sigmoid",
                                        bias_initializer="zeros",
                                        dropout_rate=0.0)
    inputs = processor.get_input_placeholders()
    model = models.get_bert_classifier(inputs, bert_params, head)
    opt = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss_fn = "binary_crossentropy"
    model.compile(optimizer=opt, loss=loss_fn)

    word_ids, token_type_ids = processor.process(docs["entity"])
    loss = model.evaluate([word_ids, token_type_ids], np.array([1]))

    print()
    print("=== Entity CLS ===")
    print(docs["entity"])
    print(processor._all_raw_tokens)
    print(word_ids)
    print(token_type_ids)
    print()
    model.summary()
    print(f"Loss: {loss}")


# Model 3
# Input: Entity
# Head: Concatenate entity start tokens -> Dense layer
# Corresponds to Matching the Blanks Figure 3f.
def run_entity_start():
    vocab_file = os.path.join(bert_model_dir, "vocab.txt")
    processor = input_processors.EntityStartProcessor(
            vocab_file=vocab_file, do_lower_case=do_lower_case,
            max_seq_length=128)

    bert_params = bert.params_from_pretrained_ckpt(bert_model_dir)
    # Make sure BERT takes the additional entity tokens into account
    bert_params["vocab_size"] = processor.vocab_size
    head = classification_heads.EntityStartHead(
            n_classes=1, out_activation="sigmoid",
            bias_initializer="zeros", dropout_rate=0.0)

    inputs = processor.get_input_placeholders()
    model = models.get_bert_classifier(inputs, bert_params, head)
    opt = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss_fn = "binary_crossentropy"
    model.compile(optimizer=opt, loss=loss_fn)

    processed_inputs = processor.process(docs["entity"])  # noqa
    word_ids, token_type_ids, e1_masks, e2_masks = processed_inputs
    inputs = [word_ids, token_type_ids, e1_masks, e2_masks]
    loss = model.evaluate(inputs, np.array([1]))

    print()
    print("=== Entity Start ===")
    print(docs["entity"])
    print(processor._all_raw_tokens)
    print(word_ids)
    print(token_type_ids)
    print(e1_masks)
    print(e2_masks)
    print()
    model.summary()
    print(f"Loss: {loss}")


if __name__ == "__main__":
    model_map = {1: run_standard_cls,
                 2: run_entity_cls,
                 3: run_entity_start,
                 }

    args = parse_args()
    model_nums = args.models
    if args.models == []:
        model_nums = model_map.keys()

    for model_num in model_nums:
        model_map[model_num]()
