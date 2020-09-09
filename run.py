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
n_classes = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_dir", type=str, required=True,
                        help="""Directory containing BERT model checkpoint.
                                E.g. uncased_L-12_H-768_A-12/""")
    parser.add_argument("--do_lower_case", type=int, default=0, choices=[0, 1],
                        help="""0: cased BERT models.
                                1 (default): uncased BERT models.""")
    parser.add_argument("--models", type=int, nargs="*", default=[],
                        help="""Models to run. 1: standard CLS,
                                               2: entity CLS,
                                               3: entity start""")
    return parser.parse_args()


# Model 1
# Input: Standard (no entity markers)
# Head: Dense layer from [CLS] embedding
# Corresponds to Matching the Blanks Figure 3a.
def run_standard_cls(bert_model_dir, do_lower_case):
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
def run_entity_marker_cls(bert_model_dir, do_lower_case):
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
# Head: Concatenate entity start token reps -> dense
# Corresponds to Matching the Blanks Figure 3f.
def run_entity_marker_start(bert_model_dir, do_lower_case):
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

    processed_inputs = processor.process(docs["entity"])
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


# Model 4
# Input: Standard + entity mentions
# Head: Max pool over entity mention wordpiece reps -> concat -> dense
# Corresponds to Matching the Blanks Figure 3b.
def run_standard_mention_pool(bert_model_dir, do_lower_case):
    vocab_file = os.path.join(bert_model_dir, "vocab.txt")
    processor = input_processors.EntityMentionProcessor(
            vocab_file=vocab_file, do_lower_case=do_lower_case,
            max_seq_length=128)

    bert_params = bert.params_from_pretrained_ckpt(bert_model_dir)
    # Make sure BERT takes the additional entity tokens into account
    bert_params["vocab_size"] = processor.vocab_size
    head = classification_heads.EntityMentionPoolHead(
            n_classes=1, out_activation="sigmoid",
            bias_initializer="zeros", dropout_rate=0.0)

    inputs = processor.get_input_placeholders()
    model = models.get_bert_classifier(inputs, bert_params, head)
    opt = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss_fn = "binary_crossentropy"
    model.compile(optimizer=opt, loss=loss_fn)

    e1_mentions = ["He"]
    e2_mentions = ["store"]
    processed_inputs = processor.process(
            docs["standard"], e1_mentions, e2_mentions)
    word_ids, token_type_ids, e1_masks, e2_masks = processed_inputs
    inputs = [word_ids, token_type_ids, e1_masks, e2_masks]
    loss = model.evaluate(inputs, np.array([1]))

    print()
    print("=== Entity Mention Pool ===")
    print(docs["standard"])
    print(e1_mentions, e2_mentions)
    print(processor._all_raw_tokens)
    print(word_ids)
    print(token_type_ids)
    print(e1_masks)
    print(e2_masks)
    print()
    model.summary()
    print(f"Loss: {loss}")


# Model 5
# Input: Entity markers
# Head: Max pool over entity mention wordpiece reps -> concat -> dense
# Corresponds to Matching the Blanks Figure 3e.
def run_entity_marker_mention_pool(bert_model_dir, do_lower_case):
    vocab_file = os.path.join(bert_model_dir, "vocab.txt")
    processor = input_processors.EntityMentionProcessor(
            vocab_file=vocab_file, do_lower_case=do_lower_case,
            max_seq_length=128)

    bert_params = bert.params_from_pretrained_ckpt(bert_model_dir)
    # Make sure BERT takes the additional entity tokens into account
    bert_params["vocab_size"] = processor.vocab_size
    head = classification_heads.EntityMentionPoolHead(
            n_classes=1, out_activation="sigmoid",
            bias_initializer="zeros", dropout_rate=0.0)

    inputs = processor.get_input_placeholders()
    model = models.get_bert_classifier(inputs, bert_params, head)
    opt = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss_fn = "binary_crossentropy"
    model.compile(optimizer=opt, loss=loss_fn)

    e1_mentions = ["He"]
    e2_mentions = ["store"]
    processed_inputs = processor.process(
            docs["entity"], e1_mentions, e2_mentions)
    word_ids, token_type_ids, e1_masks, e2_masks = processed_inputs
    inputs = [word_ids, token_type_ids, e1_masks, e2_masks]
    loss = model.evaluate(inputs, np.array([1]))

    print()
    print("=== Entity Mention Pool ===")
    print(docs["entity"])
    print(e1_mentions, e2_mentions)
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
                 2: run_entity_marker_cls,
                 3: run_entity_marker_start,
                 4: run_standard_mention_pool,
                 5: run_entity_marker_mention_pool
                 }

    args = parse_args()
    model_nums = args.models
    if args.models == []:
        model_nums = model_map.keys()

    for model_num in model_nums:
        model_map[model_num](args.bert_model_dir, args.do_lower_case)
