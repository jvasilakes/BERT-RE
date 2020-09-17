import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .tokenization.tokenization_bert import BertTokenizer


class InputProcessor(object):

    def __init__(self, vocab_file, do_lower_case, max_seq_length,
                 name="input_processor"):
        self.tokenizer = self.get_tokenizer(vocab_file, do_lower_case)
        self.max_seq_length = max_seq_length
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_tokenizer(self, vocab_file, do_lower_case):
        return BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def process(self, documents):
        raise NotImplementedError

    def cut_tokens(self, tokens):
        if len(tokens) <= self.max_seq_length:
            return tokens
        return tokens[:self.max_seq_length - 1] + ["[SEP]"]

    def _get_token_ids(self, tokens):
        """Token IDs for embedding lookup"""
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        pad_amount = self.max_seq_length - len(tokens)
        input_ids = token_ids + [0] * pad_amount
        return np.array(input_ids)

    def _get_token_mask(self, tokens):
        """Mask for padding"""
        pad_amount = self.max_seq_length - len(tokens)
        masks = [1] * len(tokens) + [0] * pad_amount
        return np.array(masks)

    def _get_token_type_ids(self, tokens):
        """AKA segments. 0 for tokens before first [SEP],
        1 for tokens after first [SEP].
        """
        tok_types = []
        current_tok_type = 0
        for token in tokens:
            tok_types.append(current_tok_type)
            if token == "[SEP]":
                current_tok_type += 1
        pad_amount = self.max_seq_length - len(tokens)
        all_tok_types = tok_types + [current_tok_type] * pad_amount
        return np.array(all_tok_types)


class StandardProcessor(InputProcessor):
    """
    Input:
      [CLS] ... [SEP]

    Returns:
      [word_ids, token_mask, token_type_ids]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "standard_processor"
        self.entity_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
        self.rm_entity_tokens_in_input = True

    def process(self, documents):
        tokenized = []
        all_token_ids = []
        all_token_masks = []
        all_token_type_ids = []
        for doc in tqdm(documents):
            # Remove entity tokens, if they are present.
            if self.rm_entity_tokens_in_input is True:
                for etok in self.entity_tokens:
                    doc = doc.replace(etok, '')
            tokens = self.tokenizer.tokenize(doc)
            tokens = self.cut_tokens(tokens)
            tokenized.append(tokens)
            token_ids = self._get_token_ids(tokens)
            token_mask = self._get_token_mask(tokens)
            token_type_ids = self._get_token_type_ids(tokens)
            all_token_ids.append(token_ids)
            all_token_masks.append(token_mask)
            all_token_type_ids.append(token_type_ids)
        bert_inputs = [np.array(all_token_ids),
                       np.array(all_token_masks),
                       np.array(all_token_type_ids)]
        return (bert_inputs, tokenized)

    def get_input_placeholders(self):
        inshape = (self.max_seq_length,)
        word_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="word_ids")
        token_mask = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="token_mask")
        token_type_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="token_type_ids")

        return [word_ids, token_mask, token_type_ids]


class EntityProcessor(StandardProcessor):
    """
    Input:
      [CLS] [E1]first entity[/E1] ... [E2]second entity[/E2] ... [SEP]

    Returns:
      [word_ids, token_mask, token_type_ids]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "entity_processor"
        self.rm_entity_tokens_in_input = False
        added = self.tokenizer.add_tokens(self.entity_tokens)
        if added != len(self.entity_tokens):
            raise ValueError("Failed to add some entity tokens to the tokenizer.")  # noqa


class EntityStartProcessor(EntityProcessor):
    """
    Input:
      [CLS] [E1]first entity[/E1] ... [E2]second entity[/E2] ... [SEP]

    Returns:
      [word_ids, token_mask, token_type_ids, e1_start_mask, e2_start_mask]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "entity_start_processor"
        self.e1_start_token = "[E1]"
        self.e2_start_token = "[E2]"

    def process(self, documents):
        tokenized = []
        all_token_ids = []
        all_token_masks = []
        all_token_type_ids = []
        all_e1_start_masks = []
        all_e2_start_masks = []
        for doc in tqdm(documents):
            tokens = self.tokenizer.tokenize(doc)
            tokens = self.cut_tokens(tokens)
            tokenized.append(tokens)
            token_ids = self._get_token_ids(tokens)
            token_mask = self._get_token_mask(tokens)
            token_type_ids = self._get_token_type_ids(tokens)
            e1_start_mask, e2_start_mask = self._get_entity_start_masks(tokens)
            all_token_ids.append(token_ids)
            all_token_masks.append(token_mask)
            all_token_type_ids.append(token_type_ids)
            all_e1_start_masks.append(e1_start_mask)
            all_e2_start_masks.append(e2_start_mask)
        bert_inputs = [np.array(all_token_ids),
                       np.array(all_token_masks),
                       np.array(all_token_type_ids),
                       np.array(all_e1_start_masks),
                       np.array(all_e2_start_masks)]
        return (bert_inputs, tokenized)

    def get_input_placeholders(self):
        inshape = (self.max_seq_length,)
        word_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="word_ids")
        token_mask = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="token_mask")
        token_type_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="token_type_ids")
        e1_start_mask = tf.keras.layers.Input(
                shape=inshape, dtype=tf.bool, name="e1_start_mask")
        e2_start_mask = tf.keras.layers.Input(
                shape=inshape, dtype=tf.bool, name="e2_start_mask")
        return [word_ids, token_mask, token_type_ids,
                e1_start_mask, e2_start_mask]

    def _get_entity_start_masks(self, tokens):
        """
        Returns a boolean mask corresponding to the indices of [E1] and [E2]
        in tokens.
        """
        e1_mask = np.zeros(shape=(self.max_seq_length,), dtype=bool)
        e1_mask[np.argwhere(np.array(tokens) == self.e1_start_token)] = True
        if e1_mask.sum() == 0:
            raise ValueError(f"Can't find token [E1] in input: {tokens}")
        if e1_mask.sum() > 1:
            raise ValueError(f"Too many [E1] tokens in input: {tokens}")

        e2_mask = np.zeros(shape=(self.max_seq_length,), dtype=bool)
        e2_mask[np.argwhere(np.array(tokens) == self.e2_start_token)] = True
        if e2_mask.sum() == 0:
            raise ValueError(f"Can't find token [E2] in input: {tokens}")
        if e2_mask.sum() > 1:
            raise ValueError(f"Too many [E2] tokens in input: {tokens}")
        return e1_mask, e2_mask


class EntityMentionProcessor(StandardProcessor):
    """
    Input:
      [CLS] [E1]first entity[/E1] ... [E2]second entity[/E2] ... [SEP]

    Returns:
      [word_ids, token_mask, token_type_ids, e1_mention_mask, e2_mention_mask]

    keep_entity_markers: default True: If False, remove entity
                         markers ([E1], [E2], etc.) from the input,
                         to make it equivalent to Standard input.
    """
    def __init__(self, keep_entity_markers=True, *args, **kwargs):
        super().__init__(*args, **kwargs, name="EntityMentionProcessor")
        self.keep_entity_markers = keep_entity_markers
        added = self.tokenizer.add_tokens(self.entity_tokens)
        if added != len(self.entity_tokens):
            msg = "Failed to add some entity tokens to the tokenizer."
            raise ValueError(msg)

    @property
    def vocab_size(self):
        n = super().vocab_size
        if self.keep_entity_markers is False:
            n -= len(self.entity_tokens)
        return n

    def process(self, documents):
        tokenized = []
        all_token_ids = []
        all_token_masks = []
        all_token_type_ids = []
        all_e1_mention_masks = []
        all_e2_mention_masks = []
        for (i, doc) in tqdm(enumerate(documents)):
            tokens = self.tokenizer.tokenize(doc)
            tokens = self.cut_tokens(tokens)

            e1_mask, e2_mask = self._get_entity_mention_masks(tokens)

            if self.keep_entity_markers is False:
                drop_idxs = [i for (i, t) in enumerate(tokens)
                             if t in self.entity_tokens]
                tokens = np.delete(tokens, drop_idxs)
                backfill = np.zeros(shape=(len(drop_idxs),), dtype=bool)
                e1_mask = np.delete(e1_mask, drop_idxs)
                e1_mask = np.concatenate([e1_mask, backfill])
                e2_mask = np.delete(e2_mask, drop_idxs)
                e2_mask = np.concatenate([e2_mask, backfill])

            token_ids = self._get_token_ids(tokens)
            token_mask = self._get_token_mask(tokens)
            token_type_ids = self._get_token_type_ids(tokens)

            tokenized.append(tokens)
            all_token_ids.append(token_ids)
            all_token_masks.append(token_mask)
            all_token_type_ids.append(token_type_ids)
            all_e1_mention_masks.append(e1_mask)
            all_e2_mention_masks.append(e2_mask)
        bert_inputs = [np.array(all_token_ids),
                       np.array(all_token_masks),
                       np.array(all_token_type_ids),
                       np.array(all_e1_mention_masks),
                       np.array(all_e2_mention_masks)]
        return (bert_inputs, tokenized)

    def get_input_placeholders(self):
        inshape = (self.max_seq_length,)
        word_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="word_ids")
        token_mask = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="token_mask")
        token_type_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="token_type_ids")
        e1_mask = tf.keras.layers.Input(
                shape=inshape, dtype=tf.bool, name="e1_mask")
        e2_mask = tf.keras.layers.Input(
                shape=inshape, dtype=tf.bool, name="e2_mask")
        return [word_ids, token_mask, token_type_ids, e1_mask, e2_mask]

    def _get_entity_mention_masks(self, tokens):
        e1_mask = np.zeros(shape=(self.max_seq_length,), dtype=bool)
        e2_mask = np.zeros(shape=(self.max_seq_length,), dtype=bool)
        in_e1 = False
        in_e2 = False
        for (i, t) in enumerate(tokens):
            if t in self.entity_tokens:
                if t in ["[E1]", "[/E1]"]:
                    in_e1 = not in_e1
                elif t in ["[E2]", "[/E2]"]:
                    in_e2 = not in_e2
            else:
                if in_e1 is True:
                    e1_mask[i] = True
                elif in_e2 is True:
                    e2_mask[i] = True
        return e1_mask, e2_mask


class PositionalEmbeddingProcessor(InputProcessor):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError
