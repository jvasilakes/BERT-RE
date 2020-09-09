import numpy as np
import tensorflow as tf

from tokenization import tokenization_bert


class InputProcessor(object):

    def __init__(self, vocab_file, do_lower_case, max_seq_length):
        self.tokenizer = self.get_tokenizer(vocab_file, do_lower_case)
        self.max_seq_length = max_seq_length

    def get_tokenizer(self, vocab_file, do_lower_case):
        return tokenization_bert.BertTokenizer(vocab_file=vocab_file,
                                               do_lower_case=True)

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
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        pad_amount = self.max_seq_length - len(tokens)
        input_ids = token_ids + [0] * pad_amount
        return np.array(input_ids)

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
      [word_ids, token_type_ids]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, documents):
        self._all_raw_tokens = []
        all_token_ids = []
        all_token_type_ids = []
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            tokens = self.cut_tokens(tokens)
            self._all_raw_tokens.append(tokens)
            token_ids = self._get_token_ids(tokens)
            token_type_ids = self._get_token_type_ids(tokens)
            all_token_ids.append(token_ids)
            all_token_type_ids.append(token_type_ids)
        return_val = [np.array(all_token_ids),
                      np.array(all_token_type_ids)]
        return return_val

    def get_input_placeholders(self):
        inshape = (self.max_seq_length,)
        word_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="word_ids")
        token_type_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="token_type_ids")

        return [word_ids, token_type_ids]


class EntityProcessor(StandardProcessor):
    """
    Input:
      [CLS] [E1]first entity[\\E1] ... [E2]second entity[\\E2] ... [SEP]

    Returns:
      [word_ids, token_type_ids]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_tokens = ["[E1]", "[\\E1]", "[E2]", "[\\E2]"]
        added = self.tokenizer.add_tokens(self.entity_tokens)
        if added != len(self.entity_tokens):
            raise ValueError("Failed to add some entity tokens to the tokenizer.")  # noqa


class EntityStartProcessor(EntityProcessor):
    """
    Input:
      [CLS] [E1]first entity[\\E1] ... [E2]second entity[\\E2] ... [SEP]

    Returns:
      [word_ids, token_type_ids, e1_start_mask, e2_start_mask]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e1_start_token = "[E1]"
        self.e2_start_token = "[E2]"

    def process(self, documents):
        self._all_raw_tokens = []
        all_token_ids = []
        all_token_type_ids = []
        all_e1_start_masks = []
        all_e2_start_masks = []
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            tokens = self.cut_tokens(tokens)
            self._all_raw_tokens.append(tokens)
            token_ids = self._get_token_ids(tokens)
            token_type_ids = self._get_token_type_ids(tokens)
            e1_start_mask, e2_start_mask = self._get_entity_start_masks(tokens)
            all_token_ids.append(token_ids)
            all_token_type_ids.append(token_type_ids)
            all_e1_start_masks.append(e1_start_mask)
            all_e2_start_masks.append(e2_start_mask)
        return_val = [np.array(all_token_ids),
                      np.array(all_token_type_ids),
                      np.array(all_e1_start_masks),
                      np.array(all_e2_start_masks)]
        return return_val

    def get_input_placeholders(self):
        inshape = (self.max_seq_length,)
        word_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="word_ids")
        token_type_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="token_type_ids")
        e1_start_mask = tf.keras.layers.Input(
                shape=inshape, dtype=tf.bool, name="e1_start_mask")
        e2_start_mask = tf.keras.layers.Input(
                shape=inshape, dtype=tf.bool, name="e2_start_mask")
        return [word_ids, token_type_ids, e1_start_mask, e2_start_mask]

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
      [CLS] entity1 ... entity2 ... [SEP]

    Returns:
      [word_ids, token_type_ids, entity1_mention_mask, entity2_mention_mask]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, documents, e1_mentions, e2_mentions):
        self._all_raw_tokens = []
        all_token_ids = []
        all_token_type_ids = []
        all_e1_mention_masks = []
        all_e2_mention_masks = []
        for (i, doc) in enumerate(documents):
            tokens = self.tokenizer.tokenize(doc)
            tokens = self.cut_tokens(tokens)
            self._all_raw_tokens.append(tokens)
            token_ids = self._get_token_ids(tokens)
            token_type_ids = self._get_token_type_ids(tokens)

            e1_tokens = self.tokenizer.tokenize(e1_mentions[i])
            e1_mask = self._get_subsequence_mask(tokens, e1_tokens)
            if e1_mask is None:
                msg = f"Bad e1 in input:\n {tokens}\n {e1_tokens}"
                raise ValueError(msg)
            e2_tokens = self.tokenizer.tokenize(e2_mentions[i])
            e2_mask = self._get_subsequence_mask(tokens, e2_tokens)
            if e2_mask is None:
                msg = f"Bad e2 in input:\n {tokens}\n {e2_tokens}"
                raise ValueError(msg)

            all_token_ids.append(token_ids)
            all_token_type_ids.append(token_type_ids)
            all_e1_mention_masks.append(e1_mask)
            all_e2_mention_masks.append(e2_mask)
        return_val = [np.array(all_token_ids),
                      np.array(all_token_type_ids),
                      np.array(all_e1_mention_masks),
                      np.array(all_e2_mention_masks)]
        return return_val

    def get_input_placeholders(self):
        inshape = (self.max_seq_length,)
        word_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="word_ids")
        token_type_ids = tf.keras.layers.Input(
                shape=inshape, dtype=tf.int32, name="token_type_ids")
        e1_mention_mask = tf.keras.layers.Input(
                shape=inshape, dtype=tf.bool, name="e1_mention_mask")
        e2_mention_mask = tf.keras.layers.Input(
                shape=inshape, dtype=tf.bool, name="e2_mention_mask")
        return [word_ids, token_type_ids, e1_mention_mask, e2_mention_mask]

    def _get_subsequence_mask(self, seq, subseq):
        mask = np.zeros(shape=(self.max_seq_length,), dtype=bool)
        j = 0
        for (i, t) in enumerate(seq):
            if j == len(subseq):
                return mask
            if t == subseq[j]:
                mask[i] = True
                j += 1
        return None


class PositionalEmbeddingProcessor(InputProcessor):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError
