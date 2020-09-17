import os
import bert
import logging
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from bert_re import datasets, get_bert_classifier, input_processors, \
                    heads, lr_schedulers


RANDOM_STATE = 0
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_dir", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True,
                        choices=["3a", "3b", "3c", "3d", "3e", "3f"],
                        help="""ID of model to run, matches those in Figure 3
                                of the Matching the Blanks paper.
                                    3a: Standard - CLS;
                                    3b: Standard - Mention Pool;
                                    3c: NOT IMPLEMENTED;
                                    3d: Entity Markers - CLS;
                                    3e: Entity Makers - Mention Pool;
                                    3f: Entity Markers - Entity Start
                              """)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--test", action="store_true", default=False)
    return parser.parse_args()


class LogLearningRateCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir):
        super().__init__()
        file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
        file_writer.set_as_default()
        self.batch = 0

    def on_train_batch_end(self, _, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            msg = "Optimizer must have a 'learning_rate' attribute."
            raise ValueError(msg)
        logs = logs or {}
        learning_rate_fn = tf.keras.backend.get_value(
                self.model.optimizer.learning_rate)
        learning_rate = learning_rate_fn(self.batch)
        tf.summary.scalar("learning_rate", data=learning_rate, step=self.batch)
        self.batch += 1


def main(args):
    if args.model_id == "3c":
        msg = "Model 3c: Positional Emb - Mention Pool, not yet implemented."
        raise NotImplementedError(msg)

    logger = get_logger(os.path.join(
        args.outdir, f"model_{args.model_id}.log"))
    logger.info(f"RUNNING MODEL: {args.model_id}")
    argstr = "Command line arguments:"
    for (k, v) in args.__dict__.items():
        argstr += f"\n  {k}: {v}"
    logger.info(argstr)

    # Build the input processor.
    vocab_file = os.path.join(args.bert_model_dir, "vocab.txt")
    processor = get_processor_from_model_id(
            args.model_id, vocab_file=vocab_file,
            do_lower_case=True, max_seq_length=128)

    # Training and validation data
    if args.test is True:
        print("TESTING MODEL OVERFITS SMALL TRAINING DATA SET")
        train_docs, train_labels = get_single_examples(args.train_file)
        val_docs = None
    else:
        all_train_docs, all_train_labels = read_dataset(args.train_file, n=-1)
        train_docs, val_docs, train_labels, val_labels = train_test_split(
                all_train_docs, all_train_labels, train_size=0.8,
                random_state=RANDOM_STATE, shuffle=True)

    train_inputs, train_tokens = processor.process(train_docs)
    assert len(train_docs) == len(train_inputs[0])
    assert len(train_docs) == len(train_labels)
    logger.info(f"Training on {len(train_labels)} examples")

    if val_docs is not None:
        val_inputs, val_tokens = processor.process(val_docs)
        assert len(val_docs) == len(val_inputs[0])
        assert len(val_docs) == len(val_labels)
        logger.info(f"Validating on {len(val_labels)} examples")
        val_data = (val_inputs, val_labels)
    else:
        val_data = None

    # Test data
    if args.test_file is not None:
        test_docs, test_labels = read_dataset(args.test_file, n=-1)
        test_inputs, _ = processor.process(test_docs)
        assert len(test_docs) == len(test_inputs[0])
        assert len(test_docs) == len(test_labels)
        val_data = (test_inputs, test_labels)
        logger.info(f"Evaluating on {len(test_labels)} examples")

    # Build the BERT model
    initial_biases = compute_initial_bias(train_labels)
    initial_biases[initial_biases == -np.inf] = 0.0
    bias_init = tf.keras.initializers.Constant(initial_biases)
    head = get_classification_head_from_model_id(
            args.model_id, n_classes=train_labels.shape[1],
            out_activation="softmax", bias_initializer=bias_init,
            dropout_rate=0.1)

    inputs = processor.get_input_placeholders()
    bert_params = bert.params_from_pretrained_ckpt(args.bert_model_dir)
    bert_params["vocab_size"] = processor.vocab_size
    logger.info(f"Input Processor: {processor.name}")
    logger.info(f"Classification Head: {head.name}")
    model_ckpt = os.path.join(args.bert_model_dir, "bert_model.ckpt")
    model = get_bert_classifier(inputs, bert_params, model_ckpt, head,
                                logging_fn=logger.info)

    # Define the learning rate schedule
    # Linear warm up with polynomial decay
    train_data_size = len(train_labels)
    steps_per_epoch = int(train_data_size / args.batch_size)
    num_train_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.epochs * train_data_size * 0.1 / args.batch_size)
    if warmup_steps == 0:
        warmup_steps = 1

    lr_schedule = lr_schedulers.PolynomialDecayWithLinearWarmup(
            target_learning_rate=args.learning_rate,
            warmup_steps=warmup_steps, decay_steps=num_train_steps)

    optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, epsilon=1e-8)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # Visualize the model
    model.summary(print_fn=logger.info)

    # Check that we've initialized properly
    expected_loss = compute_expected_loss_categorical(train_labels)
    init_bias_str = str(dict(zip(datasets.SEMEVAL_LABELS, initial_biases)))
    logger.info(f"Initial biases: {init_bias_str}")
    logger.info(f"Expected starting loss: {expected_loss}")
    actual_loss = model.evaluate(train_inputs, train_labels, verbose=1)
    logger.info(f"Actual starting loss: {actual_loss[0]}")

    # Define training callbacks
    tb_logdir = os.path.join(
            args.outdir, f"tensorboard_logs/model_{args.model_id}")
    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir)
    tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tb_logdir, histogram_freq=1,
            write_graph=True, write_images=True, update_freq="batch",
            embeddings_freq=1)
    lr_callback = LogLearningRateCallback(log_dir=tb_logdir)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=1, restore_best_weights=True)
    ckpt_outfile = os.path.join(
            args.outdir, f"model_checkpoints/model_{args.model_id}/model_ckpt")
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            ckpt_outfile, monitor="val_loss", save_best_only=True,
            save_weights_only=True)
    progress_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logger.info(
                f"Completed epoch {epoch}"))
    callbacks = [tb_callback, lr_callback, early_stop_callback,
                 ckpt_callback, progress_callback]

    # Train the model
    logger.info(f"TRAINING FOR {args.epochs} EPOCHS")
    history = model.fit(train_inputs, train_labels,
                        batch_size=args.batch_size,
                        epochs=args.epochs, validation_data=val_data,
                        shuffle=True, callbacks=callbacks, verbose=1)

    logger.info("TRAINING COMPLETE")
    res_str = "Final metrics on train/val data:"
    for (k, v) in history.history.items():
        res_str += f"\n  {k}: {v[-1]}"
    logger.info(res_str)

    # Evaluate the model
    if args.test_file is not None:
        logger.info("EVALUATING ON TEST DATA")
        predictions = model.predict(test_inputs, batch_size=args.batch_size,
                                    verbose=1)

        true_labels = np.argmax(test_labels, axis=1)
        pred_labels = np.argmax(predictions, axis=1)
        report = classification_report(true_labels, pred_labels,
                                       target_names=datasets.SEMEVAL_LABELS)
        logger.info('\n' + report)


def get_logger(logfile):
    logger = logging.getLogger("SemEval")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile)
    fmt = "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_processor_from_model_id(model_id, **kwargs):
    id_map = {
            "3a": input_processors.StandardProcessor(**kwargs),
            "3b": input_processors.EntityMentionProcessor(
                    **kwargs, keep_entity_markers=False),
            "3d": input_processors.EntityProcessor(**kwargs),
            "3e": input_processors.EntityMentionProcessor(
                    **kwargs, keep_entity_markers=True),
            "3f": input_processors.EntityStartProcessor(**kwargs)
            }
    return id_map[model_id]


def get_classification_head_from_model_id(model_id, **kwargs):
    id_map = {
            "3a": heads.CLSHead,
            "3b": heads.EntityMentionPoolHead,
            "3d": heads.CLSHead,
            "3e": heads.EntityMentionPoolHead,
            "3f": heads.EntityStartHead
            }
    return id_map[model_id](**kwargs)


def read_dataset(filepath, n=-1):
    data_reader = datasets.read_semeval(filepath)

    # TODO: My data processing pattern is off since I have to
    # exhaust the generator in order to process the input into an array...
    docs = []
    labels = []
    for (i, (sent, label)) in enumerate(data_reader):
        docs.append(sent)
        labels.append(label)
        if i == n:
            break
    docs = np.array(docs)
    labels = np.array(labels)
    return (docs, labels)


def get_single_examples(filepath):
    data_reader = datasets.read_semeval(filepath)

    num_labels = len(datasets.SEMEVAL_LABELS)
    docs = []
    labels = []
    seen_labels = set()
    for (sent, label) in data_reader:
        label_idx = np.argmax(label)
        if label_idx in seen_labels:
            continue
        docs.append(sent)
        labels.append(label)
        seen_labels.add(label_idx)
        if len(seen_labels) == num_labels:
            break
    docs = np.array(docs)
    labels = np.array(labels)
    return (docs, labels)


def compute_initial_bias(labels):
    """Assumes labels are binary encoded"""
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    biases = np.log(pos / neg)
    return biases


def compute_expected_loss_categorical(labels):
    """Assumes labels are binary encoded"""
    eps = 1e-8
    p_0 = labels.sum(axis=0) / labels.shape[0]
    p_0[p_0 == 0.0] = eps
    return -np.sum(p_0 * np.log(p_0))


if __name__ == "__main__":
    args = parse_args()
    main(args)
