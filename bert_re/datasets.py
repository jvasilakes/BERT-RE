import re
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True,
                        help="""Path to training data file.""")
    parser.add_argument("--test_file", type=str, required=True,
                        help="""Path to test data file.""")
    return parser.parse_args()


SEMEVAL_LABELS = ["Other", "Cause-Effect(e1,e2)", "Cause-Effect(e2,e1)",
                  "Component-Whole(e1,e2)", "Component-Whole(e2,e1)",
                  "Entity-Destination(e1,e2)", "Entity-Destination(e2,e1)",
                  "Message-Topic(e1,e2)", "Message-Topic(e2,e1)",
                  "Entity-Origin(e1,e2)", "Entity-Origin(e2,e1)",
                  "Member-Collection(e1,e2)", "Member-Collection(e2,e1)",
                  "Product-Producer(e1,e2)", "Product-Producer(e2,e1)",
                  "Content-Container(e1,e2)", "Content-Container(e2,e1)",
                  "Instrument-Agency(e1,e2)", "Instrument-Agency(e2,e1)"]


def read_semeval(datafile):
    label_encoder = dict(zip(SEMEVAL_LABELS, range(len(SEMEVAL_LABELS))))
    with open(datafile, 'r') as inF:
        example_lines = []
        for line in inF:
            if len(example_lines) == 4:
                # example_lines[0]: sentence
                # example_lines[1]: relation
                # example_lines[2]: comment, not used
                # example_lines[3]: blank line, not used
                sentence, relation, _, _ = example_lines
                processed_sent = process_semeval_sentence(sentence)
                onehot_label = np.zeros(shape=len(SEMEVAL_LABELS), dtype=int)
                onehot_label[label_encoder[relation]] = 1
                yield (processed_sent, onehot_label)
                example_lines = []
            example_lines.append(line.strip())

        sentence, relation, _, _ = example_lines
        processed_sent = process_semeval_sentence(sentence)
        onehot_label = np.zeros(shape=len(SEMEVAL_LABELS), dtype=int)
        onehot_label[label_encoder[relation]] = 1
        yield (processed_sent, onehot_label)


def process_semeval_sentence(sentence):
    sentence = sentence.split('\t')[1]  # Remove the number
    sentence = re.sub('<e1>', '[E1]', sentence)
    sentence = re.sub('<e2>', '[E2]', sentence)
    sentence = re.sub('</e1>', '[/E1]', sentence)
    sentence = re.sub('</e2>', '[/E2]', sentence)
    sentence = "[CLS] " + sentence + " [SEP]"
    return sentence
