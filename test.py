import random
from collections import defaultdict

class VietnameseCFG:
    def __init__(self):
        self.grammar = defaultdict(list)
        self.pos_to_words = defaultdict(list)
        self.start_symbols = set()

    def add_rule(self, lhs, rhs):
        self.grammar[lhs].append(rhs)

    def add_word(self, pos, word):
        self.pos_to_words[pos].append(word)

    def parse_input(self, input_data):
        sentences = input_data.split('\n\n')
        for sentence in sentences:
            words = []
            pos_tags = []
            chunk_tags = []
            for line in sentence.split('\n'):
                if line.strip():
                    word, pos, chunk, ner = line.split('\t')
                    words.append(word)
                    pos_tags.append(pos)
                    chunk_tags.append(chunk)
                    self.add_word(pos, word)

# Read the entire file content
with open('temp.txt', 'r', encoding='utf-8') as file:
    input_data = file.read()
    # print(input_data)

cfg = VietnameseCFG()