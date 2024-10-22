import os
from datetime import timedelta, datetime
from pathlib import Path
from typing import Union
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict, Counter
import random
import torch.nn.functional as F
from typing import List, Tuple
from torch.utils.data import DataLoader, TensorDataset
import math

class VietnameseGrammarRule:
    def __init__(self):
        self.word_pos_map = defaultdict(set)  # Map words to possible POS tags
        self.pos_transitions = defaultdict(Counter)  # POS tag sequence patterns
        self.phrase_patterns = defaultdict(Counter)  # NP, VP, etc. patterns
        self.pos_phrase_map = defaultdict(set)  # POS tags to phrase types
        self.word_ner_map = defaultdict(set)  # Words to NER tags

    def add_sentence(self, sentence):
        """Learn patterns from a sentence"""
        prev_pos = None
        prev_phrase = None
        
        for word, pos, phrase, ner in sentence:
            # Add word-POS relationship
            self.word_pos_map[word].add(pos)
            
            # Add POS transitions
            if prev_pos:
                self.pos_transitions[prev_pos][pos] += 1
            
            # Add phrase patterns
            if prev_phrase:
                self.phrase_patterns[prev_phrase][phrase] += 1
            
            # Map POS to phrase types
            self.pos_phrase_map[pos].add(phrase)
            
            # Add NER mapping
            if ner != 'O':
                self.word_ner_map[word].add(ner)
            
            prev_pos = pos
            prev_phrase = phrase

class EnhancedVietnameseCFG:
    def __init__(self):
        self.grammar = VietnameseGrammarRule()
        self.pos_vocab = set()
        self.phrase_vocab = set()
        self.ner_vocab = set()

    def parse_input(self, text_data):
        """Parse the Vietnamese text data format"""
        sentences = []
        current_sentence = []
        
        for line in text_data.split('\n'):
            line = line.strip()
            if not line:  # Empty line indicates sentence boundary
                if current_sentence:
                    sentences.append(current_sentence)
                    # Learn grammar rules from the sentence
                    self.grammar.add_sentence(current_sentence)
                current_sentence = []
                continue
                
            parts = line.split('\t')
            if len(parts) >= 4:
                word, pos, phrase, ner = parts[:4]
                current_sentence.append((word, pos, phrase, ner))
                
                # Build vocabularies
                self.pos_vocab.add(pos)
                self.phrase_vocab.add(phrase)
                self.ner_vocab.add(ner)
        
        if current_sentence:  # Don't forget the last sentence
            sentences.append(current_sentence)
            self.grammar.add_sentence(current_sentence)
        
        return sentences

    def generate_sentence(self, max_length=20):
        """Generate a new sentence using learned grammar rules"""
        sentence = []
        current_pos = None
        current_phrase = None
        
        for _ in range(max_length):
            # Choose next POS tag based on transitions
            if not current_pos:
                # Start with a common sentence-initial POS
                pos_candidates = [pos for pos, counts in self.grammar.pos_transitions.items() 
                               if sum(counts.values()) > 0]
                if not pos_candidates:
                    break
                current_pos = self.weighted_choice(Counter(pos_candidates))
            else:
                # Choose next POS based on transitions
                transitions = self.grammar.pos_transitions[current_pos]
                if not transitions:
                    break
                current_pos = self.weighted_choice(transitions)
            
            # Choose a word matching the POS tag
            possible_words = [word for word, pos_tags in self.grammar.word_pos_map.items() 
                            if current_pos in pos_tags]
            if not possible_words:
                continue  # Skip this iteration if no matching word is found
            word = random.choice(possible_words)
            
            # Get phrase type based on POS
            possible_phrases = self.grammar.pos_phrase_map[current_pos]
            if possible_phrases:
                if current_phrase:
                    # Use phrase patterns to determine the next phrase
                    phrase_transitions = self.grammar.phrase_patterns[current_phrase]
                    if phrase_transitions:
                        current_phrase = self.weighted_choice(phrase_transitions)
                    else:
                        current_phrase = random.choice(list(possible_phrases))
                else:
                    current_phrase = random.choice(list(possible_phrases))
            
            # Get NER tag if applicable
            ner_tag = 'O'
            if word in self.grammar.word_ner_map:
                ner_tag = random.choice(list(self.grammar.word_ner_map[word]))
            
            sentence.append((word, current_pos, current_phrase, ner_tag))
            
            # End sentence if we generate a terminal POS tag
            if current_pos in {'CH', '.', '!', '?'}:
                break
        
        return sentence
    
    @staticmethod
    def weighted_choice(counter):
        """Choose an item from a counter based on frequencies"""
        total = sum(counter.values())
        r = random.uniform(0, total)
        cumsum = 0
        for item, count in counter.items():
            cumsum += count
            if r <= cumsum:
                return item
        return list(counter.keys())[0]  # fallback

    def generate_coherent_sentence(self, max_length=20):
        """Generate a more coherent sentence using learned grammar rules"""
        sentence = []
        current_pos = None
        current_phrase = None
        
        # Start with a subject (typically a noun or pronoun)
        initial_pos = self.weighted_choice(Counter(['N', 'P', 'Np']))
        
        for _ in range(max_length):
            if not current_pos:
                current_pos = initial_pos
            else:
                transitions = self.grammar.pos_transitions[current_pos]
                if not transitions:
                    break
                current_pos = self.weighted_choice(transitions)
            
            possible_words = [word for word, pos_tags in self.grammar.word_pos_map.items() 
                            if current_pos in pos_tags]
            if not possible_words:
                continue
            
            word = random.choice(possible_words)
            
            possible_phrases = self.grammar.pos_phrase_map[current_pos]
            if possible_phrases:
                if current_phrase:
                    phrase_transitions = self.grammar.phrase_patterns[current_phrase]
                    if phrase_transitions:
                        current_phrase = self.weighted_choice(phrase_transitions)
                    else:
                        current_phrase = random.choice(list(possible_phrases))
                else:
                    current_phrase = random.choice(list(possible_phrases))
            
            ner_tag = 'O'
            if word in self.grammar.word_ner_map:
                ner_tag = random.choice(list(self.grammar.word_ner_map[word]))
            
            sentence.append((word, current_pos, current_phrase, ner_tag))
            
            # End sentence if we generate a terminal POS tag
            if current_pos in {'CH', '.', '!', '?'}:
                break
        
        return sentence

def main():
    # Load the corpus data
    with open("./train.txt", 'r', encoding='utf-8') as file:
        corpus = file.read()
    cfg = EnhancedVietnameseCFG()
    cfg.parse_input(corpus)

    sentence = cfg.generate_coherent_sentence()
    print(" ".join([word for word, _, _, _ in sentence]))
if __name__ == "__main__":
    main()