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
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.modules.rnn import apply_permutation
from typing import List, Tuple
from torch.utils.data import DataLoader, TensorDataset, Dataset
import math
import numpy as np
# Add at the start of your script
import torch.backends.cudnn as cudnn
# Required imports for the enhanced training
from tqdm import tqdm
import torch.cuda.amp as amp
cudnn.benchmark = True  # Enable cudnn auto-tuner
cudnn.deterministic = False  # Slightly increases training speed

class IndependentDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()

        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    def forward(self, *items):

        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p) for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(-1) for item, mask in zip(items, masks)]

        return items

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            self.b_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            input_size = hidden_size * 2

        self.reset_parameters()

    def __repr__(self):
        s = f"{self.input_size}, {self.hidden_size}"
        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)

        return h, c

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size])) for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)

        return output, hx_n

    def forward(self, sequence, hx=None):
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        if hx is None:
            ih = x.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, sequence.sorted_indices)
        h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
        c = c.view(self.num_layers, 2, batch_size, self.hidden_size)

        for i in range(self.num_layers):
            x = torch.split(x, batch_sizes)
            if self.training:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                x = [i * mask[:len(i)] for i in x]
            x_f, (h_f, c_f) = self.layer_forward(x=x,
                                                 hx=(h[i, 0], c[i, 0]),
                                                 cell=self.f_cells[i],
                                                 batch_sizes=batch_sizes)
            x_b, (h_b, c_b) = self.layer_forward(x=x,
                                                 hx=(h[i, 1], c[i, 1]),
                                                 cell=self.b_cells[i],
                                                 batch_sizes=batch_sizes,
                                                 reverse=True)
            x = torch.cat((x_f, x_b), -1)
            h_n.append(torch.stack((h_f, h_b)))
            c_n.append(torch.stack((c_f, c_b)))
        x = PackedSequence(x,
                           sequence.batch_sizes,
                           sequence.sorted_indices,
                           sequence.unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, sequence.unsorted_indices)

        return x, hx

class SharedDropout(nn.Module):
    def __init__(self, p=0.5, batch_first=True):
        super().__init__()

        self.p = p
        self.batch_first = batch_first

    def __repr__(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, x):

        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p).unsqueeze(1)
            else:
                mask = self.get_mask(x[0], self.p)
            x = x * mask

        return x

    @staticmethod
    def get_mask(x, p):
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)

class MLP(nn.Module):

    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x

class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s

class WordEmbeddings(nn.Module):
    def __init__(self, n_word: int = 100, embedding_dim: int = 100):
        super().__init__()
        self.n_word = n_word
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_word, embedding_dim, padding_idx=0)

    @classmethod
    def fit(cls, sentences: List[List[tuple]], min_freq: int = 1, embedding_dim: int = 100) -> 'WordEmbeddings':
        word_counts = Counter(word for sent in sentences for word, _, _, _ in sent)
        vocab = {word for word, count in word_counts.items() if count >= min_freq}
        n_word = len(vocab) + 2  # +2 for <PAD> and <UNK>
        return cls(n_word=n_word, embedding_dim=embedding_dim)

class CharacterEmbeddings(nn.Module):
    def __init__(self, n_chars: int, char_embedding_dim: int = 50, char_hidden_dim: int = 100):
        super().__init__()
        self.n_chars = n_chars
        self.char_embedding_dim = char_embedding_dim
        self.char_hidden_dim = char_hidden_dim
        self.char_embedding = nn.Embedding(n_chars, char_embedding_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(
            char_embedding_dim, 
            char_hidden_dim // 2, 
            bidirectional=True, 
            batch_first=True
        )

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, max_word_len = chars.shape
        device = chars.device
        dtype = self.char_embedding.weight.dtype  # Get the model's dtype

        # Handle empty sequence case
        if seq_len == 0:
            return torch.zeros(batch_size, seq_len, self.char_hidden_dim, 
                             device=device, dtype=dtype)

        chars_reshaped = chars.view(batch_size * seq_len, max_word_len)
        char_embeds = self.char_embedding(chars_reshaped)

        lengths = (chars_reshaped != 0).sum(dim=1)
        lengths = lengths.cpu().long()

        # Handle all-zero lengths case
        if lengths.max() == 0:
            return torch.zeros(batch_size, seq_len, self.char_hidden_dim, 
                             device=device, dtype=dtype)

        # Get non-zero indices
        non_zero_idx = lengths.nonzero(as_tuple=False).squeeze()
        if non_zero_idx.dim() == 0:
            non_zero_idx = non_zero_idx.view(-1)

        char_embeds = char_embeds[non_zero_idx]
        lengths = lengths[non_zero_idx]

        packed_chars = nn.utils.rnn.pack_padded_sequence(
            char_embeds, 
            lengths,
            batch_first=True, 
            enforce_sorted=False
        )

        _, (hidden, _) = self.char_lstm(packed_chars)

        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(hidden.size(0), -1)

        # Create output tensor with matching dtype
        output = torch.zeros(batch_size * seq_len, self.char_hidden_dim, 
                           device=device, dtype=dtype)
        output[non_zero_idx] = hidden.to(dtype)  # Ensure matching dtype

        return output.view(batch_size, seq_len, -1)
    
class FieldEmbeddings(nn.Module):
    def __init__(self, n_vocab: int, embedding_dim: int):
        super().__init__()
        self._n_vocab = n_vocab
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_vocab, embedding_dim, padding_idx=0)

    @property
    def n_vocab(self) -> int:
        return self._n_vocab

class TokenEmbeddings(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings, char_embeddings: CharacterEmbeddings):
        super().__init__()
        self.word_embeddings = word_embeddings
        self.char_embeddings = char_embeddings
        self.n_words = word_embeddings.n_word
        self.pad_index = 0

    def forward(self, words, chars):
        word_embeds = self.word_embeddings.embedding(words)
        char_embeds = self.char_embeddings(chars)
        return torch.cat((word_embeds, char_embeds), dim=-1)

class VietnameseGrammarRule:
    def __init__(self):
        self.word_pos_map = defaultdict(set)  # Map words to possible POS tags
        self.pos_transitions = defaultdict(Counter)  # POS tag sequence patterns
        self.phrase_patterns = defaultdict(Counter)  # NP, VP, etc. patterns
        self.pos_phrase_map = defaultdict(set)  # POS tags to phrase types
        self.word_ner_map = defaultdict(set)  # Words to NER tags
    def add_sentence(self, sentence):
        prev_pos = None
        prev_phrase = None
        for word, pos, phrase, ner in sentence:
            self.word_pos_map[word].add(pos)
            if prev_pos:
                self.pos_transitions[prev_pos][pos] += 1
            if prev_phrase:
                self.phrase_patterns[prev_phrase][phrase] += 1
            self.pos_phrase_map[pos].add(phrase)
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
        self.sentences = []
    def parse_input(self, text_data):
        current_sentence = []
        
        for line in text_data.split('\n'):
            line = line.strip()
            if not line: 
                if current_sentence:
                    self.sentences.append(current_sentence)
                    self.grammar.add_sentence(current_sentence)
                current_sentence = []
                continue    
            parts = line.split('\t')
            if len(parts) >= 4:
                word, pos, phrase, ner = parts[:4]
                current_sentence.append((word, pos, phrase, ner))
                self.pos_vocab.add(pos)
                self.phrase_vocab.add(phrase)
                self.ner_vocab.add(ner)
        
        if current_sentence: 
            self.sentences.append(current_sentence)
            self.grammar.add_sentence(current_sentence)
        return self.sentences
    def generate_sentence(self, max_length=20):
        sentence = []
        current_pos = None
        current_phrase = None
        for _ in range(max_length):
            if not current_pos:
                pos_candidates = [pos for pos, counts in self.grammar.pos_transitions.items() 
                               if sum(counts.values()) > 0]
                if not pos_candidates:
                    break
                current_pos = self.weighted_choice(Counter(pos_candidates))
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
            
            if current_pos in {'CH', '.', '!', '?'}:
                break
        
        return sentence
    
    @staticmethod
    def weighted_choice(counter):
        total = sum(counter.values())
        r = random.uniform(0, total)
        cumsum = 0
        for item, count in counter.items():
            cumsum += count
            if r <= cumsum:
                return item
        return list(counter.keys())[0]  # fallback

    def generate_coherent_sentence(self, max_length=20):
        sentence = []
        current_pos = None
        current_phrase = None        
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
            if current_pos in {'CH', '.', '!', '?'}:
                break
        
        return sentence

class EnhancedVietnameseDependencyDataset(Dataset):
    def __init__(self, cfg: EnhancedVietnameseCFG):
        self.cfg = cfg
        self.sentences = cfg.sentences
        # Initialize regular dictionaries instead of defaultdict
        self.word2idx = {}
        self.char2idx = {}
        self.pos2idx = {}
        self.phrase2idx = {}
        self.ner2idx = {}
        self.max_word_len = 0
        self.next_word_idx = 0
        self.next_char_idx = 0
        self.next_pos_idx = 0
        self.next_phrase_idx = 0
        self.next_ner_idx = 0
        self.build_vocabs()

    def get_word_idx(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.next_word_idx
            self.next_word_idx += 1
        return self.word2idx[word]

    def get_char_idx(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.next_char_idx
            self.next_char_idx += 1
        return self.char2idx[char]

    def get_pos_idx(self, pos):
        if pos not in self.pos2idx:
            self.pos2idx[pos] = self.next_pos_idx
            self.next_pos_idx += 1
        return self.pos2idx[pos]

    def get_phrase_idx(self, phrase):
        if phrase not in self.phrase2idx:
            self.phrase2idx[phrase] = self.next_phrase_idx
            self.next_phrase_idx += 1
        return self.phrase2idx[phrase]

    def get_ner_idx(self, ner):
        if ner not in self.ner2idx:
            self.ner2idx[ner] = self.next_ner_idx
            self.next_ner_idx += 1
        return self.ner2idx[ner]

    def build_vocabs(self):
        for sentence in self.sentences:
            for word, pos, phrase, ner in sentence:
                self.get_word_idx(word)
                self.get_pos_idx(pos)
                self.get_phrase_idx(phrase)
                self.get_ner_idx(ner)
                for char in word:
                    self.get_char_idx(char)
                self.max_word_len = max(self.max_word_len, len(word))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words = [self.word2idx[word] for word, _, _, _ in sentence]
        chars = [[self.char2idx[c] for c in word] for word, _, _, _ in sentence]
        pos_tags = [self.pos2idx[pos] for _, pos, _, _ in sentence]
        phrases = [self.phrase2idx[phrase] for _, _, phrase, _ in sentence]
        ner_tags = [self.ner2idx[ner] for _, _, _, ner in sentence]
        return words, chars, pos_tags, phrases, ner_tags
    
def enhanced_pad_sequences(batch):
    words, chars, pos_tags, phrases, ner_tags = zip(*batch)
    
    # Pad sequences
    words_padded = pad_sequence([torch.tensor(s) for s in words], batch_first=True, padding_value=0)
    pos_padded = pad_sequence([torch.tensor(s) for s in pos_tags], batch_first=True, padding_value=0)
    phrases_padded = pad_sequence([torch.tensor(s) for s in phrases], batch_first=True, padding_value=0)
    ner_padded = pad_sequence([torch.tensor(s) for s in ner_tags], batch_first=True, padding_value=0)
    
    # Pad char sequences
    max_word_len = max(max(len(word) for word in sent) for sent in chars)
    max_word_len = max(1, max_word_len)  # Ensure max_word_len is at least 1
    chars_padded = torch.zeros((len(chars), max(map(len, chars)), max_word_len), dtype=torch.long)
    for i, sent in enumerate(chars):
        for j, word in enumerate(sent):
            if len(word) > 0:  # Only process non-empty words
                chars_padded[i, j, :len(word)] = torch.tensor(word)
    
    # Create mask
    mask = (words_padded != 0)
    
    return words_padded, chars_padded, pos_padded, phrases_padded, ner_padded, mask

def create_dependency_dataset(text_data: str) -> EnhancedVietnameseDependencyDataset:
    cfg = EnhancedVietnameseCFG()
    cfg.parse_input(text_data)
    return EnhancedVietnameseDependencyDataset(cfg)

# class EnhancedDependencyParserLSTM(nn.Module):
#     def __init__(self, word_vocab_size, char_vocab_size, pos_size, phrase_size, ner_size, 
#                  word_embed_dim=100, char_embed_dim=50, char_hidden_dim=100, 
#                  pos_embed_dim=50, phrase_embed_dim=50, ner_embed_dim=50, 
#                  lstm_hidden_dim=200):
#         super(EnhancedDependencyParserLSTM, self).__init__()
        
#         # Word Embeddings
#         self.word_embeddings = WordEmbeddings(n_word=word_vocab_size, embedding_dim=word_embed_dim)
        
#         # Character Embeddings
#         self.char_embeddings = CharacterEmbeddings(n_chars=char_vocab_size, 
#                                                    char_embedding_dim=char_embed_dim, 
#                                                    char_hidden_dim=char_hidden_dim)
        
#         # Field Embeddings (POS, Phrase, NER)
#         self.pos_embeddings = FieldEmbeddings(n_vocab=pos_size, embedding_dim=pos_embed_dim)
#         self.phrase_embeddings = FieldEmbeddings(n_vocab=phrase_size, embedding_dim=phrase_embed_dim)
#         self.ner_embeddings = FieldEmbeddings(n_vocab=ner_size, embedding_dim=ner_embed_dim)
        
#         # Token Embeddings (combines word and char embeddings)
#         self.token_embeddings = TokenEmbeddings(self.word_embeddings, self.char_embeddings)
        
#         # Calculate total embedding dimension
#         total_embed_dim = (word_embed_dim + char_hidden_dim + 
#                            pos_embed_dim + phrase_embed_dim + ner_embed_dim)
        
#         # LSTM layer
#         self.lstm = nn.LSTM(total_embed_dim, lstm_hidden_dim, 
#                             batch_first=True, bidirectional=True)
        
#         # Output layers
#         self.hidden2arc = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim)
#         self.arc_scorer = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)

#     def forward(self, words, chars, pos_tags, phrases, ner_tags, mask):
#         # Get embeddings for each type
#         token_embeds = self.token_embeddings(words, chars)
#         pos_embeds = self.pos_embeddings.embedding(pos_tags)
#         phrase_embeds = self.phrase_embeddings.embedding(phrases)
#         ner_embeds = self.ner_embeddings.embedding(ner_tags)
        
#         # Concatenate all embeddings
#         embeds = torch.cat((token_embeds, pos_embeds, phrase_embeds, ner_embeds), dim=2)
        
#         # LSTM layer
#         lstm_out, _ = self.lstm(embeds)
        
#         # Arc scoring
#         arc_space = self.hidden2arc(lstm_out)
#         arc_h = self.arc_scorer(arc_space)
#         arc_scores = torch.bmm(arc_h, arc_h.transpose(1, 2))
        
#         # Apply mask to exclude padding tokens
#         arc_scores = arc_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        
#         return arc_scores
class EnhancedDependencyParserLSTM(nn.Module):
    def __init__(self, n_words, n_chars, n_pos, n_phrases, n_ner, 
                 word_embed_dim=100, char_embed_dim=50, char_hidden_dim=100, 
                 pos_embed_dim=50, phrase_embed_dim=50, ner_embed_dim=50, 
                 lstm_hidden_dim=200, pad_index=0):
        super(EnhancedDependencyParserLSTM, self).__init__()
        
        self.word_embed = nn.Embedding(n_words, word_embed_dim, padding_idx=pad_index)
        self.char_embed = CharacterEmbeddings(n_chars, char_embed_dim, char_hidden_dim)
        self.pos_embed = nn.Embedding(n_pos, pos_embed_dim, padding_idx=pad_index)
        self.phrase_embed = nn.Embedding(n_phrases, phrase_embed_dim, padding_idx=pad_index)
        self.ner_embed = nn.Embedding(n_ner, ner_embed_dim, padding_idx=pad_index)
        
        self.embed_dropout = IndependentDropout(p=0.33)
        self.lstm = BiLSTM(input_size=word_embed_dim + char_hidden_dim + pos_embed_dim + phrase_embed_dim + ner_embed_dim, 
                           hidden_size=lstm_hidden_dim, num_layers=3, dropout=0.33)
        self.lstm_dropout = SharedDropout(p=0.33)
        
        self.mlp_arc_d = MLP(n_in=lstm_hidden_dim * 2, n_out=500, dropout=0.33)
        self.mlp_arc_h = MLP(n_in=lstm_hidden_dim * 2, n_out=500, dropout=0.33)
        self.mlp_rel_d = MLP(n_in=lstm_hidden_dim * 2, n_out=100, dropout=0.33)
        self.mlp_rel_h = MLP(n_in=lstm_hidden_dim * 2, n_out=100, dropout=0.33)
        
        self.arc_attn = Biaffine(n_in=500, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=100, n_out=n_ner, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        
        self.pad_index = pad_index

    def forward(self, words, chars, pos_tags, phrases, ner_tags, mask):
        word_embed = self.word_embed(words)
        char_embed = self.char_embed(chars)
        pos_embed = self.pos_embed(pos_tags)
        phrase_embed = self.phrase_embed(phrases)
        ner_embed = self.ner_embed(ner_tags)
        
        word_embed, char_embed, pos_embed, phrase_embed, ner_embed = self.embed_dropout(word_embed, char_embed, pos_embed, phrase_embed, ner_embed)
        
        embed = torch.cat((word_embed, char_embed, pos_embed, phrase_embed, ner_embed), dim=-1)
        lengths = mask.sum(1).cpu().long()  # Ensure lengths are on CPU and type int64
        x = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=words.size(1))
        x = self.lstm_dropout(x)
        
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)
        
        s_arc = self.arc_attn(arc_d, arc_h)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        
        return s_arc, s_rel
    
    def forward_loss(self, s_arc, s_rel, arcs, rels, mask):
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)
        return arc_loss + rel_loss
    
    @torch.no_grad()
    def predict(self, words, chars, pos_tags, phrases, ner_tags):
        self.eval()
        mask = words.ne(self.pad_index)
        s_arc, s_rel = self.forward(words, chars, pos_tags, phrases, ner_tags, mask)
        arc_preds, rel_preds = self.decode(s_arc, s_rel, mask)
        return arc_preds, rel_preds
    
    def decode(self, s_arc, s_rel, mask):
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds

def create_char_tensor(sentences, char2idx, max_word_len):
    char_tensor = []
    for sentence in sentences:
        sent_chars = []
        for word in sentence:
            word_chars = [char2idx.get(c, 0) for c in word[:max_word_len]]
            word_chars += [0] * (max_word_len - len(word_chars))  # Pad if necessary
            sent_chars.append(word_chars)
        char_tensor.append(sent_chars)
    return torch.LongTensor(char_tensor)

# class DependencyParser:
#     def __init__(self, model, word2idx, char2idx, pos2idx, phrase2idx, ner2idx):
#         self.model = model
#         self.word2idx = word2idx
#         self.char2idx = char2idx
#         self.pos2idx = pos2idx
#         self.phrase2idx = phrase2idx
#         self.ner2idx = ner2idx
#         self.idx2word = {v: k for k, v in word2idx.items()}
#         self.idx2pos = {v: k for k, v in pos2idx.items()}
#         self.idx2phrase = {v: k for k, v in phrase2idx.items()}
#         self.idx2ner = {v: k for k, v in ner2idx.items()}

#     def parse(self, sentence):
#         self.model.eval()
#         words = torch.tensor([self.word2idx.get(word, 0) for word, _, _, _ in sentence]).unsqueeze(0)
#         pos_tags = torch.tensor([self.pos2idx.get(pos, 0) for _, pos, _, _ in sentence]).unsqueeze(0)
#         phrases = torch.tensor([self.phrase2idx.get(phrase, 0) for _, _, phrase, _ in sentence]).unsqueeze(0)
#         ner_tags = torch.tensor([self.ner2idx.get(ner, 0) for _, _, _, ner in sentence]).unsqueeze(0)
#         chars = create_char_tensor([[word for word, _, _, _ in sentence]], self.char2idx, max_word_len=20)
#         mask = torch.ones_like(words, dtype=torch.bool)
        
#         with torch.no_grad():
#             arc_scores = self.model(words,chars, pos_tags, phrases, ner_tags, mask)
        
#         # Use Chu-Liu-Edmonds algorithm to find the maximum spanning tree
#         dependencies = self.chu_liu_edmonds(arc_scores.squeeze(0).numpy())
        
#         return dependencies

#     def chu_liu_edmonds(self, scores):
#         """
#         Implements the Chu-Liu-Edmonds algorithm for finding the maximum spanning tree.
#         """
#         def find_cycle(parent, i):
#             visited = set()
#             while i not in visited:
#                 visited.add(i)
#                 i = parent[i]
#                 if i == -1:
#                     return None
#             return i, list(visited)

#         def contract(scores, cycle):
#             cycle_score = 0
#             non_cycle = set(range(len(scores))) - set(cycle)
#             cycle_repr = cycle[0]
            
#             for i in non_cycle:
#                 max_score = float('-inf')
#                 max_j = -1
#                 for j in cycle:
#                     if scores[i, j] > max_score:
#                         max_score = scores[i, j]
#                         max_j = j
#                 scores[i, cycle_repr] = max_score
#                 scores[cycle_repr, i] = max([scores[j, i] for j in cycle])
                
#             for i in cycle:
#                 cycle_score += scores[parent[i], i]
                
#             return scores, cycle_repr, cycle_score

#         parent = np.argmax(scores, axis=0)
#         parent[0] = -1  # root has no parent
        
#         cycle = find_cycle(parent, 0)
#         while cycle is not None:
#             cycle_root, cycle = cycle
#             scores, cycle_repr, cycle_score = contract(scores, cycle)
#             parent = np.argmax(scores, axis=0)
#             parent[0] = -1
#             parent[cycle_repr] = -1 if cycle_root == 0 else parent[cycle_root]
#             cycle = find_cycle(parent, 0)

#         return parent.tolist()

#     def generate_sentence(self, max_length=20, temperature=1.0, num_sentences=1):
#         """
#         Generate new sentences using learned patterns and rules
#         Args:
#             max_length: Maximum length of each sentence
#             temperature: Controls randomness (lower = more conservative)
#             num_sentences: Number of sentences to generate
#         Returns:
#             List of generated sentences, each containing (word, pos, phrase, ner) tuples
#         """
#         self.model.eval()
#         generated_sentences = []

#         for _ in range(num_sentences):
#             # Start with common sentence-initial POS tags (based on Vietnamese grammar)
#             initial_pos_options = ['N', 'P', 'Np', 'M', 'L']  # Common sentence starters
#             current_pos = random.choice(initial_pos_options)

#             # Get a suitable word for the chosen POS
#             possible_words = [word for word, idx in self.word2idx.items() 
#                              if any(self.idx2pos[pos_idx] == current_pos 
#                                    for pos_idx in range(len(self.idx2pos)))]

#             if not possible_words:
#                 possible_words = list(self.word2idx.keys())

#             current_word = random.choice(possible_words)
#             generated = []
#             current_phrase_type = 'B-NP'  # Start with noun phrase

#             for position in range(max_length):
#                 # Get appropriate phrase type based on POS
#                 if current_pos in ['N', 'Np', 'P', 'L', 'M']:
#                     phrase_options = ['B-NP', 'I-NP']
#                 elif current_pos == 'V':
#                     phrase_options = ['B-VP', 'I-VP']
#                 elif current_pos == 'A':
#                     phrase_options = ['B-AP', 'I-AP']
#                 elif current_pos == 'E':
#                     phrase_options = ['B-PP', 'I-PP']
#                 else:
#                     phrase_options = ['O']

#                 # Determine NER tag based on word characteristics
#                 if current_pos == 'Np':  # Proper noun
#                     if any(loc in current_word for loc in ['Dương', 'Biển', 'Quốc']):
#                         ner_tag = 'B-LOC'
#                     elif any(org in current_word for org in ['Trẻ', 'Báo', 'Công ty']):
#                         ner_tag = 'B-ORG'
#                     else:
#                         ner_tag = 'O'
#                 else:
#                     ner_tag = 'O'

#                 # Add current token to generated sequence
#                 generated.append((current_word, current_pos, current_phrase_type, ner_tag))

#                 # Prepare input for model prediction
#                 words = torch.tensor([self.word2idx.get(w, 0) for w, _, _, _ in generated]).unsqueeze(0)

#                 # Create character tensor
#                 max_word_len = max(len(w) for w, _, _, _ in generated)
#                 char_tensor = torch.zeros((1, len(generated), max_word_len), dtype=torch.long)
#                 for i, (word, _, _, _) in enumerate(generated):
#                     for j, c in enumerate(word):
#                         if j < max_word_len:
#                             char_tensor[0, i, j] = self.char2idx.get(c, 0)

#                 # Create other tensors
#                 pos_tags = torch.tensor([self.pos2idx.get(p, 0) for _, p, _, _ in generated]).unsqueeze(0)
#                 phrases = torch.tensor([self.phrase2idx.get(ph, 0) for _, _, ph, _ in generated]).unsqueeze(0)
#                 ner_tags = torch.tensor([self.ner2idx.get(n, 0) for _, _, _, n in generated]).unsqueeze(0)
#                 mask = torch.ones_like(words, dtype=torch.bool)

#                 # Get model predictions
#                 with torch.no_grad():
#                     arc_scores = self.model(words, char_tensor, pos_tags, phrases, ner_tags, mask)

#                 # Use model predictions to influence next token selection
#                 next_scores = arc_scores[0, -1, :] / temperature
#                 next_probs = F.softmax(next_scores, dim=0)

#                 # Sample next position based on learned patterns
#                 if position < max_length - 1:
#                     # Determine next POS tag based on Vietnamese grammar rules
#                     if current_pos in ['N', 'Np', 'P']:
#                         next_pos_options = ['V', 'A', 'CH']
#                     elif current_pos == 'V':
#                         next_pos_options = ['N', 'Np', 'P', 'A', 'R']
#                     elif current_pos == 'A':
#                         next_pos_options = ['N', 'V', 'R']
#                     elif current_pos == 'R':
#                         next_pos_options = ['V', 'A']
#                     else:
#                         next_pos_options = list(set(self.idx2pos.values()))

#                     current_pos = random.choice(next_pos_options)

#                     # Get suitable word for the chosen POS
#                     possible_words = [word for word, idx in self.word2idx.items() 
#                                     if any(self.idx2pos[pos_idx] == current_pos 
#                                           for pos_idx in range(len(self.idx2pos)))]

#                     if not possible_words:
#                         possible_words = list(self.word2idx.keys())

#                     current_word = random.choice(possible_words)

#                     # Update phrase type
#                     if current_phrase_type.startswith('B-'):
#                         current_phrase_type = 'I-' + current_phrase_type[2:]
#                     elif random.random() < 0.3:  # 30% chance to start new phrase
#                         current_phrase_type = 'B-' + random.choice(['NP', 'VP', 'AP', 'PP'])

#                 # End sentence if we generate a terminal punctuation
#                 if current_pos == 'CH' and current_word in {'.', '!', '?', '...'}:
#                     break
                
#             generated_sentences.append(generated)

#         return generated_sentences
class DependencyParser:
    def __init__(self, model, word2idx, char2idx, pos2idx, phrase2idx, ner2idx):
        self.model = model
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.pos2idx = pos2idx
        self.phrase2idx = phrase2idx
        self.ner2idx = ner2idx
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.idx2pos = {v: k for k, v in pos2idx.items()}
        self.idx2phrase = {v: k for k, v in phrase2idx.items()}
        self.idx2ner = {v: k for k, v in ner2idx.items()}

    def parse(self, sentence):
        self.model.eval()
        words = torch.tensor([self.word2idx.get(word, 0) for word, _, _, _ in sentence]).unsqueeze(0)
        pos_tags = torch.tensor([self.pos2idx.get(pos, 0) for _, pos, _, _ in sentence]).unsqueeze(0)
        phrases = torch.tensor([self.phrase2idx.get(phrase, 0) for _, _, phrase, _ in sentence]).unsqueeze(0)
        ner_tags = torch.tensor([self.ner2idx.get(ner, 0) for _, _, _, ner in sentence]).unsqueeze(0)
        chars = create_char_tensor([[word for word, _, _, _ in sentence]], self.char2idx, max_word_len=20)
        mask = torch.ones_like(words, dtype=torch.bool)

        with torch.no_grad():
            arc_scores, _ = self.model(words, chars, pos_tags, phrases, ner_tags, mask)

        # Use Chu-Liu-Edmonds algorithm to find the maximum spanning tree
        dependencies = self.chu_liu_edmonds(arc_scores.squeeze(0).numpy())

        return dependencies

    def chu_liu_edmonds(self, scores):
        """
        Implements the Chu-Liu-Edmonds algorithm for finding the maximum spanning tree.
        """
        def find_cycle(parent, i):
            visited = set()
            while i not in visited:
                visited.add(i)
                i = parent[i]
                if i == -1:
                    return None
            return i, list(visited)

        def contract(scores, cycle):
            cycle_score = 0
            non_cycle = set(range(len(scores))) - set(cycle)
            cycle_repr = cycle[0]

            for i in non_cycle:
                max_score = float('-inf')
                max_j = -1
                for j in cycle:
                    if scores[i, j] > max_score:
                        max_score = scores[i, j]
                        max_j = j
                scores[i, cycle_repr] = max_score
                scores[cycle_repr, i] = max([scores[j, i] for j in cycle])

            for i in cycle:
                cycle_score += scores[parent[i], i]

            return scores, cycle_repr, cycle_score

        parent = np.argmax(scores, axis=0)
        parent[0] = -1  # root has no parent

        cycle = find_cycle(parent, 0)
        while cycle is not None:
            cycle_root, cycle = cycle
            scores, cycle_repr, cycle_score = contract(scores, cycle)
            parent = np.argmax(scores, axis=0)
            parent[0] = -1
            parent[cycle_repr] = -1 if cycle_root == 0 else parent[cycle_root]
            cycle = find_cycle(parent, 0)

        return parent.tolist()

    def generate_sentence(self, max_length=20, temperature=1.0, num_sentences=1):
        """
        Generate new sentences using learned patterns and rules
        Args:
            max_length: Maximum length of each sentence
            temperature: Controls randomness (lower = more conservative)
            num_sentences: Number of sentences to generate
        Returns:
            List of generated sentences, each containing (word, pos, phrase, ner) tuples
        """
        self.model.eval()
        generated_sentences = []
        for _ in range(num_sentences):
            initial_pos_options = ['N', 'P', 'Np', 'M', 'L']
            current_pos = random.choice(initial_pos_options)
            possible_words = [word for word, idx in self.word2idx.items()
                              if any(self.idx2pos[pos_idx] == current_pos
                                     for pos_idx in range(len(self.idx2pos)))]
            if not possible_words:
                possible_words = list(self.word2idx.keys())
            current_word = random.choice(possible_words)
            generated = []
            current_phrase_type = 'B-NP'
            for position in range(max_length):
                if current_pos in ['N', 'Np', 'P', 'L', 'M']:
                    phrase_options = ['B-NP', 'I-NP']
                elif current_pos == 'V':
                    phrase_options = ['B-VP', 'I-VP']
                elif current_pos == 'A':
                    phrase_options = ['B-AP', 'I-AP']
                elif current_pos == 'E':
                    phrase_options = ['B-PP', 'I-PP']
                else:
                    phrase_options = ['O']
                if current_pos == 'Np':
                    if any(loc in current_word for loc in ['Dương', 'Biển', 'Quốc']):
                        ner_tag = 'B-LOC'
                    elif any(org in current_word for org in ['Trẻ', 'Báo', 'Công ty']):
                        ner_tag = 'B-ORG'
                    else:
                        ner_tag = 'O'
                else:
                    ner_tag = 'O'
                generated.append((current_word, current_pos, current_phrase_type, ner_tag))
                words = torch.tensor([self.word2idx.get(w, 0) for w, _, _, _ in generated]).unsqueeze(0)
                max_word_len = max(len(w) for w, _, _, _ in generated)
                char_tensor = torch.zeros((1, len(generated), max_word_len), dtype=torch.long)
                for i, (word, _, _, _) in enumerate(generated):
                    for j, c in enumerate(word):
                        if j < max_word_len:
                            char_tensor[0, i, j] = self.char2idx.get(c, 0)
                pos_tags = torch.tensor([self.pos2idx.get(p, 0) for _, p, _, _ in generated]).unsqueeze(0)
                phrases = torch.tensor([self.phrase2idx.get(ph, 0) for _, _, ph, _ in generated]).unsqueeze(0)
                ner_tags = torch.tensor([self.ner2idx.get(n, 0) for _, _, _, n in generated]).unsqueeze(0)
                mask = torch.ones_like(words, dtype=torch.bool)
                with torch.no_grad():
                    arc_scores, _ = self.model(words, char_tensor, pos_tags, phrases, ner_tags, mask)
                next_scores = arc_scores[0, -1, :] / temperature
                next_probs = F.softmax(next_scores, dim=0)
                if position < max_length - 1:
                    if current_pos in ['N', 'Np', 'P']:
                        next_pos_options = ['V', 'A', 'CH']
                    elif current_pos == 'V':
                        next_pos_options = ['N', 'Np', 'P', 'A', 'R']
                    elif current_pos == 'A':
                        next_pos_options = ['N', 'V', 'R']
                    elif current_pos == 'R':
                        next_pos_options = ['V', 'A']
                    else:
                        next_pos_options = list(set(self.idx2pos.values()))
                    current_pos = random.choice(next_pos_options)
                    possible_words = [word for word, idx in self.word2idx.items()
                                      if any(self.idx2pos[pos_idx] == current_pos
                                             for pos_idx in range(len(self.idx2pos)))]
                    if not possible_words:
                        possible_words = list(self.word2idx.keys())
                    current_word = random.choice(possible_words)
                    if current_phrase_type.startswith('B-'):
                        current_phrase_type = 'I-' + current_phrase_type[2:]
                    elif random.random() < 0.3:
                        current_phrase_type = 'B-' + random.choice(['NP', 'VP', 'AP', 'PP'])
                if current_pos == 'CH' and current_word in {'.', '!', '?', '...'}:
                    break
            generated_sentences.append(generated)
        return generated_sentences

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint with all information needed to resume training
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    """
    Load a saved checkpoint
    
    Args:
        model: The PyTorch model to load parameters into
        optimizer: The optimizer to load state into
        path: Path to the checkpoint file
        
    Returns:
        epoch: The epoch number when checkpoint was saved
        loss: The loss value when checkpoint was saved
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def save_final_model(model, dataset, path):
    """
    Save the final model with separate model weights and metadata
    
    Args:
        model: The trained PyTorch model
        dataset: The EnhancedVietnameseDependencyDataset instance
        path: Path to save the model
    """
    # Save model weights
    torch.save(model.state_dict(), path)
    
    # Save metadata separately
    metadata = {
        'word2idx': dict(dataset.word2idx),
        'char2idx': dict(dataset.char2idx),
        'pos2idx': dict(dataset.pos2idx),
        'phrase2idx': dict(dataset.phrase2idx),
        'ner2idx': dict(dataset.ner2idx),
        'max_word_len': dataset.max_word_len
    }
    
    metadata_path = path.replace('.pth', '_metadata.pth')
    torch.save(metadata, metadata_path)

def load_final_model(model_class, path, strict=True):
    """
    Safely load a saved final model with security considerations
    
    Args:
        model_class: The PyTorch model class to instantiate
        path: Path to the saved model file
        strict: Whether to strictly enforce state dict loading
        
    Returns:
        model: The loaded model
        vocab_info: Dictionary containing vocabulary mappings
    """
    # Load state dict with weights_only=True for security
    state = torch.load(path, weights_only=True, map_location='cpu')
    
    # Get vocabulary sizes from a separate metadata file
    # This assumes you save metadata separately
    metadata_path = path.replace('.pth', '_metadata.pth')
    vocab_info = torch.load(metadata_path, weights_only=True)
    
    # Get vocabulary sizes
    word_vocab_size = len(vocab_info['word2idx'])
    char_vocab_size = len(vocab_info['char2idx'])
    pos_size = len(vocab_info['pos2idx'])
    phrase_size = len(vocab_info['phrase2idx'])
    ner_size = len(vocab_info['ner2idx'])
    
    # Create new model instance
    model = model_class(
        n_words=word_vocab_size,
        n_chars=char_vocab_size,
        n_pos=pos_size,
        n_phrases=phrase_size,
        n_ner=ner_size
    )
    
    # Load the saved parameters
    model.load_state_dict(state, strict=strict)
    
    return model, vocab_info

def train_model(model, train_loader, num_epochs, learning_rate,
                checkpoint_dir='checkpoints', final_model_path='final_model.pth',
                resume_from=None, gradient_accumulation_steps=1):
    """
    Enhanced training function with optimized GPU utilization and gradient accumulation
    """
    import torch
    import os
    from torch import nn
    from tqdm import tqdm

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)//1024//1024}MB")
        print(f"Cached: {torch.cuda.memory_reserved(0)//1024//1024}MB")
    else:
        print("Using CPU")

    # Move model to GPU
    model = model.to(device)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if needed
    start_epoch = 0
    if resume_from:
        start_epoch, _ = load_checkpoint(model, optimizer, resume_from)
        start_epoch += 1

    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        # Add progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

        for i, (words, chars, pos_tags, phrases, ner_tags, mask) in enumerate(progress_bar):
            # Move all tensors to GPU in non-blocking mode
            words = words.to(device, non_blocking=True)
            chars = chars.to(device, non_blocking=True)
            pos_tags = pos_tags.to(device, non_blocking=True)
            phrases = phrases.to(device, non_blocking=True)
            ner_tags = ner_tags.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Zero gradients every `gradient_accumulation_steps` iterations
            if i % gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            # Mixed precision training
            with torch.cuda.amp.autocast():
                s_arc, _ = model(words, chars, pos_tags, phrases, ner_tags, mask)
                batch_size, seq_len = words.size()
                target = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                target = target.masked_fill(~mask, -1)
                loss = criterion(s_arc.transpose(1, 2), target) / gradient_accumulation_steps

            # Scale loss and backpropagate
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Update progress bar and track loss
            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Save checkpoint every few epochs
        if (epoch + 1) % 200 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Memory management after each epoch
        torch.cuda.empty_cache()

def main():
    # Enable benchmarking for faster training
    torch.backends.cudnn.benchmark = True

    # Load your data
    with open("./train.txt", 'r', encoding='utf-8') as file:
        text_data = file.read()

    # Create dataset and dataloader with multiple workers
    cfg = EnhancedVietnameseCFG()
    cfg.parse_input(text_data)
    dataset = EnhancedVietnameseDependencyDataset(cfg)

    # Use multiple workers for faster data loading and efficient CPU utilization
    train_loader = DataLoader(
    dataset,
    batch_size=2048,            # Adjust batch size based on your GPU memory capacity
    shuffle=True,               # Shuffling data for each epoch
    collate_fn=enhanced_pad_sequences,  # Custom collate function
    num_workers=12,             # Set based on CPU cores, typically number of logical processors
    pin_memory=True,            # Faster data transfer to GPU
    prefetch_factor=4,          # Load 4 batches ahead of time to reduce GPU waiting
    persistent_workers=True     # Keep workers alive between epochs to reduce start-up time
    )

    # Get vocabulary sizes and create model
    word_vocab_size = len(dataset.word2idx)
    char_vocab_size = len(dataset.char2idx)
    pos_size = len(dataset.pos2idx)
    phrase_size = len(dataset.phrase2idx)
    ner_size = len(dataset.ner2idx)
    
    model = EnhancedDependencyParserLSTM(
        n_words=word_vocab_size,
        n_chars=char_vocab_size,
        n_pos=pos_size,
        n_phrases=phrase_size,
        n_ner=ner_size
    )

    # Train with GPU support
    train_model(
        model=model,
        train_loader=train_loader,
        num_epochs=2,
        learning_rate=0.001,
        checkpoint_dir='model_checkpoints',
        final_model_path='vietnamese_parser_final.pth',
        resume_from=None
    )

    # Save the model
    save_final_model(model, dataset, 'vietnamese_parser_final.pth')

    # Load the model
    model, vocab_info = load_final_model(
        model_class=EnhancedDependencyParserLSTM,
        path='vietnamese_parser_final.pth'
    )

    # Create parser with loaded model and vocabularies
    parser = DependencyParser(
        model=model,
        word2idx=vocab_info['word2idx'],
        char2idx=vocab_info['char2idx'],
        pos2idx=vocab_info['pos2idx'],
        phrase2idx=vocab_info['phrase2idx'],
        ner2idx=vocab_info['ner2idx']
    )

    # Test parsing
    test_sentence = [("Tôi", "P", "NP", "O"), ("yêu", "V", "VP", "O"), ("Việt Nam", "Np", "NP", "LOC")]
    dependencies = parser.parse(test_sentence)
    print("Dependencies:", dependencies)

    # Create sentences with different parameters
    sentences = parser.generate_sentence(
        max_length=15,      # Maximum words per sentence
        temperature=0.8,    # Lower = more conservative generations
        num_sentences=3     # Number of sentences to generate
    )

    # Print the generated sentences
    for i, sent in enumerate(sentences, 1):
        print(f"\nSentence {i}:")
        print(" ".join([word for word, _, _, _ in sent]))
        # print("POS:", " ".join([pos for _, pos, _, _ in sent]))
        # print("Phrase:", " ".join([phrase for _, _, phrase, _ in sent]))
        # print("NER:", " ".join([ner for _, _, _, ner in sent]))

    # Print some statistics
    print(f"\nDataset statistics:")
    print(f"Number of sentences: {len(dataset)}")
    print(f"Word vocabulary size: {word_vocab_size}")
    print(f"Character vocabulary size: {char_vocab_size}")
    print(f"Number of POS tags: {pos_size}")
    print(f"Number of phrase types: {phrase_size}")
    print(f"Number of NER tags: {ner_size}")

if __name__ == "__main__":
    main()
