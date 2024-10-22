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

class WordEmbeddings(nn.Module):
    def __init__(self, n_word: int = 100, embedding_dim: int = 100):
        super().__init__()
        self.n_word = n_word
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_word, embedding_dim, padding_idx=0)

    @classmethod
    def fit(cls, sentences: List[List[tuple]], min_freq: int = 1, embedding_dim: int = 100) -> 'WordEmbeddings':
        # Count word frequencies
        word_counts = Counter(word for sent in sentences for word, _, _, _ in sent)
        # Filter words by minimum frequency
        vocab = {word for word, count in word_counts.items() if count >= min_freq}
        # Add special tokens
        n_word = len(vocab) + 2  # +2 for <PAD> and <UNK>
        return cls(n_word=n_word, embedding_dim=embedding_dim)

class CharacterEmbeddings(nn.Module):
    def __init__(self, n_chars: int, char_embedding_dim: int = 50, char_hidden_dim: int = 100):
        super().__init__()
        self.n_chars = n_chars  # Added to store vocabulary size
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
        
        # Handle empty sequences
        if seq_len == 0:
            return torch.zeros(batch_size, seq_len, self.char_hidden_dim, device=chars.device)
        
        # Reshape for character embedding
        chars_reshaped = chars.view(batch_size * seq_len, max_word_len)
        char_embeds = self.char_embedding(chars_reshaped)
        
        # Create packed sequence for efficiency
        lengths = (chars_reshaped != 0).sum(dim=1)
        packed_chars = nn.utils.rnn.pack_padded_sequence(
            char_embeds, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Run through LSTM
        _, (hidden, _) = self.char_lstm(packed_chars)
        
        # Concatenate forward and backward hidden states
        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(hidden.size(0), -1)
        
        # Reshape back to (batch_size, seq_len, hidden_dim)
        return hidden.view(batch_size, seq_len, -1)
    
class FieldEmbeddings(nn.Module):
    def __init__(self, n_vocab: int, embedding_dim: int):
        super().__init__()
        self._n_vocab = n_vocab
        self.embedding_dim = embedding_dim  # Added to store embedding dimension
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
        """
        Args:
            words: Tensor of shape (batch_size, seq_len)
            chars: Tensor of shape (batch_size, seq_len, max_word_len)
        Returns:
            combined_embeddings: Tensor of shape (batch_size, seq_len, word_dim + char_hidden_dim)
        """
        word_embeds = self.word_embeddings.embedding(words)  # (batch_size, seq_len, word_dim)
        char_embeds = self.char_embeddings(chars)  # (batch_size, seq_len, char_hidden_dim)
        return torch.cat((word_embeds, char_embeds), dim=-1)

class StackEmbeddings(nn.Module):
    def __init__(self, embeddings: List[nn.Module]):
        super().__init__()
        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, x):
        return torch.cat([embed(x) for embed in self.embeddings], dim=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class ImprovedVietnameseTransformerModel(nn.Module):
    def __init__(
            self,
            token_embeddings: TokenEmbeddings,
            pos_embeddings: FieldEmbeddings,
            chunk_embeddings: FieldEmbeddings,
            ner_embeddings: FieldEmbeddings,
            d_model: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1
    ):
        super().__init__()
        
        self.token_embeddings = token_embeddings
        self.pos_embeddings = pos_embeddings
        self.chunk_embeddings = chunk_embeddings
        self.ner_embeddings = ner_embeddings
        
        # Fixed calculation of total embedding dimension
        total_embedding_dim = (
            token_embeddings.word_embeddings.embedding_dim +  # Changed from n_word
            token_embeddings.char_embeddings.char_hidden_dim +
            pos_embeddings.embedding_dim +
            chunk_embeddings.embedding_dim +
            ner_embeddings.embedding_dim
        )
        
        self.embedding_to_transformer = nn.Linear(total_embedding_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Added batch_first parameter
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc = nn.Linear(d_model, token_embeddings.word_embeddings.n_word)

    def forward(self, words: torch.Tensor, chars: torch.Tensor, 
                pos: torch.Tensor, chunk: torch.Tensor, 
                ner: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        token_embeds = self.token_embeddings(words, chars)
        pos_embeds = self.pos_embeddings(pos)
        chunk_embeds = self.chunk_embeddings(chunk)
        ner_embeds = self.ner_embeddings(ner)
        
        # Concatenate all embeddings
        x = torch.cat((token_embeds, pos_embeds, chunk_embeds, ner_embeds), dim=-1)
        x = self.embedding_to_transformer(x)
        
        # Apply transformer (now using batch_first=True)
        x = self.transformer(x, src_key_padding_mask=~mask)
        
        # Project to vocabulary size
        logits = self.fc(x)
        return logits
    
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

    def prepare_batch(self, sentences, batch_size, word_vocab, char_vocab, max_word_len=20):
        """Prepare batches for training"""
        random.shuffle(sentences)
        batches = []
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            
            # Process each sentence in the batch
            words, pos_tags, phrases, ners = [], [], [], []
            char_seqs = []
            
            max_length = max(len(s) for s in batch_sentences)
            
            for sentence in batch_sentences:
                # Pad sequences to max_length
                sentence_words = []
                sentence_pos = []
                sentence_phrase = []
                sentence_ner = []
                sentence_chars = []
                
                for word, pos, phrase, ner in sentence:
                    # Convert word to index
                    word_idx = word_vocab.get(word, word_vocab['<UNK>'])
                    sentence_words.append(word_idx)
                    
                    # Convert POS tag to index
                    pos_idx = self.pos_to_idx.get(pos, self.pos_to_idx['<UNK>'])
                    sentence_pos.append(pos_idx)
                    
                    # Convert phrase tag to index
                    phrase_idx = self.phrase_to_idx.get(phrase, self.phrase_to_idx['<UNK>'])
                    sentence_phrase.append(phrase_idx)
                    
                    # Convert NER tag to index
                    ner_idx = self.ner_to_idx.get(ner, self.ner_to_idx['<UNK>'])
                    sentence_ner.append(ner_idx)
                    
                    # Process characters
                    char_indices = [char_vocab.get(c, char_vocab['<UNK>']) for c in word[:max_word_len]]
                    char_indices = char_indices + [0] * (max_word_len - len(char_indices))
                    sentence_chars.append(char_indices)
                
                # Pad sequences
                padding_length = max_length - len(sentence)
                sentence_words.extend([0] * padding_length)
                sentence_pos.extend([0] * padding_length)
                sentence_phrase.extend([0] * padding_length)
                sentence_ner.extend([0] * padding_length)
                sentence_chars.extend([[0] * max_word_len] * padding_length)
                
                words.append(sentence_words)
                pos_tags.append(sentence_pos)
                phrases.append(sentence_phrase)
                ners.append(sentence_ner)
                char_seqs.append(sentence_chars)
            
            # Convert to tensors
            words_tensor = torch.LongTensor(words)
            chars_tensor = torch.LongTensor(char_seqs)
            pos_tensor = torch.LongTensor(pos_tags)
            phrase_tensor = torch.LongTensor(phrases)
            ner_tensor = torch.LongTensor(ners)
            mask = words_tensor != 0
            
            batches.append((words_tensor, chars_tensor, pos_tensor, phrase_tensor, ner_tensor, mask))
        
        return batches

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
                current_pos = random.choice(pos_candidates)
            else:
                # Choose next POS based on transitions
                transitions = self.grammar.pos_transitions[current_pos]
                if not transitions:
                    break
                current_pos = weighted_choice(transitions)
            
            # Choose a word matching the POS tag
            possible_words = [word for word, pos_tags in self.grammar.word_pos_map.items() 
                            if current_pos in pos_tags]
            if not possible_words:
                break
            word = random.choice(possible_words)
            
            # Get phrase type based on POS
            possible_phrases = self.grammar.pos_phrase_map[current_pos]
            if possible_phrases:
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

def prepare_data(data, word_to_idx, pos_to_idx, chunk_to_idx, ner_to_idx):
    word_ids, pos_ids, chunk_ids, ner_ids = [], [], [], []
    
    # Get special token indices
    pad_idx = 0  # <PAD>
    unk_idx = 1  # <UNK>
    
    for word, pos, chunk, ner in data:
        # Ensure word index is within vocabulary range
        word_id = word_to_idx.get(word, unk_idx)
        pos_id = pos_to_idx.get(pos, unk_idx)
        chunk_id = chunk_to_idx.get(chunk, unk_idx)
        ner_id = ner_to_idx.get(ner, unk_idx)
        
        word_ids.append(word_id)
        pos_ids.append(pos_id)
        chunk_ids.append(chunk_id)
        ner_ids.append(ner_id)

    return list(zip(word_ids, pos_ids, chunk_ids, ner_ids))

def batch_data(data, batch_size, max_word_len=20):
    random.shuffle(data)
    batches = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        word_seqs, pos_seqs, chunk_seqs, ner_seqs = zip(*batch)
        
        # Pad sequences
        words = pad_sequence([torch.LongTensor(s) for s in word_seqs], 
                           batch_first=True, padding_value=0)
        pos = pad_sequence([torch.LongTensor(s) for s in pos_seqs],
                          batch_first=True, padding_value=0)
        chunk = pad_sequence([torch.LongTensor(s) for s in chunk_seqs],
                           batch_first=True, padding_value=0)
        ner = pad_sequence([torch.LongTensor(s) for s in ner_seqs],
                          batch_first=True, padding_value=0)
        
        # Create mask
        mask = (words != 0)
        
        # Prepare character data
        chars = prepare_chars(words, max_word_len)
        
        batches.append((words, chars, pos, chunk, ner, mask))
    
    return batches

def prepare_batch(batch, word_to_idx, char_to_idx, max_word_len=20):
    """
    Prepare a batch of data with proper character-level encoding
    """
    words, pos, chunk, ner = batch
    batch_size, seq_len = words.shape
    
    # Initialize character tensor with padding index (0)
    chars = torch.zeros((batch_size, seq_len, max_word_len), dtype=torch.long)
    
    # Convert words to characters
    for b in range(batch_size):
        for s in range(seq_len):
            if words[b, s] != 0:  # Skip padding tokens
                word = word_to_idx.inv[words[b, s].item()]  # Convert back to word
                char_ids = [char_to_idx.get(c, 1) for c in word[:max_word_len]]  # 1 is UNK
                chars[b, s, :len(char_ids)] = torch.tensor(char_ids)
    
    return words, chars, pos, chunk, ner

def prepare_chars(words, max_word_len):
    batch_size, seq_len = words.shape
    chars = torch.zeros((batch_size, seq_len, max_word_len), dtype=torch.long)
    return chars
# def weighted_choice(counter):
#     """Choose an item from a counter based on frequencies"""
#     total = sum(counter.values())
#     r = random.uniform(0, total)
#     cumsum = 0
#     for item, count in counter.items():
#         cumsum += count
#         if r <= cumsum:
#             return item
#     return list(counter.keys())[0]  # fallback

class DependencyParserTrainer:
    def __init__(self, corpus: str):
        self.corpus = corpus
        self.sentences = self.parse_input(corpus)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize vocabularies
        self.word_vocab = self.create_vocab([word for sent in self.sentences for word, _, _, _ in sent])
        self.pos_vocab = self.create_vocab([pos for sent in self.sentences for _, pos, _, _ in sent])
        self.chunk_vocab = self.create_vocab([chunk for sent in self.sentences for _, _, chunk, _ in sent])
        self.ner_vocab = self.create_vocab([ner for sent in self.sentences for _, _, _, ner in sent])
        
        # Initialize embeddings
        self.word_embeddings = WordEmbeddings(len(self.word_vocab), 100)
        self.char_embeddings = CharacterEmbeddings(len(self.char_vocab), 50, 100)
        self.token_embeddings = TokenEmbeddings(self.word_embeddings, self.char_embeddings)
        
        # Initialize field embeddings
        self.pos_embeddings = FieldEmbeddings(len(self.pos_vocab), 50)
        self.chunk_embeddings = FieldEmbeddings(len(self.chunk_vocab), 50)
        self.ner_embeddings = FieldEmbeddings(len(self.ner_vocab), 50)
        
        # Initialize model
        self.model = ImprovedVietnameseTransformerModel(
            token_embeddings=self.token_embeddings,
            pos_embeddings=self.pos_embeddings,
            chunk_embeddings=self.chunk_embeddings,
            ner_embeddings=self.ner_embeddings
        ).to(self.device)

    def create_vocab(self, items):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, item in enumerate(set(items), start=2):
            vocab[item] = idx
        return vocab

    def parse_input(self, text_data):
        sentences = []
        current_sentence = []
        
        for line in text_data.split('\n'):
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                current_sentence = []
                continue
                
            parts = line.split('\t')
            if len(parts) >= 4:
                word, pos, phrase, ner = parts[:4]
                current_sentence.append((word, pos, phrase, ner))
        
        if current_sentence:
            sentences.append(current_sentence)
        
        return sentences
                
def main():
    # Load the corpus data
    with open("./temp.txt", 'r', encoding='utf-8') as file:
        corpus = file.read()

    # Initialize the DependencyParserTrainer
    trainer = DependencyParserTrainer(corpus)

    # Define training parameters
    base_path = Path("./model_checkpoints/vietnamese_parser.pt")
    batch_size = 32
    lr = 0.0001
    max_epochs = 10

    # Train the model
    trainer.train(
        base_path=base_path,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs
    )

    # Print model statistics
    print("\nTraining Statistics:")
    print(f"Total sentences processed: {len(trainer.sentences)}")
    print(f"Vocabulary sizes:")
    print(f"- Words: {len(trainer.word_vocab)}")
    print(f"- POS tags: {len(trainer.pos_vocab)}")
    print(f"- Chunks: {len(trainer.chunk_vocab)}")
    print(f"- NER tags: {len(trainer.ner_vocab)}")
    
    # Print model parameters
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\nModel Architecture:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

if __name__ == "__main__":
    main()