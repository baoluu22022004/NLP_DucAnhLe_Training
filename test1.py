import os
from datetime import datetime
from collections import defaultdict
import random

from underthesea.utils import logger
from underthesea.utils.sp_data import Dataset
from underthesea.utils.sp_field import Field
from underthesea.utils.sp_fn import ispunct
from underthesea.utils.sp_init import PRETRAINED
from underthesea.utils.sp_metric import AttachmentMetric
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from underthesea.transforms.conll import CoNLL
from underthesea.modules.base import CharLSTM, IndependentDropout, BiLSTM, SharedDropout, MLP, Biaffine
from underthesea.modules.bert import BertEmbedding
from underthesea.utils.sp_alg import eisner, mst
import underthesea.modules.nn
from underthesea.utils.util_deep_learning import device
import warnings
from abc import abstractmethod
import torch.nn
import os
from underthesea.utils.util_deep_learning import device

class Model(torch.nn.Module):
    @staticmethod
    @abstractmethod
    def _init_model_with_state_dict(state):
        pass

    @classmethod
    def load(cls, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model file {model_path} does not exist.")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            state = torch.load(model_path, map_location='cpu')

        model = cls._init_model_with_state_dict(state)
        model.eval()
        model.to(device)
        return model

class DependencyParser(underthesea.modules.nn.Model):
    NAME = 'biaffine-dependency'

    def __init__(
            self,
            n_words=None,
            pad_index=0,
            unk_index=1,
            n_feats=None,
            n_rels=None,
            feat='char',
            n_embed=50,
            n_feat_embed=100,
            n_char_embed=50,
            bert=None,
            n_bert_layers=4,
            embed_dropout=.33,
            max_len=None,
            mix_dropout=.0,
            embeddings=[],
            embed=False,

            n_lstm_hidden=400,
            n_lstm_layers=3,
            lstm_dropout=.33,
            n_mlp_arc=500,
            n_mlp_rel=100,
            mlp_dropout=.33,
            feat_pad_index=0,
            init_pre_train=False,
            transform=None
    ):
        super(DependencyParser, self).__init__()
        self.embed = embed
        self.feat = feat
        self.embeddings = embeddings
        if len(self.embeddings) > 0:
            print(self.embeddings[0])
        self.args = {
            "n_words": n_words,
            'pad_index': pad_index,
            'unk_index': unk_index,
            "n_feats": n_feats,
            "n_rels": n_rels,
            'feat_pad_index': feat_pad_index,
            "feat": feat,
            'tree': False,
            'proj': False,
            'punct': False
        }

        if init_pre_train:
            return

        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        if feat == 'char':
            self.feat_embed = CharLSTM(n_chars=n_feats,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=feat_pad_index)
        elif feat == 'bert':
            self.feat_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=feat_pad_index,
                                            max_len=max_len,
                                            dropout=mix_dropout)
            self.n_feat_embed = self.feat_embed.n_out
        elif feat == 'tag':
            self.feat_embed = nn.Embedding(num_embeddings=n_feats,
                                           embedding_dim=n_feat_embed)
        else:
            raise RuntimeError("The feat type should be in ['char', 'bert', 'tag'].")
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_embed + n_feat_embed,
                           hidden_size=n_lstm_hidden,
                           num_layers=n_lstm_layers,
                           dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)

        # the MLP layers
        self.mlp_arc_d = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_arc_h = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index

        self.transform = transform
        feat = self.feat
        if feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.transform.FORM
        else:
            self.WORD, self.FEAT = self.transform.FORM, self.transform.CPOS
        self.ARC, self.REL = self.transform.HEAD, self.transform.DEPREL
        self.puncts = torch.tensor([i
                                    for s, i in self.WORD.vocab.stoi.items()
                                    if ispunct(s)]).to(device)

    def _get_state_dict(self):
        pass

    @torch.no_grad()
    def predict(
            self,
            data,
            buckets=8,
            batch_size=5000,
            pred=None,
            prob=False,
            tree=True,
            proj=False,
    ):
        self.transform.eval()
        if prob:
            self.transform.append(Field('probs'))

        logger.debug('Loading the data')
        dataset = Dataset(self.transform, data)
        dataset.build(batch_size, buckets)
        logger.debug(f'\n{dataset}')

        logger.debug('Making predictions on the dataset')
        start = datetime.now()
        loader = dataset.loader
        self.eval()

        arcs, rels, probs = [], [], []
        for words, feats in loader:
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.forward(words, feats)
            arc_preds, rel_preds = self.decode(s_arc, s_rel, mask, tree, proj)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if prob:
                arc_probs = s_arc.softmax(-1)
                probs.extend([prob[1:i + 1, :i + 1].cpu() for i, prob in zip(lens, arc_probs.unbind())])
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if prob:
            preds['probs'] = probs

        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None:
            logger.debug(f'Saving predicted results to {pred}')
            self.transform.save(pred, dataset.sentences)
        logger.debug(f'{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s')

        return dataset

    @torch.no_grad()
    def evaluate(self, loader):
        self.eval()

        total_loss, metric = 0, AttachmentMetric()

        tree = self.args['tree']
        proj = self.args['proj']

        for words, feats, arcs, rels in loader:
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.forward(words, feats)
            loss = self.forward_loss(s_arc, s_rel, arcs, rels, mask)
            arc_preds, rel_preds = self.decode(s_arc, s_rel, mask, tree, proj)
            # ignore all punctuation if not specified
            if not self.args['punct']:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @staticmethod
    def _init_model_with_state_dict(state):
        args = state['args']
        transform = state['transform']
        embeddings = state['embeddings']
        model = DependencyParser(
            n_words=args['n_words'],
            n_feats=args['n_feats'],
            n_rels=args['n_feats'],
            pad_index=args['pad_index'],
            unk_index=args['unk_index'],
            # bos_index=args.bos_index,
            feat_pad_index=args['feat_pad_index'],
            transform=transform,
            embeddings=embeddings
        )
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        return model

    @classmethod
    def load(cls, path):
        if os.path.exists(path):
            state = torch.load(path)
        else:
            path = PRETRAINED[path] if path in PRETRAINED else path
            state = torch.hub.load_state_dict_from_url(path)

        model = cls._init_model_with_state_dict(state)
        model.eval()
        model.to(device)
        return model

    def save(self, path):
        model = self
        if hasattr(self, 'module'):
            model = self.module

        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {
            'name': self.NAME,
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained,
            'transform': self.transform,
            'embeddings': self.embeddings
        }
        torch.save(state, path)

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self, words, feats):
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    def forward_loss(self, s_arc, s_rel, arcs, rels, mask):
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i + 1], proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds
    

class ModifiedDependencyParser(DependencyParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules = defaultdict(lambda: defaultdict(int))

    def extract_rules(self, words, feats, arcs, rels):
        # Convert tensors to lists for easier processing
        words = words.tolist()
        arcs = arcs.tolist()
        rels = rels.tolist()

        for i, (word, arc, rel) in enumerate(zip(words, arcs, rels)):
            if arc == 0:  # Root
                self.rules['ROOT'][self.WORD.vocab[word]] += 1
            else:
                head_word = self.WORD.vocab[words[arc-1]]
                dependent_word = self.WORD.vocab[word]
                relation = self.REL.vocab[rel]
                self.rules[head_word][(dependent_word, relation)] += 1

    def generate_sentence(self, max_length=10):
        sentence = []
        stack = ['ROOT']
        
        while stack and len(sentence) < max_length:
            current = stack.pop()
            if current != 'ROOT':
                sentence.append(current)
            
            possible_dependents = self.rules[current]
            if possible_dependents:
                dependents = sorted(possible_dependents.items(), key=lambda x: x[1], reverse=True)
                for (word, rel), _ in dependents[:3]:  # Consider top 3 most frequent dependents
                    if random.random() < 0.5:  # 50% chance to add each dependent
                        stack.append(word)

        return ' '.join(sentence)

    @torch.no_grad()
    def predict(self, data, *args, **kwargs):
        dataset = super().predict(data, *args, **kwargs)
        
        for words, feats, arcs, rels in zip(dataset.words, dataset.feats, dataset.arcs, dataset.rels):
            self.extract_rules(words, feats, arcs, rels)

        return dataset

    def train(self, train_data, dev_data, test_data, buckets, batch_size, epochs, **kwargs):
        # Assuming the original train method is called here
        super().train(train_data, dev_data, test_data, buckets, batch_size, epochs, **kwargs)

        # After training, extract rules from the training data
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        for words, feats, arcs, rels in train_loader:
            mask = words.ne(self.args.pad_index)
            self.extract_rules(words[mask], feats[mask], arcs[mask], rels[mask])

# Usage example:
model_path = "./temp.txt"
parser = DependencyParser.load(model_path)

# Generate new sentences based on extracted rules
# for _ in range(5):
#     print(parser.generate_sentence())