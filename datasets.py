import torch
from torch.utils.data import DataLoader


class Vocab(object):
    def __init__(self, vocab):

        self.int2obj = dict()
        self.obj2int = dict()

        # ------------------
        # Define a vocabulary here.
        # input: a list of unique vocabs

        # Indexing vocabulary, starting from 1.
        self.int2obj = {ii: word for ii, word in enumerate(vocab, start=1)}  # integers: words
        num_words = len(vocab)
        self.int2obj[num_words+1] = "<unk>"  # unknown token: last index
        self.int2obj[0] = "<pad>"  # pad token: index 0, so that no need for attention_mask
        self.obj2int = {v: k for k, v in self.int2obj.items()}  # words: integers

        # ------------------

    def index_of(self, x: str) -> int:
        ''' Get index of a given character x'''
        # ------------------
        # converts an input character into its index

        return self.obj2int[x]

        # ------------------

    def object_of(self, x: int) -> str:
        ''' Get character of a given index'''
        # ------------------
        # converts an input index into corresponding character

        return self.int2obj[x]

        # ------------------

    def __len__(self) -> int:
        ''' Return the size of your vocabulary'''
        # ------------------

        return len(self.obj2int)  # including pad tokens

        # ------------------


def create_cls_dataloader(data_df, vocab, batch_size: int) -> DataLoader:
    vocabs = Vocab(vocab)
    max_len = int(max(data_df["wordcount"]))  # maximum length of a sentence

    def generate_cls_batch(batch):
        # convert sentences into indices
        # padding/truncate to make sentences to have same length
        X = []
        Y = []
        for sent, y in batch:  # each sentence has same length
            words_list = sent.split()
            num_words = len(words_list)
            feat = []  # indices of a sentence

            if num_words >= 1000:  # if longer: truncate text, head + tail
                words_list = words_list[:500] + words_list[-500:]
            else:  # if shorter: fill with pad first!!!
                feat += [vocabs.index_of("<pad>")] * (max_len - num_words)

            for word in words_list:  # a list of words in a sentence
                # append indices one word by one word
                if word in vocabs.obj2int:
                    feat.append(vocabs.index_of(word))
                else:
                    feat.append(vocabs.index_of("<unk>"))

            X.append(feat)  # indices of many sentences
            Y.append(y)  # labels of many sentences

        return torch.tensor(X), torch.tensor(Y).unsqueeze(1).float()

    train_df, valid_df = data_df[:4000], data_df[4000:]
    train_X = train_df["sent"].values.tolist()  # tokens separated by space in a text
    train_Y = train_df["label"].values
    valid_X = valid_df["sent"].values.tolist()
    valid_Y = valid_df["label"].values

    # use dataloader, to split data into batches
    train_dataloader = DataLoader(list(zip(train_X, train_Y)), batch_size=batch_size,
                                  shuffle=True, collate_fn=generate_cls_batch)
    valid_dataloader = DataLoader(list(zip(valid_X, valid_Y)), batch_size=batch_size,
                                  shuffle=False, collate_fn=generate_cls_batch)

    return train_dataloader, valid_dataloader, vocabs


def feat_extractor(data_df, vocab):
    # You also need vocab to convert each word into an index
    vocabs = Vocab(vocab)  # stored dictionaries
    max_len = int(max(data_df["wordcount"]))  # maximum length of a sentence

    X = []
    Y = []
    for i, row in data_df.iterrows():  # each sentence has length 20
        sent, y = row.sent, row.label
        feat = []  # indices of a sentence
        for word in sent.split():  # a list of words in a sentence
            # append indices one word by one word
            if word in vocabs.obj2int:
                feat.append(vocabs.index_of(word))
            else:
                feat.append(vocabs.index_of("<unk>"))
        num_words = len(feat)
        feat += [vocabs.index_of("<pad>")] * (max_len - num_words)

        X.append(feat)  # indices of many sentences
        Y.append(y)  # labels of many sentences

    return torch.tensor(X), torch.tensor(Y).unsqueeze(1).float()