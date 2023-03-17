import pandas as pd
import torch
from torch.utils.data import Dataset
from util import cleantext

class Covid19Dataset(Dataset):
    r"""PyTorch Dataset class for loading data.

    This is where the data parsing happens and where the text gets encoded using
    loaded tokenizer.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed
    straight into the model - `model(**batch)`.

    Arguments:

    path (:obj:`str`):
        Path to the data partition.

    use_tokenizer (:obj:`transformers.tokenization_?`):
        Transformer type tokenizer used to process raw text into numbers.

    labels_ids (:obj:`dict`):
        Dictionary to encode any labels names into numbers. Keys map to
        labels names and Values map to number associated to those labels.

    max_sequence_len (:obj:`int`, `optional`)
        Value to indicate the maximum desired sequence to truncate or pad text
        sequences. If no value is passed it will used maximum sequence size
        supported by the tokenizer and model.

    """

    def __init__(self, mode, use_tokenizer, labels_ids, data_path, max_sequence_len=None):
        assert mode in ['Train', 'Val', 'Test']
        self.mode = mode
        self.df = pd.read_excel(f'{data_path}/Constraint_English_' + self.mode + '.xlsx')
        self.df['tweet'] = self.df['tweet'].map(lambda x: cleantext(x))
        self.df['label'] = pd.get_dummies(self.df['label'])['real']
        self.len = len(self.df)
        self.texts = []
        self.labels = []

        for index, row in self.df.iterrows():
            # Save content.
            self.texts.append(row['tweet'])
            # Save encode labels.
            self.labels.append(row['label'])

            # Use tokenizer on texts. This can take a while.
        print('Using tokenizer on all texts. This can take a while...')
        self.inputs = use_tokenizer(self.texts, add_special_tokens=True, truncation=True, padding=True,
                                    return_tensors='pt', max_length=max_sequence_len)
        # Get maximum sequence length.
        self.sequence_len = self.inputs['input_ids'].shape[-1]
        print('Texts padded or truncated to %d length!' % self.sequence_len)
        # Add labels.
        self.inputs.update({'labels': torch.tensor(self.labels)})
        print('Finished!\n')

        return

    def __len__(self):
        r"""When used `len` return the number of examples.

        """

        return self.len

    def __getitem__(self, item):
        r"""Given an index return an example from the position.

        Arguments:

          item (:obj:`int`):
              Index position to pick an example to return.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.

        """

        return {key: self.inputs[key][item] for key in self.inputs.keys()}