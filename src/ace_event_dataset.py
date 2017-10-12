import torch
import torch.utils.data as torch_data

class MyDataset_batch(torch_data.Dataset):
    def __init__(self, sents, labels, triggers=None):
        self.sents = sents
        self.labels = labels
        self.triggers = triggers

    def __getitem__(self, index): # return tensor
        sent, target = self.sents[index], self.labels[index]
        sent_tensor = pad(torch.LongTensor(sent), 100)
        target_tensor = pad(torch.LongTensor(target), 100)
        if 0:
            print len(sent), len(target)
            print sent_tensor.size(), target_tensor.size()
        return sent_tensor, target_tensor

    def __len__(self):
        return len(self.sents)

class MyDataset(torch_data.Dataset):
    def __init__(self, sents, labels, triggers=None):
        self.sents = sents
        self.labels = labels
        self.triggers = triggers

    def __getitem__(self, index): # return tensor
        sent, target = self.sents[index], self.labels[index]
        sent_tensor = torch.LongTensor(sent)
        target_tensor = torch.LongTensor(target)
        return sent_tensor, target_tensor

    def __len__(self):
        return len(self.sents)


def pad(tensor, length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])


