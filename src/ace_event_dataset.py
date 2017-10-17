import torch
import torch.utils.data as torch_data

class MyDataset_batch(torch_data.Dataset):
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels
        self.sent_lens = [len(sent) for sent in sents]

    def __getitem__(self, index): # return tensor
        sent, target = self.sents[index], self.labels[index]
        sent_tensor = pad(torch.LongTensor(sent), 80)
        target_tensor = pad(torch.LongTensor(target), 80)
        if 0:
            print len(sent), len(target)
            print sent_tensor.size(), target_tensor.size()
        return sent_tensor, target_tensor, self.sent_lens[index]

    def __len__(self):
        return len(self.sents)

class MyDataset(torch_data.Dataset):
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels
        self.sent_lens = [len(sent) for sent in sents]

    def __getitem__(self, index): # return tensor
        sent, target = self.sents[index], self.labels[index]
        sent_tensor = torch.LongTensor(sent)
        target_tensor = torch.LongTensor(target)
        return sent_tensor, target_tensor, self.sent_lens[index]

    def __len__(self):
        return len(self.sents)


def pad(tensor, length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

def pad_batch(batch_tensor, length, batch_first=False):
    if batch_first: # batch * sent_len * ...
        batch_tensor = batch_tensor.transpose(0, 1)
    else: # sent_len * batch * ...
        batch_tensor = torch.cat([batch_tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()], dim=0)
    return batch_tensor.transpose(0, 1)

