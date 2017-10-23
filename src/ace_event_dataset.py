import torch
import torch.utils.data as torch_data

class MyDataset(torch_data.Dataset):
    def __init__(self, dataset, use_tensor=False, use_pad=False):
        self.sents = [item[0] for item in dataset]
        self.labels = [item[1] for item in dataset]
        self.use_tensor = use_tensor
        self.use_pad = use_pad
        self.sent_lens = [len(sent) for sent in self.sents]

    def __getitem__(self, index): # return tensor
        sent, target = self.sents[index], self.labels[index]
        if self.use_tensor:
            if self.use_pad:
                return pad(torch.LongTensor(sent), 80), pad(torch.LongTensor(target), 80), self.sent_lens[index]
            else:
                return torch.LongTensor(sent), torch.LongTensor(target), self.sent_lens[index]
        else:
            return sent, target

    def __len__(self):
        return len(self.sents)

def pad(tensor, length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

def pad_batch_tensor(batch_tensor, length, batch_first=False):
    if batch_first: # batch * sent_len * ...
        batch_tensor = batch_tensor.transpose(0, 1)
    else: # sent_len * batch * ...
        batch_tensor = torch.cat([batch_tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()], dim=0)
    return batch_tensor.transpose(0, 1)

