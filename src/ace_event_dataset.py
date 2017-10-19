import torch
import torch.utils.data as torch_data

class MyDataset(torch_data.Dataset):
    def __init__(self, dataset, use_tensor=False):
        self.sents = [item[0] for item in dataset]
        self.labels = [item[1] for item in dataset]
        self.use_tensor = use_tensor

    def __getitem__(self, index): # return tensor
        sent, target = self.sents[index], self.labels[index]
        if self.use_tensor:
            return torch.LongTensor(sent), torch.LongTensor(target)
        else:
            return sent, target

    def __len__(self):
        return len(self.sents)


def pad_batch_tensor(batch_tensor, length, batch_first=False):
    if batch_first: # batch * sent_len * ...
        batch_tensor = batch_tensor.transpose(0, 1)
    else: # sent_len * batch * ...
        batch_tensor = torch.cat([batch_tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()], dim=0)
    return batch_tensor.transpose(0, 1)

