import json
from torch.utils.data import Dataset

def collate_fn(batch):
    return tuple(zip(*batch))
            
class CLIRDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.src_ids = []
        self.queries = []
        self.targets = []
        with open(data_path, "r") as fin:
            for line in fin.readlines():
                """
                {
                    "src_id": "39", 
                    "src_query": "Albedo", 
                    "tgt_results": [["553658", 6], ["1712206", 5], ["1849020", 5], ["1841381", 5], ["1541246", 3], ["5248845", 2], ["1498501", 2], ["5748160", 2], ["718267", 2], ["5392042", 2], ["2764586", 2], ["202402", 2], ["3316208", 2], ["5375638", 2], ["1161946", 2], ["3542927", 2], ["801173", 2], ["5378920", 2], ["3543134", 2], ["1782326", 1], ["939382", 1], ["3779245", 1], ["2938855", 1], ["3316164", 1], ["5702473", 1], ["939409", 1], ["2938822", 1], ["3315916", 1], ["3542015", 1], ["1156740", 1], ["1042704", 1], ["1690586", 1], ["701461", 1], ["3544317", 1], ["5255753", 1], ["3547732", 1], ["5892814", 1], ["3545801", 1], ["3315788", 1], ["1661774", 1], ["3316223", 1], ["6231204", 1], ["3313232", 1], ["3314496", 1], ["5377208", 1], ["3543616", 1], ["1664298", 1], ["5525991", 1], ["1669824", 1], ["1454516", 1]]
                }
                """
                l_data = json.loads(line)
                self.src_ids.append(l_data["src_id"])
                self.queries.append(l_data["src_query"])
                self.targets.append(l_data["tgt_results"])

    def __len__(self):
        return len(self.src_ids)
    
    def __getitem__(self, index):
        return self.src_ids[index], self.queries[index], self.targets[index]