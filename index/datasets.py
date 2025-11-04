import numpy as np
import torch
import torch.utils.data as data
import os

class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        loaded_data = np.load(data_path, allow_pickle=True)

        # Handle both old array format and new dictionary format
        if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object:
            # Try to convert to dict (new format with ASIN info)
            try:
                data_dict = loaded_data.item()
                if isinstance(data_dict, dict) and 'embeddings' in data_dict:
                    self.embeddings = data_dict['embeddings']
                    self.asins = data_dict.get('asins', None)
                    print(f"[EmbDataset] Loaded from dict format: {len(self.embeddings)} items")
                    if self.asins:
                        print(f"[EmbDataset] ASIN information available: {len(self.asins)} ASINs")
                else:
                    # Old format: plain array
                    self.embeddings = loaded_data
                    self.asins = None
                    print(f"[EmbDataset] Loaded from plain array format: {self.embeddings.shape}")
            except (ValueError, AttributeError):
                # Old format: plain array
                self.embeddings = loaded_data
                self.asins = None
                print(f"[EmbDataset] Loaded from plain array format: {self.embeddings.shape}")
        else:
            # Old format: plain array
            self.embeddings = loaded_data
            self.asins = None
            print(f"[EmbDataset] Loaded from plain array format: {self.embeddings.shape}")

        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetAll(data.Dataset):

    def __init__(self, args):

        self.datasets = args.datasets.split(',')
        embeddings = []
        self.dataset_count = []
        self.all_asins = []  # Store ASINs from all datasets

        for dataset in self.datasets:
            print(dataset)
            embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
            loaded_data = np.load(embedding_path, allow_pickle=True)

            # Handle both old array format and new dictionary format
            if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object:
                try:
                    data_dict = loaded_data.item()
                    if isinstance(data_dict, dict) and 'embeddings' in data_dict:
                        embedding = data_dict['embeddings']
                        asins = data_dict.get('asins', None)
                        print(f"[EmbDatasetAll] Loaded {dataset} from dict format: {len(embedding)} items")
                        if asins:
                            self.all_asins.extend(asins)
                            print(f"[EmbDatasetAll] ASIN information available for {dataset}")
                    else:
                        embedding = loaded_data
                        print(f"[EmbDatasetAll] Loaded {dataset} from plain array format")
                except (ValueError, AttributeError):
                    embedding = loaded_data
                    print(f"[EmbDatasetAll] Loaded {dataset} from plain array format")
            else:
                embedding = loaded_data
                print(f"[EmbDatasetAll] Loaded {dataset} from plain array format")

            embeddings.append(embedding)
            self.dataset_count.append(embedding.shape[0])

        self.embeddings = np.concatenate(embeddings)
        self.dim = self.embeddings.shape[-1]

        print(self.dataset_count)
        print(self.embeddings.shape[0])
        if self.all_asins:
            print(f"[EmbDatasetAll] Total ASINs collected: {len(self.all_asins)}")

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetOne(data.Dataset):

    def __init__(self, args, dataset):


        print(dataset)
        embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
        loaded_data = np.load(embedding_path, allow_pickle=True)

        # Handle both old array format and new dictionary format
        if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object:
            try:
                data_dict = loaded_data.item()
                if isinstance(data_dict, dict) and 'embeddings' in data_dict:
                    self.embedding = data_dict['embeddings']
                    self.asins = data_dict.get('asins', None)
                    print(f"[EmbDatasetOne] Loaded {dataset} from dict format: {len(self.embedding)} items")
                    if self.asins:
                        print(f"[EmbDatasetOne] ASIN information available: {len(self.asins)} ASINs")
                else:
                    self.embedding = loaded_data
                    self.asins = None
                    print(f"[EmbDatasetOne] Loaded {dataset} from plain array format")
            except (ValueError, AttributeError):
                self.embedding = loaded_data
                self.asins = None
                print(f"[EmbDatasetOne] Loaded {dataset} from plain array format")
        else:
            self.embedding = loaded_data
            self.asins = None
            print(f"[EmbDatasetOne] Loaded {dataset} from plain array format")

        self.dim = self.embedding.shape[-1]

        self.data_count = self.embedding.shape[0]

        print(self.embedding.shape)

    def __getitem__(self, index):
        emb = self.embedding[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embedding)
