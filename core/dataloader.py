"""BRACS Dataset loader."""
import os
import h5py
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from glob import glob 
import dgl 

from histocartography.utils import set_graph_on_cuda


IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'int': lambda x: torch.LongTensor(x).to(DEVICE)
}

def h5_to_tensor(h5_path):
    h5_object = h5py.File(h5_path, 'r')
    out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
    return out


class BRACSDataset(Dataset):
    """BRACS dataset."""

    def __init__(
            self,
            cg_path: str = None,
            tg_path: str = None,
            assign_mat_path: str = None,
            load_in_ram: bool = False,
    ):
        """
        BRACS dataset constructor.

        Args:
            cg_path (str, optional): Cell Graph path to a given split (eg, cell_graphs/test/). Defaults to None.
            tg_path (str, optional): Tissue Graph path. Defaults to None.
            assign_mat_path (str, optional): Assignment matrices path. Defaults to None.
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
        """
        super(BRACSDataset, self).__init__()

        assert not (cg_path is None and tg_path is None), "You must provide path to at least 1 modality."

        self.cg_path = cg_path
        self.tg_path = tg_path
        self.assign_mat_path = assign_mat_path
        self.load_in_ram = load_in_ram

        if cg_path is not None:
            self._load_cg()

        if tg_path is not None:
            self._load_tg()

        if assign_mat_path is not None:
            self._load_assign_mat()

    def _load_cg(self):
        """
        Load cell graphs
        """
        self.cg_fnames = glob(os.path.join(self.cg_path, '*.bin'))
        self.cg_fnames.sort()
        self.num_cg = len(self.cg_fnames)
        if self.load_in_ram:
            cell_graphs = [load_graphs(os.path.join(self.cg_path, fname)) for fname in self.cg_fnames]
            self.cell_graphs = [entry[0][0] for entry in cell_graphs]
            self.cell_graph_labels = [entry[1]['label'].item() for entry in cell_graphs]

    def _load_tg(self):
        """
        Load tissue graphs
        """
        self.tg_fnames = glob(os.path.join(self.tg_path, '*.bin'))
        self.tg_fnames.sort()
        self.num_tg = len(self.tg_fnames)
        if self.load_in_ram:
            tissue_graphs = [load_graphs(os.path.join(self.tg_path, fname)) for fname in self.tg_fnames]
            self.tissue_graphs = [entry[0][0] for entry in tissue_graphs]
            self.tissue_graph_labels = [entry[1]['label'].item() for entry in tissue_graphs]

    def _load_assign_mat(self):
        """
        Load assignment matrices 
        """
        self.assign_fnames = glob(os.path.join(self.assign_mat_path, '*.h5'))
        self.assign_fnames.sort()
        self.num_assign_mat = len(self.assign_fnames)
        if self.load_in_ram:
            self.assign_matrices = [
                h5_to_tensor(os.path.join(self.assign_mat_path, fname)).float().t()
                    for fname in self.assign_fnames
            ]

    def __getitem__(self, index):
        """
        Get an example.
        Args:
            index (int): index of the example.
        """

        # 1. HACT configuration
        if hasattr(self, 'num_tg') and hasattr(self, 'num_cg'):
            if self.load_in_ram:
                cg = self.cell_graphs[index]
                tg = self.tissue_graphs[index]
                assign_mat = self.assign_matrices[index]
                assert self.cell_graph_labels[index] == self.tissue_graph_labels[index], "The CG and TG are not the same. There was an issue while creating HACT."
                label = self.cell_graph_labels[index]
            else:
                cg, label = load_graphs(self.cg_fnames[index])
                cg = cg[0]
                label = label['label'].item()
                tg, _ = load_graphs(self.tg_fnames[index])
                tg = tg[0]
                assign_mat = h5_to_tensor(self.assign_fnames[index]).float().t()

            cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg
            assign_mat = assign_mat.cuda() if IS_CUDA else assign_mat

            return cg, tg, assign_mat, label

        # 2. TG-GNN configuration 
        elif hasattr(self, 'num_tg'):
            if self.load_in_ram:
                tg = self.tissue_graphs[index]
                label = self.tissue_graph_labels[index]
            else:
                tg, label = load_graphs(self.tg_fnames[index])
                label = label['label'].item()
                tg = tg[0]
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg
            return tg, label

        # 3. CG-GNN configuration 
        else:
            if self.load_in_ram:
                cg = self.cell_graphs[index]
                label = self.cell_graph_labels[index]
            else:
                cg, label = load_graphs(self.cg_fnames[index])
                label = label['label'].item()
                cg = cg[0]
            cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            return cg, label

    def __len__(self):
        """Return the number of samples in the BRACS dataset."""
        if hasattr(self, 'num_cg'):
            return self.num_cg
        else:
            return self.num_tg


def collate(batch):
    """
    Collate a batch.
    Args:
        batch (torch.tensor): a batch of examples.
    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """
    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    # collate the data
    num_modalities = len(batch[0])  # should 2 if CG or TG processing or 4 if HACT
    batch = tuple([collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__)
                  for mod_id in range(num_modalities)])

    return batch


def make_data_loader(
        batch_size,
        shuffle=True,
        num_workers=0,
        **kwargs
    ):
    """
    Create a BRACS data loader.
    """

    dataset = BRACSDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate
        )

    return dataloader