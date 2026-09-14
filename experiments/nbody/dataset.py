import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class NBodyDataset(torch.utils.data.Dataset):
    """N-body prediction dataset.

    Loads data generated with generate_nbody_dataset.py from disk.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the npz file with the dataset to be loaded.
    subsample : None or float
        If not None, defines the fraction of the dataset to be used. For instance, `subsample=0.1`
        uses just 10% of the samples in the dataset.
    keep_trajectories : bool
        Whether to keep the full particle trajectories in the dataset. They are neither needed
        for training nor evaluation, but can be useful for visualization.
    """

    def __init__(
        self,
        filename,
        subsample=None,
        keep_trajectories=False,
        group_size=None,
        group_shuffle=False,
        seed=0,
    ):
        super().__init__()
        self.x, self.y, self.trajectories = self._load_data(
            filename,
            subsample,
            keep_trajectories=keep_trajectories,
            group_size=group_size,
            group_shuffle=group_shuffle,
            seed=seed,
        )

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """Returns the `idx`-th sample from the dataset."""
        return self.x[idx], self.y[idx]

    @staticmethod
    def _load_data(
        filename,
        subsample=None,
        keep_trajectories=False,
        group_size=None,
        group_shuffle=False,
        seed=0,
    ):
        """Loads data from file and converts to input and output tensors."""
        # Load data from file
        npz = np.load(filename, "r")
        m, x_initial, v_initial, x_final = (
            npz["m"],
            npz["x_initial"],
            npz["v_initial"],
            npz["x_final"],
        )

        # Convert to tensors
        m = torch.from_numpy(m).to(torch.float32).unsqueeze(2)
        x_initial = torch.from_numpy(x_initial).to(torch.float32)
        v_initial = torch.from_numpy(v_initial).to(torch.float32)
        x_final = torch.from_numpy(x_final).to(torch.float32)

        # Concatenate into inputs and outputs
        x = torch.cat((m, x_initial, v_initial), dim=2)  # (batchsize, num_objects, 7)
        y = x_final  # (batchsize, num_objects, 3)

        # Optionally, keep raw trajectories around (for plotting)
        if keep_trajectories:
            trajectories = npz["trajectories"]
        else:
            trajectories = None

        # Subsample
        if subsample is not None and subsample < 1.0:
            n_original = len(x)

            if group_size is not None and group_size > 0:
                n_groups = n_original // group_size
                n_keep_groups = round(subsample * n_groups)
                assert 0 < n_keep_groups <= n_groups

                if group_shuffle:
                    rng = np.random.default_rng(seed)
                    group_indices = rng.permutation(n_groups)[:n_keep_groups]
                else:
                    group_indices = np.arange(n_keep_groups)

                idx = (group_indices[:, None] * group_size + np.arange(group_size)).reshape(-1)
                x = x[idx]
                y = y[idx]
                if trajectories is not None:
                    trajectories = trajectories[idx]
            else:
                n_keep = round(subsample * n_original)
                assert 0 < n_keep <= n_original
                x = x[:n_keep]
                y = y[:n_keep]
                if trajectories is not None:
                    trajectories = trajectories[:n_keep]

        return x, y, trajectories


class SpringNBodyDataset(torch.utils.data.Dataset):
    """N-body prediction dataset with spring constants.

    Extends NBodyDataset
    """

    def __init__(
        self,
        filename,
        subsample=None,
        keep_trajectories=False,
        group_size=None,
        group_shuffle=False,
        seed=0,
    ):
        super().__init__()
        self.x, self.y, self.trajectories, self.adj, self.spring_k = self._load_data(
            filename,
            subsample,
            keep_trajectories=keep_trajectories,
            group_size=group_size,
            group_shuffle=group_shuffle,
            seed=seed,
        )

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """Returns the `idx`-th sample from the dataset."""
        return self.x[idx], self.y[idx], self.adj[idx], self.spring_k[idx]

    @staticmethod
    def _load_data(
        filename,
        subsample=None,
        keep_trajectories=False,
        group_size=None,
        group_shuffle=False,
        seed=0,
    ):
        """Loads data from file and converts to input and output tensors."""
        # Load data from file
        npz = np.load(filename, "r")
        m, x_initial, v_initial, x_final, adj, spring_k = (
            npz["m"],
            npz["x_initial"],
            npz["v_initial"],
            npz["x_final"],
            npz["adjacency"],
            npz["spring_k"],
        )
        # Convert to tensors
        m = torch.from_numpy(m).to(torch.float32).unsqueeze(2)
        x_initial = torch.from_numpy(x_initial).to(torch.float32)
        v_initial = torch.from_numpy(v_initial).to(torch.float32)
        x_final = torch.from_numpy(x_final).to(torch.float32)
        adj = torch.from_numpy(adj).to(torch.float32)
        spring_k = torch.from_numpy(spring_k).to(torch.float32)
        # Concatenate into inputs and outputs
        x = torch.cat((m, x_initial, v_initial), dim=2)  # (batchsize, num_objects, 7)
        y = x_final  # (batchsize, num_objects, 3)
        # Optionally, keep raw trajectories around (for plotting)
        if keep_trajectories:
            trajectories = npz["trajectories"]
        else:
            trajectories = None
        # Subsample
        if subsample is not None and subsample < 1.0:
            n_original = len(x)

            if group_size is not None and group_size > 0:
                n_groups = n_original // group_size
                n_keep_groups = round(subsample * n_groups)
                assert 0 < n_keep_groups <= n_groups

                if group_shuffle:
                    rng = np.random.default_rng(seed)
                    group_indices = rng.permutation(n_groups)[:n_keep_groups]
                else:
                    group_indices = np.arange(n_keep_groups)

                idx = (group_indices[:, None] * group_size + np.arange(group_size)).reshape(-1)
                x = x[idx]
                y = y[idx]
                adj = adj[idx]
                spring_k = spring_k[idx]
                if trajectories is not None:
                    trajectories = trajectories[idx]
            else:
                n_keep = round(subsample * n_original)
                assert 0 < n_keep <= n_original
                x = x[:n_keep]
                y = y[:n_keep]
                adj = adj[:n_keep]
                spring_k = spring_k[:n_keep]
                if trajectories is not None:
                    trajectories = trajectories[:n_keep]
        return x, y, trajectories, adj, spring_k


class SpringNBodyDatasetSparse(Dataset):
    """N-body prediction dataset with spring constants in PyG sparse format.

    Converts dense adjacency matrices to PyG-style edge_index and edge_attr.
    """

    def __init__(
        self,
        filename,
        subsample=None,
        keep_trajectories=False,
        group_size=None,
        group_shuffle=False,
        seed=0,
        root=None,
        transform=None,
        pre_transform=None,
    ):
        self.filename = filename
        self.subsample = subsample
        self.keep_trajectories_flag = keep_trajectories
        self.group_size = group_size
        self.group_shuffle = group_shuffle
        self.seed = seed

        # Load and process data
        self.x, self.y, self.trajectories, self.adj, self.spring_k = self._load_data(
            filename,
            subsample,
            keep_trajectories=keep_trajectories,
            group_size=group_size,
            group_shuffle=group_shuffle,
            seed=seed,
        )

        super().__init__(root, transform, pre_transform)

    def len(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)

    def get(self, idx):
        """Returns the `idx`-th sample as a PyG Data object."""
        # Node features: (num_nodes, 7) with [mass, x_initial, v_initial]
        node_features = self.x[idx]

        # Target positions: (num_nodes, 3)
        target = self.y[idx]

        # Convert dense adjacency to sparse edge_index and edge_attr
        edge_index, edge_attr = self._dense_to_sparse(self.adj[idx], self.spring_k[idx])

        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=target,
        )

        return data

    @staticmethod
    def _dense_to_sparse(adj_matrix, spring_k):
        """Converts dense adjacency matrix to PyG sparse format.

        Args:
            adj_matrix: (num_nodes, num_nodes) adjacency matrix with spring constants
            spring_k: (num_nodes, num_nodes) spring constants matrix
        Returns:
            edge_index: (2, num_edges) tensor of edge connections
            edge_attr: (num_edges, 1) tensor of spring constants
        """
        # Find non-zero entries (edges)
        edge_index = adj_matrix.nonzero(as_tuple=False).t()  # (2, num_edges)

        # Extract edge attributes (spring constants)
        edge_attr = spring_k[edge_index[0], edge_index[1]].unsqueeze(1)  # (num_edges, 1)

        return edge_index, edge_attr

    @staticmethod
    def _load_data(
        filename,
        subsample=None,
        keep_trajectories=False,
        group_size=None,
        group_shuffle=False,
        seed=0,
    ):
        """Loads data from file and converts to input and output tensors."""
        # Load data from file
        npz = np.load(filename, "r")
        m, x_initial, v_initial, x_final, adj, spring_k = (
            npz["m"],
            npz["x_initial"],
            npz["v_initial"],
            npz["x_final"],
            npz["adjacency"],
            npz["spring_k"],
        )

        # Convert to tensors
        m = torch.from_numpy(m).to(torch.float32).unsqueeze(2)
        x_initial = torch.from_numpy(x_initial).to(torch.float32)
        v_initial = torch.from_numpy(v_initial).to(torch.float32)
        x_final = torch.from_numpy(x_final).to(torch.float32)
        adj = torch.from_numpy(adj).to(torch.float32)
        spring_k = torch.from_numpy(spring_k).to(torch.float32)
        # Concatenate into inputs and outputs
        x = torch.cat((m, x_initial, v_initial), dim=2)  # (batchsize, num_objects, 7)
        y = x_final  # (batchsize, num_objects, 3)

        # Optionally, keep raw trajectories around (for plotting)
        if keep_trajectories:
            trajectories = npz["trajectories"]
        else:
            trajectories = None

        # Subsample
        if subsample is not None and subsample < 1.0:
            n_original = len(x)
            if group_size is not None and group_size > 0:
                n_groups = n_original // group_size
                n_keep_groups = round(subsample * n_groups)
                assert 0 < n_keep_groups <= n_groups

                if group_shuffle:
                    rng = np.random.default_rng(seed)
                    group_indices = rng.permutation(n_groups)[:n_keep_groups]
                else:
                    group_indices = np.arange(n_keep_groups)

                idx = (group_indices[:, None] * group_size + np.arange(group_size)).reshape(-1)
                x = x[idx]
                y = y[idx]
                adj = adj[idx]
                if trajectories is not None:
                    trajectories = trajectories[idx]
            else:
                n_keep = round(subsample * n_original)
                assert 0 < n_keep <= n_original
                x = x[:n_keep]
                y = y[:n_keep]
                adj = adj[:n_keep]
                if trajectories is not None:
                    trajectories = trajectories[:n_keep]

        return x, y, trajectories, adj, spring_k
