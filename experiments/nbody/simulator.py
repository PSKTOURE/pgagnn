from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import special_ortho_group

from src.utils import seed_everything


def sample_uniform_in_circle(n, min_radius=0.0, max_radius=1.0):
    """Samples uniformly in a 2D circle using batched rejection sampling."""

    assert 0.0 <= min_radius < max_radius, "Inconsistent inputs to sample_uniform_in_circle"

    mask = None
    samples = None

    while samples is None or np.sum(mask) > 0:
        new_samples = max_radius * np.random.uniform(low=-1, high=1, size=(n, 2))

        if samples is None:
            samples = new_samples
        else:
            samples = (1 - mask) * samples + mask * new_samples

        r2 = np.sum(samples**2, axis=-1)
        mask = np.logical_or((r2 < min_radius**2), (r2 > max_radius**2))[:, np.newaxis]

    return samples


def sample_log_uniform(min_, max_, size):
    """Samples log-uniformly from (min_, max_)."""
    if isinstance(size, tuple):
        n = int(np.product(size))
    else:
        n = size

    log_x = np.random.rand(n) * (np.log(max_) - np.log(min_)) + np.log(min_)
    x = np.exp(log_x)

    if isinstance(size, tuple):
        x = x.reshape(size)

    return x


class BaseSimulator:
    """Base class for physics simulators with shared functionality.

    This class provides common functionality for:
    - Spatial transformations (shift and rotate)
    - Data augmentation
    - Trajectory simulation loop
    - Outlier removal

    Subclasses must implement:
    - _sample_initial_state(): Create initial masses, positions, and velocities
    - _compute_accelerations(): Compute accelerations based on physics model
    """

    def __init__(
        self,
        time=0.1,
        time_steps=100,
        shift_std=20.0,
        ood_shift=200.0,
        outlier_threshold=2.0,
        augmentation_shifts=None,
    ):
        """Initialize base simulator parameters.

        Parameters
        ----------
        time : float
            Overall time period between initial and final state.
        time_steps : int
            Number of time steps for integration.
        shift_std : float
            Noise scale for spatial translation.
        ood_shift : float
            Noise scale for spatial translation for domain-shifted samples.
        outlier_threshold : float
            Threshold for rejection of outlier samples.
        augmentation_shifts : list of tuples or None
            List of explicit position shifts [x, y, z] for data augmentation.
        """
        self.time = time
        self.delta_t = time / time_steps
        self.time_steps = time_steps
        self.shift_std = shift_std
        self.ood_shift = ood_shift
        self.outlier_threshold = outlier_threshold
        self.augmentation_shifts = augmentation_shifts if augmentation_shifts is not None else []

    def sample(self, num_samples, num_objects=5, domain_shift=False, use_augmentation=False):
        """Base sampling method with common pipeline.

        Override this method in subclasses if additional return values are needed.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        num_objects : int
            Number of objects per sample.
        domain_shift : bool
            Whether to apply domain shift (spatial translation).
        use_augmentation : bool
            Whether to apply data augmentation.

        Returns
        -------
        Tuple of arrays containing simulation results.
        """
        # Sample a bit more initially for outlier removal
        num_candidates = round(1.1 * num_samples)

        # Sample initial state
        masses, x_initial, v_initial, extra_data = self._sample_initial_state(num_candidates, num_objects)

        # Apply transformations
        masses, x_initial, v_initial = self._shift_and_rotate(
            masses, x_initial, v_initial, domain_shift=domain_shift
        )

        # Simulate
        x_final, trajectory = self._simulate(masses, x_initial, v_initial, extra_data)

        # Remove outliers
        masses, x_initial, v_initial, x_final, trajectory, extra_data = self._remove_outliers(
            masses, x_initial, v_initial, x_final, trajectory, extra_data
        )

        # Cut to requested number
        masses, x_initial, v_initial, x_final, trajectory, extra_data = self._truncate_to_num_samples(
            num_samples, masses, x_initial, v_initial, x_final, trajectory, extra_data
        )

        # Apply augmentation
        if use_augmentation and len(self.augmentation_shifts) > 0:
            masses, x_initial, v_initial, x_final, trajectory, extra_data = self._apply_augmentation(
                masses, x_initial, v_initial, x_final, trajectory, extra_data
            )

        return self._format_output(masses, x_initial, v_initial, x_final, trajectory, extra_data)

    def _sample_initial_state(self, num_candidates, num_objects):
        """Sample initial masses, positions, velocities, and any extra data.

        Must be implemented by subclasses.

        Returns
        -------
        masses : np.ndarray
            Shape (num_candidates, num_objects)
        x_initial : np.ndarray
            Shape (num_candidates, num_objects, 3)
        v_initial : np.ndarray
            Shape (num_candidates, num_objects, 3)
        extra_data : dict
            Any additional data needed by the specific simulator
        """
        raise NotImplementedError("Subclasses must implement _sample_initial_state")

    def _shift_and_rotate(self, m, x, v, domain_shift=False):
        """Performs random E(3) transformations and permutations.

        For domain-shifted (OOD) samples, an additional random reflection
        (improper orthogonal transform, det = -1) is applied on top of the
        usual rotation. This pushes OOD samples outside the SO(3) manifold
        the model is trained on, so the domain-shift test checks O(3)
        generalization (rotation + reflection + translation), not just a
        large translation.
        """
        batchsize, num_objects, _ = x.shape

        # Permutations over objects
        for i in range(batchsize):
            perm = np.random.permutation(num_objects)
            m[i] = m[i][perm]
            x[i] = x[i][perm, :]
            v[i] = v[i][perm, :]

        # Rotations from Haar measure (proper, det = +1)
        rotations = special_ortho_group(3).rvs(size=batchsize).reshape(batchsize, 1, 3, 3)
        x = np.einsum("bnij,bnj->bni", rotations, x)
        v = np.einsum("bnij,bnj->bni", rotations, v)

        # OOD reflection: compose a Haar-random rotation with a fixed reflection
        # to get a uniformly random improper transform (det = -1).
        if domain_shift:
            reflect_rotations = special_ortho_group(3).rvs(size=batchsize).reshape(batchsize, 3, 3)
            flip = np.diag([-1.0, 1.0, 1.0])
            reflections = np.einsum("bij,jk->bik", reflect_rotations, flip).reshape(batchsize, 1, 3, 3)
            x = np.einsum("bnij,bnj->bni", reflections, x)
            v = np.einsum("bnij,bnj->bni", reflections, v)

        # Translations
        shifts = np.random.normal(scale=self.shift_std, size=(batchsize, 1, 3))
        x = x + shifts

        # OOD shift
        if domain_shift:
            shifts = np.array([self.ood_shift, 0, 0]).reshape((1, 1, 3))
            x = x + shifts

        return m, x, v

    def _simulate(self, m, x_initial, v_initial, extra_data):
        """Evolves the system using the physics model."""
        x, v = x_initial, v_initial
        trajectory = [x_initial]

        for _ in range(self.time_steps):
            a = self._compute_accelerations(m, x, extra_data)
            v = v + self.delta_t * a
            x = x + self.delta_t * v
            trajectory.append(x)

        return x, np.array(trajectory).transpose([1, 2, 0, 3])

    def _compute_accelerations(self, m, x, extra_data):
        """Compute accelerations based on physics model.

        Must be implemented by subclasses.

        Parameters
        ----------
        m : np.ndarray
            Masses, shape (batchsize, num_objects)
        x : np.ndarray
            Positions, shape (batchsize, num_objects, 3)
        extra_data : dict
            Additional data specific to the physics model

        Returns
        -------
        accelerations : np.ndarray
            Shape (batchsize, num_objects, 3)
        """
        raise NotImplementedError("Subclasses must implement _compute_accelerations")

    def _remove_outliers(self, masses, x_initial, v_initial, x_final, trajectory, extra_data):
        """Remove samples where objects moved too far."""
        max_distance = np.max(np.linalg.norm(x_final - x_initial, axis=-1), axis=-1)
        mask = max_distance <= self.outlier_threshold

        masses = masses[mask]
        x_initial = x_initial[mask]
        v_initial = v_initial[mask]
        x_final = x_final[mask]
        trajectory = trajectory[mask]

        # Filter extra_data
        if extra_data:
            extra_data = {k: v[mask] for k, v in extra_data.items()}

        return masses, x_initial, v_initial, x_final, trajectory, extra_data

    def _truncate_to_num_samples(
        self, num_samples, masses, x_initial, v_initial, x_final, trajectory, extra_data
    ):
        """Truncate all arrays to exactly num_samples."""
        masses = masses[:num_samples]
        x_initial = x_initial[:num_samples]
        v_initial = v_initial[:num_samples]
        x_final = x_final[:num_samples]
        trajectory = trajectory[:num_samples]

        if extra_data:
            extra_data = {k: v[:num_samples] for k, v in extra_data.items()}

        return masses, x_initial, v_initial, x_final, trajectory, extra_data

    def _apply_augmentation(self, masses, x_initial, v_initial, x_final, trajectory, extra_data):
        """Creates augmented copies with position shifts.

        For each sample, creates copies with explicit shifts from self.augmentation_shifts.
        The original sample (no additional shift) is also included.
        """
        num_samples = masses.shape[0]
        num_augmentations = len(self.augmentation_shifts) + 1  # +1 for original

        # Prepare output arrays
        masses_aug = np.repeat(masses, num_augmentations, axis=0)
        x_initial_aug = np.zeros((num_samples * num_augmentations,) + x_initial.shape[1:])
        v_initial_aug = np.repeat(v_initial, num_augmentations, axis=0)
        x_final_aug = np.zeros((num_samples * num_augmentations,) + x_final.shape[1:])
        trajectory_aug = np.zeros((num_samples * num_augmentations,) + trajectory.shape[1:])

        # Apply shifts
        for i in range(num_samples):
            # Original (no additional shift)
            idx = i * num_augmentations
            x_initial_aug[idx] = x_initial[i]
            x_final_aug[idx] = x_final[i]
            trajectory_aug[idx] = trajectory[i]

            # Augmented copies with explicit shifts
            for j, shift in enumerate(self.augmentation_shifts, start=1):
                idx = i * num_augmentations + j
                shift_vector = np.array(shift).reshape(1, 3)
                x_initial_aug[idx] = x_initial[i] + shift_vector
                x_final_aug[idx] = x_final[i] + shift_vector
                trajectory_aug[idx] = trajectory[i] + shift_vector.reshape(1, 1, 3)

        # Augment extra_data
        if extra_data:
            extra_data_aug = {k: np.repeat(v, num_augmentations, axis=0) for k, v in extra_data.items()}
        else:
            extra_data_aug = extra_data

        return masses_aug, x_initial_aug, v_initial_aug, x_final_aug, trajectory_aug, extra_data_aug

    def _format_output(self, masses, x_initial, v_initial, x_final, trajectory, extra_data):
        """Format the output. Override in subclasses to add extra return values."""
        return masses, x_initial, v_initial, x_final, trajectory


class NBodySimulator(BaseSimulator):
    """Simulator for the n-body dataset.

    Each sample consists of positions of n particles (1 star and n - 1 planets),
    both before and after evolution of 100 time steps. Particle masses are also included.

    The data is generated as follows:
    1. Masses for star and planets are sampled.
    2. Initial planet positions are sampled around origin in x-y plane. Star is at (0, 0).
    3. Initial planet velocities are given by velocity of stable circular orbit, plus noise.
       Star is initially at rest.
    4. Initial state is translated and rotated to arbitrary position/orientation.
    5. Final state is computed using Newton's equations and Euler integration.
    6. Samples where objects moved too far are removed as outliers.

    We use units where G = 1.

    Parameters
    ----------
    star_mass_range : tuple of float
        Minimum and maximum mass for the star.
    planet_mass_range : tuple of float
        Minimum and maximum mass for the planets.
    radius_range : tuple of float
        Minimum and maximum initial distance of planets from star.
    vel_std : float
        Noise scale for initial planet velocity.
    **kwargs : dict
        Additional arguments passed to BaseSimulator.
    """

    def __init__(
        self,
        star_mass_range=(1.0, 10.0),
        planet_mass_range=(0.01, 0.1),
        radius_range=(0.1, 1.0),
        vel_std=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.star_mass_range = star_mass_range
        self.planet_mass_range = planet_mass_range
        self.radius_range = radius_range
        self.vel_std = vel_std

    def _sample_initial_state(self, num_candidates, num_objects):
        """Sample initial state for n-body problem."""
        # Sample planet masses
        star_mass = sample_log_uniform(*self.star_mass_range, size=(num_candidates, 1))
        planet_masses = sample_log_uniform(*self.planet_mass_range, size=(num_candidates, num_objects))
        masses = np.concatenate((star_mass, planet_masses), axis=1)

        # Sample initial positions in x-y plane around origin
        planet_pos = np.zeros((num_candidates, num_objects, 3))
        planet_pos[..., :2] = sample_uniform_in_circle(
            num_candidates * num_objects,
            min_radius=self.radius_range[0],
            max_radius=self.radius_range[1],
        ).reshape((num_candidates, num_objects, 2))
        x_initial = np.concatenate((np.zeros((num_candidates, 1, 3)), planet_pos), axis=1)

        # Sample initial velocities (stable circular orbits + noise)
        planet_vel = self._sample_planet_velocities(star_mass, planet_pos)
        v_initial = np.concatenate((np.zeros((num_candidates, 1, 3)), planet_vel), axis=1)

        return masses, x_initial, v_initial, {}  # No extra data

    def _sample_planet_velocities(self, star_mass, x):
        """Samples planet velocities around stable circular orbits."""
        batchsize, num_objects, _ = x.shape

        # Rotation plane direction (random clockwise/counterclockwise)
        orientation = np.zeros((batchsize, num_objects, 3))
        orientation[:, :, 2] = np.random.choice(a=[-1.0, 1.0], size=(batchsize, num_objects), p=[0.5, 0.5])

        # Compute stable velocities
        star_mass = star_mass[:, :, np.newaxis]  # (batchsize, 1, 1)
        radii = np.linalg.norm(x, axis=-1)[:, :, np.newaxis]
        v_stable = np.cross(orientation, x) * star_mass**0.5 / radii**1.5

        # Add noise
        v = v_stable + np.random.normal(scale=self.vel_std, size=(batchsize, num_objects, 3))

        return v

    def _compute_accelerations(self, m, x, extra_data):
        """Computes accelerations using Newtonian gravity."""
        batchsize, num_objects, _ = x.shape
        mm = m.reshape((batchsize, 1, num_objects, 1)) * m.reshape(
            (batchsize, num_objects, 1, 1)
        )  # (b, n, n, 1)

        distance_vectors = x.reshape((batchsize, 1, num_objects, 3)) - x.reshape(
            (batchsize, num_objects, 1, 3)
        )  # (b, n, n, 3)
        distances = np.linalg.norm(distance_vectors, axis=-1)[:, :, :, np.newaxis]  # (b, n, n, 1)
        distances[np.abs(distances) < 1e-9] = 1.0

        forces = distance_vectors * mm / distances**3  # (b, n, n, 3)
        accelerations = np.sum(forces, axis=2) / m.reshape(batchsize, num_objects, 1)  # (b, n, 3)

        return accelerations


class RandomGraphSpringSimulator(BaseSimulator):
    """Simulator with spring forces on a sparse, randomly generated graph.

    Unlike gravity simulator, this generates random masses connected by springs.
    All objects are treated equally (no star/planet distinction).

    Edges are sampled with probability `edge_prob` (Erdos-Renyi). Each edge gets
    a spring constant sampled uniformly from `spring_constant_range`.

    Parameters
    ----------
    edge_prob : float
        Probability of edge between any two nodes.
    spring_constant_range : tuple of float
        Minimum and maximum spring constant.
    mass_range : tuple of float
        Minimum and maximum mass for objects.
    position_range : tuple of float
        Minimum and maximum initial position coordinates.
    velocity_std : float
        Standard deviation for initial velocities.
    symmetric : bool
        Whether to make adjacency matrix symmetric.
    **kwargs : dict
        Additional arguments passed to BaseSimulator.
    """

    def __init__(
        self,
        edge_prob=0.3,
        spring_constant_range=(0.5, 2.0),
        mass_range=(0.5, 2.0),
        position_range=(-5.0, 5.0),
        velocity_std=0.5,
        symmetric=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.edge_prob = edge_prob
        self.spring_constant_range = spring_constant_range
        self.mass_range = mass_range
        self.position_range = position_range
        self.velocity_std = velocity_std
        self.symmetric = symmetric

    def _sample_initial_state(self, num_candidates, num_objects):
        """Sample initial state for spring graph problem."""
        # Sample random masses for all objects
        masses = sample_log_uniform(*self.mass_range, size=(num_candidates, num_objects))

        # Sample random initial positions in 3D
        pos_min, pos_max = self.position_range
        x_initial = np.random.uniform(low=pos_min, high=pos_max, size=(num_candidates, num_objects, 3))

        # Sample random initial velocities
        v_initial = np.random.normal(loc=0.0, scale=self.velocity_std, size=(num_candidates, num_objects, 3))

        # Sample graph structure and spring constants
        adj, spring_k = self._sample_graph(batchsize=num_candidates, num_objects=num_objects)

        # Store in extra_data
        extra_data = {"adjacency": adj, "spring_k": spring_k}

        return masses, x_initial, v_initial, extra_data

    def _sample_graph(self, batchsize, num_objects):
        """Samples undirected random graph and spring constants."""
        adj = np.random.rand(batchsize, num_objects, num_objects) < self.edge_prob
        idx = np.arange(num_objects)
        adj[:, idx, idx] = False  # no self loops

        if self.symmetric:
            adj = np.logical_or(adj, np.transpose(adj, axes=(0, 2, 1)))

        k_min, k_max = self.spring_constant_range
        spring_k = np.random.uniform(k_min, k_max, size=(batchsize, num_objects, num_objects))
        if self.symmetric:
            spring_k = 0.5 * (spring_k + np.transpose(spring_k, axes=(0, 2, 1)))

        spring_k = spring_k * adj
        return adj.astype(np.float32), spring_k.astype(np.float32)

    def _compute_accelerations(self, m, x, extra_data):
        """Compute accelerations using spring forces.

        Spring forces only on existing edges: F = -k * distance_vector
        """
        batchsize, num_objects, _ = x.shape
        adjacency = extra_data["adjacency"]
        spring_k = extra_data["spring_k"]

        distance_vectors = x.reshape((batchsize, 1, num_objects, 3)) - x.reshape(
            (batchsize, num_objects, 1, 3)
        )  # (b, n, n, 3)

        edge_mask = adjacency[:, :, :, None]
        forces = distance_vectors * spring_k[:, :, :, None] * edge_mask  # (b, n, n, 3)
        accelerations = np.sum(forces, axis=2) / m.reshape(batchsize, num_objects, 1)

        return accelerations

    def _format_output(self, masses, x_initial, v_initial, x_final, trajectory, extra_data):
        """Include adjacency and spring_k in output."""
        return (
            masses,
            x_initial,
            v_initial,
            x_final,
            trajectory,
            extra_data["adjacency"],
            extra_data["spring_k"],
        )


augmentation_shifts = [
    [50, 0, 0],
    [100, 0, 0],
    [150, 0, 0],
    [200, 0, 0],
    [-50, 0, 0],
    [-100, 0, 0],
    [-150, 0, 0],
    [-200, 0, 0],
    [250, 0, 0],
    [-250, 0, 0],
]


def compute_statistics(path: Path):
    """Computes and prints statistics of the generated dataset."""
    print("Dataset Statistics:")
    print("=" * 60)

    # Load all splits at once
    splits = ["train_augmented", "val", "test", "test_ood"]
    data_dict = {split: np.load(path / f"spring_{split}.npz") for split in splits}

    statistics = {}
    for split in splits:
        data = data_dict[split]
        m = data["m"]
        x_initial = data["x_initial"]
        v_initial = data["v_initial"]
        x_final = data["x_final"]
        num_samples, num_objects, _ = x_initial.shape

        statistics[split] = {
            "num_samples": num_samples,
            "num_objects": num_objects,
            "mass_range": (m.min(), m.max()),
            "initial_position_range": (x_initial.min(), x_initial.max()),
            "final_position_range": (x_final.min(), x_final.max()),
            "initial_velocity_range": (v_initial.min(), v_initial.max()),
            "pos_mean_initial": x_initial.mean(),
            "pos_mean_final": x_final.mean(),
            "pos_std_initial": x_initial.std(),
            "pos_std_final": x_final.std(),
        }

        print(f"{split.upper()}:")
        print(f"  → Number of samples: {num_samples}")
        print(f"  → Number of objects per sample: {num_objects}")
        print(f"  → Mass range: [{m.min():.3f}, {m.max():.3f}]")
        print(f"  → Initial position range: [{x_initial.min():.3f}, {x_initial.max():.3f}]")
        print(f"  → Final position range: [{x_final.min():.3f}, {x_final.max():.3f}]")
        print(f"  → Initial velocity range: [{v_initial.min():.3f}, {v_initial.max():.3f}]")
        print(f"  → Mean initial position: {x_initial.mean():.3f}, Mean final position: {x_final.mean():.3f}")
        print(f"  → Std initial position: {x_initial.std():.3f}, Std final position: {x_final.std():.3f}")
        print("-" * 60)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Distributions Across Splits", fontsize=16, fontweight="bold")

    # All splits for plotting (including 'train')
    all_splits = ["train", "train_augmented", "val", "test", "test_ood"]
    colors = ["purple", "blue", "green", "orange", "red"]

    # Helper function to plot histograms
    def plot_histogram(ax, data_key, title, xlabel, ylabel, log=False, density=False):
        all_data = []
        for split in all_splits:
            data = np.load(path / f"spring_{split}.npz")
            all_data.append(data[data_key][..., 0].flatten())
        all_data = np.concatenate(all_data)
        bins = np.histogram_bin_edges(all_data, bins=100)

        for z, (split, color) in enumerate(zip(all_splits, colors), start=1):
            data = np.load(path / f"spring_{split}.npz")
            ax.hist(
                data[data_key][..., 0].flatten(),
                bins=bins,
                histtype="bar",
                alpha=0.35,
                edgecolor=color,
                linewidth=1.0,
                label=split,
                color=color,
                density=density,
                log=log,
                zorder=z,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot each histogram
    plot_histogram(
        axes[0, 0],
        "x_initial",
        "Initial Positions Distribution in x-dimension",
        "Position Value",
        "Frequency",
        log=True,
    )
    plot_histogram(
        axes[0, 1],
        "x_final",
        "Final Positions Distribution in x-dimension",
        "Position Value",
        "Frequency",
        log=True,
    )
    plot_histogram(axes[1, 0], "m", "Mass Distribution", "Mass Value", "Frequency", density=True)
    plot_histogram(
        axes[1, 1], "v_initial", "Initial Velocity Distribution", "Velocity Value", "Frequency", density=True
    )

    plt.tight_layout()
    plt.savefig(path / "dataset_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()

    return statistics


def generate_dataset(path: Path, type: str = "spring", edge_prob=0.3):
    """Generates a spring-graph dataset for GNN experiments."""
    if type == "nbody":
        simulator = NBodySimulator(augmentation_shifts=augmentation_shifts)
    elif type == "spring":
        simulator = RandomGraphSpringSimulator(edge_prob=edge_prob, augmentation_shifts=augmentation_shifts)

    # Helper function to avoid repetition
    def save_dataset(filename, num_samples, num_objects, use_augmentation=False, domain_shift=False):
        result = simulator.sample(
            num_samples=num_samples,
            num_objects=num_objects,
            domain_shift=domain_shift,
            use_augmentation=use_augmentation,
        )

        # Unpack based on simulator type
        if type == "spring":
            m, x_initial, v_initial, x_final, _trajectory, adj, spring_k = result
        else:
            m, x_initial, v_initial, x_final, _trajectory = result
            adj = None
            spring_k = None

        print(f"Saving {filename}: {m.shape[0]} samples")

        save_dict = {
            "m": m,
            "x_initial": x_initial,
            "v_initial": v_initial,
            "x_final": x_final,
        }

        if adj is not None:
            save_dict["adjacency"] = adj
            save_dict["spring_k"] = spring_k

        np.savez(path / filename, **save_dict)

    # Training dataset
    prefix = "nbody" if type == "nbody" else "spring"
    save_dataset(f"{prefix}_train.npz", num_samples=100000, num_objects=15)
    save_dataset(
        f"{prefix}_train_augmented.npz",
        num_samples=ceil(100000 / (1 + len(simulator.augmentation_shifts))),
        num_objects=15,
        use_augmentation=True,
    )

    # Validation datasets
    save_dataset(f"{prefix}_val.npz", num_samples=5000, num_objects=15)
    save_dataset(
        f"{prefix}_val_augmented.npz",
        num_samples=ceil(5000 / (1 + len(simulator.augmentation_shifts))),
        num_objects=15,
        use_augmentation=True,
    )

    # Test datasets
    save_dataset(f"{prefix}_test.npz", num_samples=5000, num_objects=20)
    save_dataset(
        f"{prefix}_test_ood.npz",
        num_samples=5000,
        num_objects=15,
        use_augmentation=False,
        domain_shift=True,
    )


if __name__ == "__main__":
    seed_everything(42)  # Set a fixed seed for reproducibility
    # Dataset path
    dataset_path = Path(__file__).resolve().parents[2] / "datasets" / "nbody"
    print(f"Generating n-body dataset in {dataset_path}")
    dataset_path.mkdir(parents=True, exist_ok=True)
    generate_dataset(dataset_path, type="spring", edge_prob=0.4)
    print("Dataset generation completed.")
    generate_dataset(dataset_path, type="nbody")
    print("N-body dataset generation completed.")
    # statistics = compute_statistics(dataset_path)
