import torch
import itertools
import numpy as np
import torchvision
import torchvision.transforms as transforms


class BASDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, img_dim):
        self.data = self.generate_bas_dataset_equal_prob(n_samples, img_dim).float().unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0
    
    # Function to generate all unique valid Bars-and-Stripes patterns
    def generate_all_unique_bas_patterns(self, img_dim):
        """
        Generates all unique valid Bars-and-Stripes patterns for img_dim x img_dim,
        ensuring that 'all 0' and 'all 1' patterns are not duplicated.
        """
        patterns = set()  # Use a set to avoid duplicates

        # Generate vertical bars (valid): Every column is the same
        for column in itertools.product([0, 1], repeat=img_dim):
            vertical_bar = tuple(np.tile(np.array(column).reshape(img_dim, 1), (1, img_dim)).flatten())
            patterns.add(vertical_bar)

        # Generate horizontal stripes (valid): Every row is the same
        for row in itertools.product([0, 1], repeat=img_dim):
            horizontal_stripe = tuple(np.tile(np.array(row).reshape(1, img_dim), (img_dim, 1)).flatten())
            patterns.add(horizontal_stripe)

        self.number_of_patterns = len(patterns)

        # Convert patterns back to list of NumPy arrays
        return [np.array(pattern).reshape(img_dim, img_dim) for pattern in patterns]

    # Function to sample uniformly from the set of valid unique patterns
    def generate_bas_dataset_equal_prob(self, n_samples, img_dim):
        """
        Generates a Bars-and-Stripes dataset with n_samples, ensuring equal probability
        for all unique valid configurations.
        """
        # Get all unique valid patterns
        valid_patterns = self.generate_all_unique_bas_patterns(img_dim)

        # Randomly sample from the set of valid patterns
        sampled_data = [valid_patterns[np.random.randint(0, len(valid_patterns))] for _ in range(n_samples)]

        # Convert sampled_data (list of NumPy arrays) into a NumPy array
        sampled_data_np = np.array(sampled_data)
        
        # Convert the NumPy array to a PyTorch tensor
        sampled_data_tensor = torch.tensor(sampled_data_np, dtype=torch.float32)

        return sampled_data_tensor
    
    def get_number_of_patterns(self):
        return self.number_of_patterns



class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        
        self.data = torchvision.datasets.MNIST(
            root='./data'
            , train=True
            , transform=transform
            , download=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0
