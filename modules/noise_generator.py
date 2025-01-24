import torch
import numpy as np
import scipy.stats as stats

import sys

# Absolute path to the directory you want to add
path = "/home/ffb/projetos_individuais/PhFC_simulator"

# Add the path if it's not already in sys.path
if path not in sys.path:
    sys.path.append(path)
import PhotonicFrequencyCircuitSimulator as pfcs


class NoiseGenerator:
    """
    Class to generate noise vectors based on different probability distributions
    and perform various statistical and randomness tests on the generated noise.
    """

    def __init__(self, config, device):
        """
        Initializes the NoiseGenerator with configuration and device.

        Parameters:
            config (object): Configuration containing noise type and parameters.
            device (torch.device): The device (CPU/GPU) to generate the noise on.
        """
        self.config = config
        self.device = device

    def generate_noise(self):
        """
        Generates noise based on the specified distribution in the config.

        Returns:
            torch.Tensor: Generated noise vector.
        """
        # Normal (Gaussian) distribution
        if self.config.noise_type == 'normal':
            z = torch.randn(self.config.batch_size, self.config.latent_size).to(self.device) * self.config.noise_std + self.config.noise_mean

        # Uniform distribution
        elif self.config.noise_type == 'uniform':
            z = torch.rand(self.config.batch_size, self.config.latent_size).to(self.device) * (self.config.noise_max - self.config.noise_min) + self.config.noise_min

        # Exponential distribution
        elif self.config.noise_type == 'exponential':
            z = torch.distributions.Exponential(self.config.noise_lambda).sample((self.config.batch_size, self.config.latent_size)).to(self.device)

        # Log-normal distribution
        elif self.config.noise_type == 'lognormal':
            z = torch.distributions.LogNormal(self.config.noise_mean, self.config.noise_std).sample((self.config.batch_size, self.config.latent_size)).to(self.device)

        # Gamma distribution
        elif self.config.noise_type == 'gamma':
            z = torch.distributions.Gamma(self.config.noise_alpha, self.config.noise_beta).sample((self.config.batch_size, self.config.latent_size)).to(self.device)

        # Poisson distribution
        elif self.config.noise_type == 'poisson':
            z = torch.poisson(torch.full((self.config.batch_size, self.config.latent_size), self.config.noise_lambda)).to(self.device)

        # Binary random noise (0s and 1s)
        elif self.config.noise_type == 'random_binary':
            z = torch.randint(0, 2, (self.config.batch_size, self.config.latent_size)).float().to(self.device)

        elif self.config.noise_type == 'pfc_sim':

            prob=[]
            while len(prob)!=self.config.d:
                testCircuit = pfcs.PhotonicFrequencyCircuitSimulator(l=self.config.l, d=self.config.d)

                randomPhaseParameters = np.random.uniform(-np.pi, +np.pi, self.config.l*self.config.d).reshape(self.config.l, self.config.d)

                testCircuit.set_WSbins(randomPhaseParameters) 
                testCircuit._propagate()

                freq, signal = testCircuit.get_outputSpectrum()
                prob = testCircuit.get_peaks_prob()

                N = self.config.d #len(prob) # number of detectors = d (number of modes) ?
                m = (self.config.batch_size,self.config.latent_size) # (batch_size, input_size) 

            numbers = np.random.choice(a=N, size=m, p=prob)

            # numbers = numbers / np.max(numbers)

            z = torch.tensor(numbers, dtype=torch.float32).to(self.device)
        
        else:
            # Raise error for unsupported noise types
            raise ValueError(f"Unsupported noise type: {self.config.noise_type}")

        
        return z


    # def generate_noise_photonic_sim(self):

    #     testCircuit = pfcs.PhotonicFrequencyCircuitSimulator(l=self.config.l, d=self.config.d)

    #     randomPhaseParameters = np.random.uniform(-np.pi, +np.pi, self.config.l*self.config.d).reshape(self.config.l, self.config.d)

    #     testCircuit.set_WSbins(randomPhaseParameters) 
    #     testCircuit._propagate()

    #     freq, signal = testCircuit.get_outputSpectrum()
    #     prob = testCircuit.get_peaks_prob()

    #     N = len(prob) # number of detectors = d (number of modes) ?
    #     m = (self.config.batch_size,self.config.latent_size) # (batch_size, input_size) 

    #     numbers = np.random.choice(a=N, size=m, p=prob)

    #     numbers = numbers / np.max(numbers)

    #     z = torch.tensor(numbers, dtype=torch.float32).to(self.device)
    #     return z
























    def monobit_frequency_test(self, noise):
        """
        Monobit Frequency Test: Check if the number of 1's and 0's in the binary noise 
        are approximately equal, indicating randomness.

        Parameters:
            noise (np.ndarray): Binary noise sequence.

        Returns:
            float: p-value from the monobit frequency test.
        """
        n = len(noise)
        count_ones = np.sum(noise)
        S = abs(count_ones - (n - count_ones))
        # Normal distribution approximation to calculate p-value
        p_value = stats.norm.cdf(S / np.sqrt(n))
        return p_value

    def block_frequency_test(self, noise, block_size=128):
        """
        Block Frequency Test: Check if blocks of the binary sequence have 
        an equal number of 1's and 0's, indicating randomness.

        Parameters:
            noise (np.ndarray): Binary noise sequence.
            block_size (int): Size of each block to test.

        Returns:
            float: p-value from the block frequency test.
        """
        num_blocks = len(noise) // block_size
        # Sum the number of 1's in each block
        block_sums = [np.sum(noise[i * block_size: (i + 1) * block_size]) for i in range(num_blocks)]
        block_sums = np.array(block_sums)

        # Chi-squared statistic for block sums
        chi_squared = 4 * block_size * np.sum(((block_sums - block_size / 2) / block_size) ** 2)
        # Calculate p-value from chi-squared distribution
        p_value = stats.chi2.sf(chi_squared, num_blocks)
        return p_value

    def runs_test(self, noise):
        """
        Runs Test: Tests the randomness of a sequence by examining the number of runs 
        (consecutive identical elements) in the binary noise.

        Parameters:
            noise (np.ndarray): Binary noise sequence.

        Returns:
            float: p-value from the runs test.
        """
        n = len(noise)
        count_ones = np.sum(noise)
        count_zeros = n - count_ones

        # Check if the noise sequence is valid for the runs test
        if n <= 1 or count_ones == 0 or count_zeros == 0:
            return 1.0  # Return high p-value (randomness not detectable)

        # Number of runs in the sequence
        runs = 1 + np.sum(noise[1:] != noise[:-1])
        # Expected number of runs
        expected_runs = 2 * count_ones * count_zeros / n + 1

        denominator = n ** 2 * (n - 1)
        if denominator == 0:
            return 1.0  # High p-value for invalid sequence

        # Variance of runs
        variance_runs = 2 * count_ones * count_zeros * (2 * count_ones * count_zeros - n) / denominator
        if variance_runs == 0:
            return 1.0  # High p-value for invalid variance

        # z-score for the runs test
        z = abs(runs - expected_runs) / np.sqrt(variance_runs)
        # Two-tailed test to calculate p-value
        p_value = 2 * stats.norm.cdf(-z)
        return p_value

    def longest_runs_of_ones_test(self, noise, block_size=128):
        """
        Longest Runs of Ones in a Block Test: Checks the length of the longest 
        consecutive sequence of 1's within blocks of the binary noise sequence.

        Parameters:
            noise (np.ndarray): Binary noise sequence.
            block_size (int): Size of each block.

        Returns:
            float: p-value from the longest runs of ones test.
        """
        num_blocks = len(noise) // block_size
        # Find the longest consecutive run of 1's in each block
        longest_runs = [
            np.max(np.diff(np.where(np.concatenate(([0], noise[i * block_size:(i + 1) * block_size], [0])) == 0)))
            for i in range(num_blocks)
        ]
        # p-value from chi-squared distribution
        p_value = stats.chi2.sf(np.sum(longest_runs), num_blocks)
        return p_value

    def noise_metrics(self, noise):
        """
        Calculates statistical metrics and randomness tests for the generated noise.

        Metrics include:
            - Mean
            - Standard Deviation
            - Skewness
            - Kurtosis
            - Entropy
            - Range
            - Randomness Tests: Monobit Frequency, Block Frequency, Runs Test, Longest Runs of Ones

        Parameters:
            noise (torch.Tensor): Generated noise tensor.

        Returns:
            dict: A dictionary containing all calculated metrics.
        """
        # Convert the noise tensor to a NumPy array for easier processing
        noise_np = noise.cpu().numpy()

        # Statistical metrics
        mean = np.mean(noise_np)
        std = np.std(noise_np)
        skewness = np.mean((noise_np - mean) ** 3) / std ** 3
        kurtosis = np.mean((noise_np - mean) ** 4) / std ** 4 - 3
        noise_range = np.max(noise_np) - np.min(noise_np)

        # Calculate entropy
        hist, _ = np.histogram(noise_np, bins=100, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))

        # Convert the noise to binary for randomness tests
        binary_noise = (noise_np > 0.5).astype(int)

        # Perform randomness tests
        p_monobit = self.monobit_frequency_test(binary_noise)
        p_block = self.block_frequency_test(binary_noise)
        p_runs = self.runs_test(binary_noise)
        p_longest_runs = self.longest_runs_of_ones_test(binary_noise)

        # Create a dictionary to hold all calculated metrics
        metrics = {
            "Noise Mean": mean,
            "Noise Std": std,
            "Noise Skewness": skewness,
            "Noise Kurtosis": kurtosis,
            "Noise Range": noise_range,
            "Noise Entropy": entropy,
            "Monobit Frequency Test": p_monobit,
            "Block Frequency Test": p_block,
            "Runs Test": p_runs,
            "Longest Runs of Ones Test": p_longest_runs
        }

        return metrics
