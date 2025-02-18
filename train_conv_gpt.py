import modules.data_loader
import modules.noise_generator
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
import argparse
import time
import datetime as dt
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN with different noise types and parameters.")
    parser.add_argument('--noise_type', type=str, default='normal', help='Type of noise distribution')
    parser.add_argument('--noise_mean', type=float, default=0.0, help='Mean of the noise distribution')
    parser.add_argument('--noise_std', type=float, default=1.0, help='Standard deviation of the noise distribution')
    parser.add_argument('--noise_min', type=float, default=-1.0, help='Min value for uniform noise')
    parser.add_argument('--noise_max', type=float, default=1.0, help='Max value for uniform noise')
    parser.add_argument('--noise_lambda', type=float, default=1.0, help='Lambda for exponential or Poisson noise')
    parser.add_argument('--noise_alpha', type=float, default=2.0, help='Alpha for gamma noise')
    parser.add_argument('--noise_beta', type=float, default=1.0, help='Beta for gamma noise')
    parser.add_argument('--hidden_layers_d', type=int, default=5, help='Hidden layers in the discriminator')
    parser.add_argument('--hidden_layers_g', type=int, default=5, help='Hidden layers in the generator')
    parser.add_argument('--latent_size', type=int, default=64, help='Size of the latent vector')
    parser.add_argument('--hidden_size', type=int, default=256, help='Size of the hidden layers')
    parser.add_argument('--image_size', type=int, default=4, help='Size of the image (flattened)')
    parser.add_argument('--img_dim', type=int, default=2, help='Size of the image')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--reduction_factor', type=float, default=0.8, help='Reduction on the number of neurons of the next layer')
    parser.add_argument('--project_wandb', type=str, default='gan-noise-investigation-test', help='Project name for W&B')
    parser.add_argument('--n_samples', type=int, default=10_000, help='Number of samples for training set when applicable')
    parser.add_argument('--run_on_notebook', type=bool, default=False, help='Set to True if running on notebook')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use (MNIST, BAS, etc.)')
    parser.add_argument('--log_wandb', type=bool, default=False, help='Decide whether to log results')
    parser.add_argument('--penalty_weight', type=float, default=0.5, help='Penalty weight for RL')
    parser.add_argument('--penalty_weight_entropy', type=float, default=0.5, help='Penalty weight for RL')
    parser.add_argument('--tolerance', type=float, default=0.3, help='Tolerance for Penalty for RL')
    parser.add_argument('--photonic_layers', type=int, default=1, help='Photonic Circuit Layers')
    parser.add_argument('--photonic_modes', type=int, default=4, help='Photonic Circuit modes')
    parser.add_argument('--noise_from_sim', type=bool, default=False, help='Flag to use noise from simulator')
    parser.add_argument('--output_channels', type=int, default=1, help='Number of channels in generated image (e.g., 1 for grayscale, 3 for RGB)')
    parser.add_argument('--model_ref', type=str, default='None' )

    return parser.parse_args()

# Initialize W&B
def init_wandb(args):
    if args.log_wandb:
        wandb.init(project=args.project_wandb)
        config = wandb.config
        config.update(vars(args))

# Load dataset
def load_dataset(args, custom_wandb_dir):
    if args.dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        dataset = torchvision.datasets.MNIST(
            root=f'{custom_wandb_dir}/data',
            train=True,
            transform=transform,
            download=True
        )
        number_of_patterns = 10
    elif args.dataset == 'BAS':
        dataset = modules.data_loader.BASDataset(n_samples=args.n_samples, img_dim=args.img_dim)
        number_of_patterns = dataset.get_number_of_patterns()
        all_valid_patterns = dataset.generate_all_unique_bas_patterns(args.img_dim)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    return dataset, number_of_patterns, all_valid_patterns if args.dataset == 'BAS' else None

# Count unique patterns
def count_unique_patterns(generated_data, img_dim):
    pattern_counter = defaultdict(int)
    for data in generated_data:
        rounded_pattern = np.round(data.cpu().detach().numpy(), 1).astype(int).flatten()
        pattern_tuple = tuple(rounded_pattern)
        pattern_counter[pattern_tuple] += 1
    return len(pattern_counter), pattern_counter

# Calculate penalty
def calculate_penalty(generated_data, img_dim, tolerance, n_valid_patterns):
    n_generated_patterns, _ = count_unique_patterns(generated_data, img_dim)
    lower_bound = (1 - tolerance) * n_valid_patterns
    upper_bound = (1 + tolerance) * n_valid_patterns
    penalty = abs(n_generated_patterns - n_valid_patterns) if n_generated_patterns < lower_bound or n_generated_patterns > upper_bound else 0
    return penalty

# Calculate entropy penalty
def calculate_entropy_penalty(generated_data, img_dim, valid_patterns):
    _, pattern_counter = count_unique_patterns(generated_data, img_dim)
    pattern_frequencies = np.array([pattern_counter.get(tuple(p.flatten()), 0) for p in valid_patterns])
    total_samples = sum(pattern_frequencies)
    if total_samples == 0:
        return 0
    pattern_probs = pattern_frequencies / total_samples
    entropy_generated = -np.sum(pattern_probs * np.log2(pattern_probs + 1e-12))
    uniform_prob = 1 / len(valid_patterns)
    expected_entropy = -len(valid_patterns) * uniform_prob * np.log2(uniform_prob)
    return abs(entropy_generated - expected_entropy)

# Define Generator and Discriminator
# Replaced Generator class

class Generator(torch.nn.Module):
    def __init__(self, latent_dims: int = 100, hidden_dims: int = 128, image_dims: tuple[int, int] = (28, 28)):
        super(Generator, self).__init__()
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.image_dims = image_dims
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.latent_dims, out_features=np.prod(self.image_dims)),
            torch.nn.BatchNorm1d(num_features=np.prod(self.image_dims)),
            torch.nn.LeakyReLU(0.01),
        )

        self.reshape = torch.nn.Sequential(torch.nn.Unflatten(1, (16, 7, 7)))

        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=32, padding=(2, 2),
                                     kernel_size=(5, 5), stride=(2, 2),
                                     output_padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(negative_slope=0.01)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, padding=(2, 2),
                                     kernel_size=(5, 5), stride=(2, 2),
                                     output_padding=(1, 1), bias=False),
            torch.nn.Sigmoid()
        )

        self.to(self.device)

    def forward(self, latent_vectors: torch.Tensor):
        latent_vectors = latent_vectors.to(self.device)
        assert len(latent_vectors.shape) == 2, "Batch of latent vectors should have shape: (batch size, latent_dims)"
        assert latent_vectors.shape[1] == self.latent_dims, f'Each latent vector in batch should be of size: {self.latent_dims}'

        generated_images = self.fc(latent_vectors)
        generated_images = self.reshape(generated_images)
        generated_images = self.conv1(generated_images)
        generated_images = self.conv2(generated_images)
        return generated_images

    def __init__(self, input_size, hidden_size, output_size, hidden_layers, reduction_factor=0.8):
        super(Generator, self).__init__()
        layers = []
        current_size = hidden_size
        self.fc1 = nn.Linear(input_size, current_size)
        layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            next_size = int(current_size * reduction_factor)
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            current_size = next_size
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_output = nn.Linear(current_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.hidden_layers(x)
        x = self.tanh(self.fc_output(x))
        return x

# Replaced Discriminator class

class Discriminator(torch.nn.Module):
    def __init__(self, image_dims: tuple[int, int] = (28, 28), hidden_dims: int = 128):
        super(Discriminator, self).__init__()
        self.image_dims = image_dims
        self.hidden_dims = hidden_dims
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5),
                            stride=(2, 2), padding=(2, 2), bias=False),
            torch.nn.LeakyReLU(negative_slope=0.01)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 5),
                            stride=(2, 2), padding=(2, 2), bias=False),
            torch.nn.LeakyReLU(negative_slope=0.01)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=np.prod(self.image_dims), out_features=np.prod(self.image_dims)),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=np.prod(self.image_dims), out_features=1)
        )

        self.to(self.device)

    def forward(self, images: torch.Tensor):
        images = images.to(self.device)
        assert len(images.shape) == 4, f'Images should be given as shape: (batch_size, nr_channels, height, width), but is {images.shape}'
        assert images.shape[2:] == self.image_dims, f'Dimension of each image in batch should be: {self.image_dims}, but is {images.shape[2:]}'

        prediction = self.conv1(images)
        prediction = self.conv2(prediction)
        prediction = self.fc(prediction)
        return prediction

    def __init__(self, input_size, hidden_size, hidden_layers, reduction_factor=0.8):
        super(Discriminator, self).__init__()
        layers = []
        current_size = hidden_size
        self.fc1 = nn.Linear(input_size, current_size)
        layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            next_size = int(current_size * reduction_factor)
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            current_size = next_size
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_output = nn.Linear(current_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.hidden_layers(x)
        x = self.sigmoid(self.fc_output(x))
        return x

# Training loop
def train_gan(args, G, D, data_loader, noise_gen, device, custom_wandb_dir, all_valid_patterns=None):
    criterion = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=args.learning_rate)
    g_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate)

    stop_epochs_threshold = 10
    no_change_threshold = 0.01
    no_change_epochs_threshold = 10
    consecutive_epochs = 0
    no_change_epochs = 0
    previous_d_x_value = None
    previous_d_gz_value = None

    for epoch in range(args.num_epochs):
        start_time = time.time()
        for i, (images, _) in enumerate(data_loader):
            images = images.reshape(args.batch_size, -1).to(device)
            real_labels = torch.ones(args.batch_size, 1).to(device)
            fake_labels = torch.zeros(args.batch_size, 1).to(device)

            # Train Discriminator
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            z = noise_gen.generate_noise()
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            z = noise_gen.generate_noise()
            fake_images = G(z)
            outputs = D(fake_images)
            g_loss = criterion(outputs, real_labels)

            if args.dataset == 'BAS':
                penalty = calculate_penalty(fake_images, args.img_dim, args.tolerance, number_of_patterns)
                entropy_penalty = calculate_entropy_penalty(fake_images, args.img_dim, all_valid_patterns)
            else:
                penalty = 0
                entropy_penalty = 0
            g_loss = g_loss + args.penalty_weight * penalty + args.penalty_weight_entropy * entropy_penalty

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Log progress
            if (i + 1) % (len(data_loader) // 4) == 0:
                elapsed_time = time.time() - start_time
                d_x_value = real_score.mean().item()
                d_gz_value = fake_score.mean().item()
                print(f'Epoch [{epoch}/{args.num_epochs}], Step [{i + 1}/{len(data_loader)}], '
                      f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                      f'D(x): {d_x_value:.2f}, D(G(z)): {d_gz_value:.2f}, '
                      f'Time Elapsed: {elapsed_time:.2f} sec')

                if args.log_wandb:
                    wandb.log({
                        'Epoch': epoch,
                        'D Loss': d_loss.item(),
                        'G Loss': g_loss.item(),
                        'D(x)': d_x_value,
                        'D(G(z))': d_gz_value,
                        'Time Elapsed': elapsed_time,
                    })

        # Check stopping criteria
        if d_x_value >= 0.98 and d_gz_value <= 0.02:
            consecutive_epochs += 1
        else:
            consecutive_epochs = 0

        if previous_d_x_value is not None and previous_d_gz_value is not None:
            d_x_change = abs(d_x_value - previous_d_x_value) / max(abs(previous_d_x_value), 1e-8)
            d_gz_change = abs(d_gz_value - previous_d_gz_value) / max(abs(previous_d_gz_value), 1e-8)
            if d_x_change <= no_change_threshold and d_gz_change <= no_change_threshold:
                no_change_epochs += 1
            else:
                no_change_epochs = 0
            previous_d_x_value = d_x_value
            previous_d_gz_value = d_gz_value
        else:
            previous_d_x_value = d_x_value
            previous_d_gz_value = d_gz_value

        if consecutive_epochs >= stop_epochs_threshold:
            print(f"Stopping training as D(x) ≈ 1.00 and D(G(z)) ≈ 0.00 for {stop_epochs_threshold} consecutive epochs.")
            break

        # Save and visualize generated images
        if args.run_on_notebook and ((epoch + 1) % 20 == 0):
            with torch.no_grad():
                fake_images = fake_images.reshape(fake_images.size(0), 1, args.img_dim, args.img_dim)
                grid = torchvision.utils.make_grid(fake_images, nrow=10, normalize=True)
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.title('Fake Images')
                plt.show()
                _, __ = count_unique_patterns(fake_images, args.img_dim)
                print(f'Fake patterns: {_}')

            with torch.no_grad():
                # Display real images from the dataset
                real_images_display = images.reshape(images.size(0), 1, args.img_dim, args.img_dim)
                real_grid = torchvision.utils.make_grid(real_images_display, nrow=10, normalize=True)
                plt.subplot(1, 2, 2)
                plt.imshow(real_grid.permute(1, 2, 0).cpu().numpy())
                plt.title('Real Images')
                plt.show()
                _, __ = count_unique_patterns(real_images_display, args.img_dim)
                print(f'Real patterns: {_}')

    # Save models
    prefix = ''
    if args.run_on_notebook:
        prefix = '../'

    timestamp = dt.datetime.now().strftime("%y%m%d%H%M%S")
    torch.save(G.state_dict(), f'{prefix}{args.dataset}/models/generator_{wandb.run.id}_{timestamp}.pth')
    torch.save(D.state_dict(), f'{prefix}{args.dataset}/models/discriminator_{wandb.run.id}_{timestamp}.pth')

    if args.log_wandb:
        wandb.finish()

# Main function
def main():
    args = parse_args()
    custom_wandb_dir = f'{args.dataset}'
    os.makedirs(custom_wandb_dir, exist_ok=True)
    os.makedirs(f'{custom_wandb_dir}/models', exist_ok=True)
    os.makedirs(f'{custom_wandb_dir}/data', exist_ok=True)
    os.environ['WANDB_DIR'] = custom_wandb_dir

    init_wandb(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, number_of_patterns, all_valid_patterns = load_dataset(args, custom_wandb_dir)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    G = Generator(args.latent_size, args.hidden_size, args.image_size, args.hidden_layers_g, args.reduction_factor).to(device)
    D = Discriminator(args.image_size, args.hidden_size, args.hidden_layers_d, args.reduction_factor).to(device)
    noise_gen = modules.noise_generator.NoiseGenerator(args, device)

    train_gan(args, G, D, data_loader, noise_gen, device, custom_wandb_dir, all_valid_patterns)

if __name__ == "__main__":
    main()