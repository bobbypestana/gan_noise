import modules.data_loader
import modules.noise_generator

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import wandb
import argparse
import time
import datetime as dt
import matplotlib.pyplot as plt
import os
import numpy as np

# Argument parsing
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

parser.add_argument('--l', type=int, default=1, help='Photonic Circuit Layers')
parser.add_argument('--d', type=int, default=4, help='Photonic Circuit modes')
parser.add_argument('--noise_from_sim', type=bool, default=False, help='Flag to use noise from simulator')

parser.add_argument('--output_channels', type=int, default=1, help='Number of channels in generated image (e.g., 1 for grayscale, 3 for RGB)')
args = parser.parse_args()

# Define the custom directory for wandb logs
custom_wandb_dir = f'{args.dataset}'

# Create the directory if it doesn't exist
os.makedirs(custom_wandb_dir, exist_ok=True)
os.makedirs(f'{custom_wandb_dir}/models', exist_ok=True)
os.makedirs(f'{custom_wandb_dir}/data', exist_ok=True)

# Set the environment variable to use the custom directory
os.environ['WANDB_DIR'] = custom_wandb_dir


if args.log_wandb:
    # Initialize Weights & Biases
    wandb.init(project=args.project_wandb)

    # Configuration for W&B
    config = wandb.config
    config.latent_size = args.latent_size
    config.hidden_size = args.hidden_size
    config.image_size = args.image_size
    config.img_dim = args.img_dim
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.noise_type = args.noise_type
    config.noise_mean = args.noise_mean
    config.noise_std = args.noise_std
    config.noise_min = args.noise_min
    config.noise_max = args.noise_max
    config.noise_lambda = args.noise_lambda
    config.noise_alpha = args.noise_alpha
    config.noise_beta = args.noise_beta
    config.n_samples = args.n_samples
    config.hidden_layers_d = args.hidden_layers_d
    config.hidden_layers_g = args.hidden_layers_g
    config.dataset = args.dataset
    config.run_on_notebook = args.run_on_notebook  # Updated from print_sample
    config.reduction_factor = args.reduction_factor 
    config.penalty_weight = args.penalty_weight 
    config.penalty_weight_entropy = args.penalty_weight_entropy
    config.tolerance = args.tolerance
    config.l = args.l
    config.d = args.d
    config.noise_from_sim = args.noise_from_sim 
    config.output_channels = args.output_channels


# Check for GPU and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Load dataset based on argument
if args.dataset == 'MNIST':
    import torchvision.transforms as transforms

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

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)



from collections import defaultdict
def count_unique_patterns(generated_data, img_dim):
    """
    Count unique patterns in the generated data, rounding pixels to 0 or 1.
    """
    pattern_counter = defaultdict(int)

    for data in generated_data:
        rounded_pattern = np.round(data.cpu().detach().numpy(),1).astype(int).flatten() 
        pattern_tuple = tuple(rounded_pattern)
        pattern_counter[pattern_tuple] += 1

    return len(pattern_counter), pattern_counter

def calculate_penalty(generated_data, img_dim, tolerance=args.tolerance):
    """
    Calculate a penalty if the number of unique patterns is outside the tolerance range.
    """

    global number_of_patterns 
    n_valid_patterns = number_of_patterns

    # Count unique patterns in the generated data
    n_generated_patterns, _ = count_unique_patterns(generated_data, img_dim)

    # Define acceptable range (+/- 30% tolerance)
    lower_bound = (1 - tolerance) * n_valid_patterns
    upper_bound = (1 + tolerance) * n_valid_patterns

    # Apply penalty if the generated patterns are outside the tolerance range
    penalty = 0
    if n_generated_patterns < lower_bound or n_generated_patterns > upper_bound:
        penalty = abs(n_generated_patterns - n_valid_patterns)  # Penalty proportional to deviation

    # print(n_generated_patterns, lower_bound, upper_bound, penalty)

    return penalty





def calculate_entropy_penalty(generated_data, img_dim, valid_patterns):
    """
    Calculate a penalty based on how the entropy of the generated pattern distribution
    compares to the ideal uniform distribution.
    """
    # Count unique patterns in the generated data
    _, pattern_counter = count_unique_patterns(generated_data, img_dim)

    # Convert the counts into a probability distribution
    pattern_frequencies = np.array([pattern_counter.get(tuple(p.flatten()), 0) for p in valid_patterns])
    total_samples = sum(pattern_frequencies)
    if total_samples == 0:
        return 0  # No penalty if no valid patterns were generated

    pattern_probs = pattern_frequencies / total_samples

    # Calculate the entropy of the generated distribution
    entropy_generated = -np.sum(pattern_probs * np.log2(pattern_probs + 1e-12))  # Add small value to avoid log(0)

    # Expected entropy for uniform distribution
    uniform_prob = 1 / len(valid_patterns)
    expected_entropy = -len(valid_patterns) * uniform_prob * np.log2(uniform_prob)

    # Penalty is based on how far the generated entropy is from the expected entropy
    entropy_penalty = abs(entropy_generated - expected_entropy)

    return entropy_penalty


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers, reduction_factor=0.8):
        super(Generator, self).__init__()
        layers = []
        current_size = hidden_size
        self.fc1 = nn.Linear(input_size, current_size)
        layers.append(nn.ReLU())
        
        # Create progressively smaller hidden layers
        for _ in range(hidden_layers):
            next_size = int(current_size * reduction_factor)
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            current_size = next_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_output = nn.Linear(current_size, output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.hidden_layers(x)
        x = self.tanh(self.fc_output(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, reduction_factor=0.8):
        super(Discriminator, self).__init__()
        layers = []
        current_size = hidden_size
        self.fc1 = nn.Linear(input_size, current_size)
        layers.append(nn.ReLU())
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Create progressively smaller hidden layers
        for _ in range(hidden_layers):
            next_size = int(current_size * reduction_factor)
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            current_size = next_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_output = nn.Linear(current_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.hidden_layers(x)
        x = self.sigmoid(self.fc_output(x))
        return x

# Instantiate models and move to device
G = Generator(args.latent_size, args.hidden_size, args.image_size, args.hidden_layers_g, args.reduction_factor).to(device)
D = Discriminator(args.image_size, args.hidden_size, args.hidden_layers_d, args.reduction_factor).to(device)


# def random_fake_data_for_qgan(batch_size, input_size):
#     return torch.bernoulli(torch.rand(batch_size, input_size)).to(device)

# Loss and optimizers
criterion = nn.BCELoss() # Binary Cross-Entropy Loss
# criterion = torch.nn.BCEWithLogitsLoss() #
# criterion = nn.MSELoss()
# criterion = nn.KLDivLoss(reduction="sum")
# criterion = nn.KLDivLoss()


d_optimizer = optim.Adam(D.parameters(), lr=args.learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate)


# Define stopping criteria thresholds
stop_epochs_threshold = 10  # Stop if D(x) ≈ 1.00 and D(G(z)) ≈ 0.00 for this many epochs
no_change_threshold = 0.01  # X% change threshold (e.g., 1% -> 0.01)
no_change_epochs_threshold = 10  # Number of epochs for no change
# performance_decline_epochs_threshold = 5  # Stop if performance declines for this many epochs

# Initialize tracking variables
consecutive_epochs = 0
no_change_epochs = 0
previous_d_x_value = None
previous_d_gz_value = None
# performance_decline_epochs = 0


# Create an instance of the NoiseGenerator class
noise_gen = modules.noise_generator.NoiseGenerator(args, device)


# Training loop
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

        # Generate noise
        z = noise_gen.generate_noise()
        # z = random_fake_data_for_qgan(args.batch_size, args.img_dim)

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
        # z = random_fake_data_for_qgan(args.batch_size, args.img_dim)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)

        

        if args.dataset == 'BAS':
            penalty = calculate_penalty(fake_images, args.img_dim)
            entropy_penalty = calculate_entropy_penalty(fake_images, args.img_dim, all_valid_patterns)
        else:
            penalty = 0
            entropy_penalty = 0
        g_loss = g_loss #+ args.penalty_weight * penalty + args.penalty_weight_entropy * entropy_penalty
        

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Print and log progress
        if (i + 1) % (len(data_loader) // 4) == 0:
            elapsed_time = time.time() - start_time

            d_x_value = real_score.mean().item()
            d_gz_value = fake_score.mean().item()

            print(f'Epoch [{epoch}/{args.num_epochs}], Step [{i + 1}/{len(data_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                  f'D(x): {d_x_value:.2f}, D(G(z)): {d_gz_value:.2f}, '
                  f'Time Elapsed: {elapsed_time:.2f} sec')


            # Log metrics to W&B
            if args.log_wandb:

                wandb.log({
                    'Epoch': epoch,
                    'D Loss': d_loss.item(),
                    'G Loss': g_loss.item(),
                    'D(x)': d_x_value,
                    'D(G(z))': d_gz_value,
                    'Time Elapsed': elapsed_time,
                })

            # Get noise metrics
            # noise_metrics = noise_gen.noise_metrics(z)

            # # Log the metrics to Weights & Biases
            # wandb.log(noise_metrics)


    # Criterion 1: Stop if D(x) ≈ 1.00 and D(G(z)) ≈ 0.00 for consecutive epochs
    if d_x_value >= 0.98 and d_gz_value <= 0.02:
        consecutive_epochs += 1
    else:
        consecutive_epochs = 0

    # Criterion 2: Stop if no significant change in D(x) and D(G(z)) for N epochs
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

    # # Criterion 3: Stop if performance goes down
    # if g_loss.item() > d_loss.item():  # Performance decline: G is failing against D
    #     performance_decline_epochs += 1
    # else:
    #     performance_decline_epochs = 0

    # Check stopping conditions
    if consecutive_epochs >= stop_epochs_threshold:
        print(f"Stopping training as D(x) ≈ 1.00 and D(G(z)) ≈ 0.00 for {stop_epochs_threshold} consecutive epochs.")
        break
    # if no_change_epochs >= no_change_epochs_threshold:
    #     print(f"Stopping training as D(x) and D(G(z)) did not change by more than {no_change_threshold * 100}% "
    #           f"for {no_change_epochs_threshold} consecutive epochs.")
    #     break
    # if performance_decline_epochs >= performance_decline_epochs_threshold:
    #     print(f"Stopping training as performance is declining for {performance_decline_epochs_threshold} consecutive epochs.")
    #     break

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

# Save models only if not running on notebook
if not args.run_on_notebook:
    timestamp = dt.datetime.now().strftime("%y%m%d%H%M%S")
    torch.save(G.state_dict(), f'{args.dataset}/models/generator_{wandb.run.id}_{timestamp}.pth')
    torch.save(D.state_dict(), f'{args.dataset}/models/discriminator_{wandb.run.id}_{timestamp}.pth')

# End W&B run
if args.log_wandb:
    wandb.finish()


