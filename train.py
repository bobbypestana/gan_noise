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
import importlib
import inspect

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
    # parser.add_argument('--noise_from_sim', type=bool, default=False, help='Flag to use noise from simulator')
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
def load_dataset(args, custom_dir):
    if args.dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        dataset = torchvision.datasets.MNIST(
            root=f'{custom_dir}/data',
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




def load_architecture(model_ref):
    """
    Dynamically load the Generator and Discriminator classes from the specified architecture file.
    """
    try:
        # Import the module dynamically
        module = importlib.import_module(f"architectures.{model_ref}")
        # Retrieve the Generator and Discriminator classes
        Generator = getattr(module, "Generator")
        Discriminator = getattr(module, "Discriminator")
        return Generator, Discriminator
    except ModuleNotFoundError:
        raise ValueError(f"Architecture '{model_ref}' not found in 'architectures' directory.")
    except AttributeError:
        raise ValueError(f"Generator or Discriminator class not found in 'architectures/{model_ref}.py'.")
    

def get_constructor_args(cls):
    """
    Extract the constructor arguments and their default values for a given class.
    Returns a dictionary mapping argument names to their default values (or None if no default is provided).
    """
    sig = inspect.signature(cls.__init__)
    args = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue  # Skip the 'self' parameter
        args[name] = param.default if param.default != inspect.Parameter.empty else None
    return args


def compute_derived_args(args, generator_args, discriminator_args):
    """
    Compute derived arguments for Generator and Discriminator based on parsed arguments.
    """
    derived_generator_args = {}
    derived_discriminator_args = {}

    # For Generator
    if 'input_size' in generator_args:
        derived_generator_args['input_size'] = args.latent_size
    if 'output_size' in generator_args:
        derived_generator_args['output_size'] = args.image_size
    if 'hidden_layers' in generator_args:
        derived_generator_args['hidden_layers'] = args.hidden_layers_g

    # For Discriminator
    if 'input_size' in discriminator_args:
        derived_discriminator_args['input_size'] = args.image_size
    if 'hidden_layers' in discriminator_args:
        derived_discriminator_args['hidden_layers'] = args.hidden_layers_d

    return derived_generator_args, derived_discriminator_args

def validate_arguments(cls, args):
    """
    Validate that all required arguments for the given class are present in the provided arguments.
    """
    required_args = [
        name for name, param in inspect.signature(cls.__init__).parameters.items()
        if param.default == inspect.Parameter.empty and name != "self"
    ]
    missing_args = [arg for arg in required_args if arg not in args]
    if missing_args:
        raise ValueError(f"Missing required arguments for {cls.__name__}: {missing_args}")
    
def build_models(args, Generator, Discriminator, device):
    """
    Build the Generator and Discriminator models using the provided arguments.
    Dynamically determines the required arguments for each model and moves them to the specified device.
    
    Args:
        args: Parsed command-line arguments.
        Generator: The Generator class.
        Discriminator: The Discriminator class.
        device: The device (e.g., 'cpu' or 'cuda') to which the models should be moved.
    
    Returns:
        generator: The instantiated Generator model on the specified device.
        discriminator: The instantiated Discriminator model on the specified device.
    """
    # Get the constructor arguments for Generator and Discriminator
    generator_args = get_constructor_args(Generator)
    discriminator_args = get_constructor_args(Discriminator)

    # Compute derived arguments
    derived_generator_args, derived_discriminator_args = compute_derived_args(
        args, generator_args, discriminator_args
    )

    # Filter the parsed arguments to match the required arguments for each model
    filtered_generator_args = {k: v for k, v in vars(args).items() if k in generator_args}
    filtered_discriminator_args = {k: v for k, v in vars(args).items() if k in discriminator_args}

    # Add derived arguments to the filtered arguments
    filtered_generator_args.update(derived_generator_args)
    filtered_discriminator_args.update(derived_discriminator_args)

    # Validate arguments
    validate_arguments(Generator, filtered_generator_args)
    validate_arguments(Discriminator, filtered_discriminator_args)

    # Instantiate the models
    generator = Generator(**filtered_generator_args)
    discriminator = Discriminator(**filtered_discriminator_args)

    # Move models to the specified device
    generator.to(device)
    discriminator.to(device)

    return generator, discriminator

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


# Training loop
def train_gan(args, G, D, data_loader, noise_gen, device, custom_dir, all_valid_patterns=None):
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
                # penalty = calculate_penalty(fake_images, args.img_dim, args.tolerance, number_of_patterns)
                # entropy_penalty = calculate_entropy_penalty(fake_images, args.img_dim, all_valid_patterns)
                penalty = 0
                entropy_penalty = 0
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
    timestamp = dt.datetime.now().strftime("%y%m%d%H%M%S")
    torch.save(G.state_dict(), f'{custom_dir}/models/generator_{wandb.run.id}_{timestamp}.pth')
    torch.save(D.state_dict(), f'{custom_dir}/models/discriminator_{wandb.run.id}_{timestamp}.pth')

    if args.log_wandb:
        wandb.finish()

# Main function
def main():
    args = parse_args()

    if args.run_on_notebook:
        custom_dir = f'../{args.dataset}'
    else:
        custom_dir = f'{args.dataset}'

    
    os.makedirs(custom_dir, exist_ok=True)
    os.makedirs(f'{custom_dir}/models', exist_ok=True)
    os.makedirs(f'{custom_dir}/data', exist_ok=True)
    os.environ['WANDB_DIR'] = custom_dir

    init_wandb(args)

    # Determine the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset, number_of_patterns, all_valid_patterns = load_dataset(args, custom_dir)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

     # Load the architecture dynamically
    Generator, Discriminator = load_architecture(args.model_ref)

    # Build the models and move them to the specified device
    G, D = build_models(args, Generator, Discriminator, device)

    noise_gen = modules.noise_generator.NoiseGenerator(args, device)

    train_gan(args, G, D, data_loader, noise_gen, device, custom_dir, all_valid_patterns)

if __name__ == "__main__":
    main()