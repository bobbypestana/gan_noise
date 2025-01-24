# Create an instance of the NoiseGenerator class
import sys

# Absolute path to the directory you want to add
path = f"/home/ffb/projetos_individuais/GAN-S/"

# Add the path if it's not already in sys.path
if path not in sys.path:
    sys.path.append(path)

import modules.noise_generator
import argparse


device = 'cpu'

parser = argparse.ArgumentParser(description="Train GAN with different noise types and parameters.", exit_on_error=False)
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
parser.add_argument('--latent_size', type=int, default=16, help='Size of the latent vector')
parser.add_argument('--noise_type', type=str, default='normal', help='Type of noise distribution')
parser.add_argument('--noise_mean', type=float, default=0.0, help='Mean of the noise distribution')
parser.add_argument('--noise_std', type=float, default=1.0, help='Standard deviation of the noise distribution')
args = parser.parse_args()

noise_gen = modules.noise_generator.NoiseGenerator(args, device)
y= noise_gen.generate_noise()


parser = argparse.ArgumentParser(description="Train GAN with different noise types and parameters.", exit_on_error=False)
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
parser.add_argument('--latent_size', type=int, default=16, help='Size of the latent vector')
parser.add_argument('--noise_type', type=str, default='pfc_sim', help='Type of noise distribution')
parser.add_argument('--l', type=int, default=21, help='Photonic Circuit Layers')
parser.add_argument('--d', type=int, default=8, help='Photonic Circuit modes')
args = parser.parse_args()

noise_gen = modules.noise_generator.NoiseGenerator(args, device)
z = noise_gen.generate_noise()

print(y)
print(z)
print(len(z[0]), len(y[0]))
