import torch
import torch.nn as nn

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