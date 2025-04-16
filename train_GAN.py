import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

# Global configuration variables
BATCH_SIZE = 64
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "output"
RESUME_PATH = None  # Path to a saved model to resume training from
DEBUG_MODE = True  # Set to True to enable debug prints for tensor shapes


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.debug = DEBUG_MODE
        self.debug_printed = False  # Add flag to track if debug info has been printed
        
        # Initial projection of noise
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 7*7*256),
            nn.LeakyReLU(0.2)
        )
        
        # Encoder (downsampling)
        self.conv1 = nn.Conv2d(256, 128, 3, stride=2, padding=1)  # 7x7 -> 4x4
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 64, 3, stride=2, padding=1)   # 4x4 -> 2x2
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1, padding=1)    # 2x2 -> 2x2
        self.bn3 = nn.BatchNorm2d(32)
        
        # Middle
        self.conv_mid = nn.Conv2d(32, 64, 3, padding=1)  # 2x2 -> 2x2
        self.bn_mid = nn.BatchNorm2d(64)
        
        # Decoder (upsampling)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2x2 -> 4x4
        self.conv_up1 = nn.Conv2d(64, 128, 3, padding=1)  # Keep channels
        self.bn_up1 = nn.BatchNorm2d(128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 4x4 -> 8x8
        self.conv_up2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_up2 = nn.BatchNorm2d(256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 8x8 -> 16x16
        self.conv_up3 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn_up3 = nn.BatchNorm2d(128)
        
        # Final output
        self.up_final = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True)  # Any size -> 28x28
        self.conv_out = nn.Conv2d(128, 1, 3, padding=1)
        
    def forward(self, z):
        should_debug = self.debug and not self.debug_printed
        
        if should_debug:
            print("\n=== Generator Forward Pass ===")
            print(f"Input noise shape: {z.shape}")
        
        # Project and reshape
        x = self.project(z)
        x = x.view(-1, 256, 7, 7)
        if should_debug:
            print(f"After projection shape: {x.shape}")
        
        # Encoder path
        x = F.relu(self.bn1(self.conv1(x)))
        if should_debug:
            print(f"After conv1 shape: {x.shape}")
        
        x = F.relu(self.bn2(self.conv2(x)))
        if should_debug:
            print(f"After conv2 shape: {x.shape}")
        
        x = F.relu(self.bn3(self.conv3(x)))
        if should_debug:
            print(f"After conv3 shape: {x.shape}")
        
        # Middle
        x = F.relu(self.bn_mid(self.conv_mid(x)))
        if should_debug:
            print(f"After conv_mid shape: {x.shape}")
        
        # Decoder Path
        x = self.up1(x)
        x = F.relu(self.bn_up1(self.conv_up1(x)))
        if should_debug:
            print(f"After up1+conv shape: {x.shape}")
        
        x = self.up2(x)
        x = F.relu(self.bn_up2(self.conv_up2(x)))
        if should_debug:
            print(f"After up2+conv shape: {x.shape}")
        
        x = self.up3(x)
        x = F.relu(self.bn_up3(self.conv_up3(x)))
        if should_debug:
            print(f"After up3+conv shape: {x.shape}")
        
        # Final upsampling and output
        x = self.up_final(x)  # Directly upsample to 28x28
        x = torch.tanh(self.conv_out(x))
        if should_debug:
            print(f"Final output shape: {x.shape}")
            print("=== End Generator Forward Pass ===\n")
            self.debug_printed = True  # Set flag after printing debug info
        
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.debug = DEBUG_MODE
        self.debug_printed = False  # Add flag to track if debug info has been printed
        
        # Modify the architecture to handle 28x28 images properly
        self.main = nn.Sequential(
            # Layer 1: 28x28x1 -> 14x14x64
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            
            # Layer 2: 14x14x64 -> 7x7x128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            
            # Layer 3: 7x7x128 -> 4x4x256
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            
            # Layer 4: 4x4x256 -> 2x2x512
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Output layer: 2x2x512 -> 1x1x1
            nn.Conv2d(512, 1, 2, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        should_debug = self.debug and not self.debug_printed
        
        if should_debug:
            print("\n=== Discriminator Forward Pass ===")
            print(f"Input image shape: {x.shape}")
        
        for idx, layer in enumerate(self.main):
            x = layer(x)
            if should_debug and isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                print(f"After conv{idx} shape: {x.shape}")
        
        if should_debug:
            print(f"Final output shape: {x.shape}")
            print("=== End Discriminator Forward Pass ===\n")
            self.debug_printed = True  # Set flag after printing debug info
        
        return x.view(-1, 1)

class GANModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.latent_dim = 100
        self.generator = Generator(self.latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.mnist_data = None
        self.dataloader = None
        
    def load_data_by_digit(self, digit, batch_size=64):
        """Load only MNIST data for a specific digit"""
        if self.mnist_data is None:
            self.mnist_data = torchvision.datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=self.transform
            )
            
        indices = [i for i, (_, label) in enumerate(self.mnist_data) if label == digit]
        filtered_dataset = torch.utils.data.Subset(self.mnist_data, indices)
        self.dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"MNIST dataset loaded with {len(filtered_dataset)} samples of digit {digit}")
        
    def train_step(self):
        """Train the GAN for one epoch"""
        total_d_loss = 0
        total_g_loss = 0
        num_batches = 0

        for batch_idx, (real_images, _) in enumerate(self.dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)

            # Train Discriminator
            self.d_optimizer.zero_grad()
            label_real = torch.ones(batch_size, 1).to(self.device)
            label_fake = torch.zeros(batch_size, 1).to(self.device)

            output_real = self.discriminator(real_images)
            d_loss_real = self.criterion(output_real, label_real)

            noise = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_images = self.generator(noise)
            output_fake = self.discriminator(fake_images.detach())
            d_loss_fake = self.criterion(output_fake, label_fake)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()

            # Train Generator
            self.g_optimizer.zero_grad()
            output_fake = self.discriminator(fake_images)
            g_loss = self.criterion(output_fake, label_real)
            g_loss.backward()
            self.g_optimizer.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: D_loss = {d_loss.item():.4f}, G_loss = {g_loss.item():.4f}")

        return total_d_loss / num_batches, total_g_loss / num_batches
    
    def sample(self, num_samples=1):
        """Generate and save sample images"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            fake_images = self.generator(noise)
            samples = (fake_images.cpu() + 1) / 2  # Normalize to [0, 1]
        return samples

    def generate_samples(self, num_samples=10, digit=None):
        """Generate and save sample images"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            fake_images = self.generator(noise)
            samples = (fake_images.cpu() + 1) / 2  # Normalize to [0, 1]
        
        # Create directory to save samples
        if digit is not None:
            save_dir = os.path.join('gan_output', f'digit_{digit}', 'samples')
        else:
            save_dir = os.path.join('gan_output', 'samples')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save samples as an image grid
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
        for i in range(num_samples):
            if num_samples > 1:
                ax = axes[i]
            else:
                ax = axes
            ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        sample_path = os.path.join(save_dir, f'samples_{digit}.png')
        plt.savefig(sample_path)
        print(f"Generated samples saved to {sample_path}")
        plt.close()
        
        return samples
    
    def save_model(self, digit):
        """Save the GAN models"""
        save_dir = os.path.join('gan_output', f'digit_{digit}', 'models')
        os.makedirs(save_dir, exist_ok=True)
        
        g_path = os.path.join(save_dir, f'generator.pt')
        d_path = os.path.join(save_dir, f'discriminator.pt')
        
        torch.save({
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.g_optimizer.state_dict(),
        }, g_path)
        
        torch.save({
            'model_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.d_optimizer.state_dict(),
        }, d_path)
        
        print(f"Models saved to {save_dir}")

    def load_model(self, path):
        """
        Load saved GAN models from a specified path
        Args:
            path: Directory path containing the saved model files
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            # Construct paths for generator and discriminator models
            g_path = os.path.join(path, 'generator.pt')
            d_path = os.path.join(path, 'discriminator.pt')
            
            if not os.path.exists(g_path) or not os.path.exists(d_path):
                print(f"No saved models found at {path}")
                return False
            
            # Load Generator
            g_checkpoint = torch.load(g_path, map_location=self.device)
            self.generator.load_state_dict(g_checkpoint['model_state_dict'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
            
            # Load Discriminator
            d_checkpoint = torch.load(d_path, map_location=self.device)
            self.discriminator.load_state_dict(d_checkpoint['model_state_dict'])
            self.d_optimizer.load_state_dict(d_checkpoint['optimizer_state_dict'])
            
            print(f"Models successfully loaded from {path}")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

def main():
    # Create output directory
    os.makedirs('gan_output', exist_ok=True)
    
    # Train a separate GAN for each digit
    for digit in [7]:
        print(f"\n{'='*50}")
        print(f"Training GAN for digit {digit}")
        print(f"{'='*50}\n")
        
        # Initialize GAN
        gan = GANModel(device=DEVICE)
        
        # Check if model exists and load it
        model_path = os.path.join('gan_output', f'digit_{digit}', 'models')
        if os.path.exists(model_path):
            print(f"Found existing model for digit {digit}, loading...")
            if gan.load_model(model_path):
                print("Successfully loaded existing model, continuing training...")
            else:
                print("Failed to load model, starting from scratch...")
        else:
            print(f"No existing model found for digit {digit}, starting from scratch...")
        
        # Load digit-specific data
        gan.load_data_by_digit(digit=digit, batch_size=BATCH_SIZE)
        
        # Train the GAN
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            d_loss, g_loss = gan.train_step()
            print(f"Average losses - D: {d_loss:.4f}, G: {g_loss:.4f}")
            
            # Generate and save samples every 5 epochs
            # if (epoch + 1) % 5 == 0:
            #     gan.generate_samples(num_samples=10, digit=digit)
        
        # Save the final model
        gan.save_model(digit)
        
        # Generate final samples
        print(f"Generating final samples for digit {digit}...")
        gan.generate_samples(num_samples=10, digit=digit)
    
    print("\nTraining of all GAN models complete!")

if __name__ == "__main__":
    main() 