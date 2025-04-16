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
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "output"
RESUME_PATH = None  # Path to a saved model to resume training from
DEBUG_MODE = False  # Set to True to enable debug prints for tensor shapes

# Define a simple U-Net style model for the diffusion process
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, debug=DEBUG_MODE):
        super(UNet, self).__init__()
        self.debug = debug
        self.debug_printed = False  # Track if we've already printed debug info
        
        # Noise embedding - keep size at 256 to match largest feature dimension
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        
        # Encoder (downsampling) - 3 layers
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # Middle
        self.conv_mid = nn.Conv2d(256, 256, 3, padding=1)
        
        # Decoder (upsampling) - 3 layers with alternative upsampling
        self.conv_up1 = nn.Conv2d(256, 128, 3, padding=1)
        self.dconv1 = nn.Conv2d(128 + 256, 128, 3, padding=1)
        
        self.conv_up2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dconv2 = nn.Conv2d(64 + 128, 64, 3, padding=1)
        
        self.conv_up3 = nn.Conv2d(64, 32, 3, padding=1)
        self.dconv3 = nn.Conv2d(32 + in_channels, 32, 3, padding=1)
        
        # Final output layer
        self.conv_out = nn.Conv2d(32, out_channels, 3, padding=1)
        
    def forward(self, x, t):
        # Only print debug info on first forward pass
        should_debug = self.debug and not self.debug_printed
        
        if should_debug:
            print("\n==== SHAPE DEBUGGING ====")
            print(f"Input x shape: {x.shape}")
            print(f"Input t shape: {t.shape}")
        
        # Time embedding
        t_emb = self.time_embedding(t.unsqueeze(-1))
        if should_debug:
            print(f"t after unsqueeze: {t.unsqueeze(-1).shape}")
            print(f"t_emb after time_embedding: {t_emb.shape}")
        
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        if should_debug:
            print(f"t_emb after unsqueeze twice: {t_emb.shape}")
        
        # Save input for final skip connection
        x0 = x
        
        # Encoder Path (3 layers)
        # Layer 1
        x1 = F.relu(self.conv1(x))
        x = self.pool(x1)
        
        # Layer 2
        x2 = F.relu(self.conv2(x))
        x = self.pool(x2)
        
        # Layer 3
        x3 = F.relu(self.conv3(x))
        x = self.pool(x3)
        
        # Middle with time embedding
        # Resize time embedding to match feature map dimensions
        t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])
        if should_debug:
            print(f"t_emb after expand: {t_emb.shape}")
            print(f"x before adding t_emb: {x.shape}")
        
        # Add time embedding to features
        x = x + t_emb  # Now both are 256 channels
        if should_debug:
            print(f"x after adding t_emb: {x.shape}")
        
        x = F.relu(self.conv_mid(x))
        if should_debug:
            print(f"x after conv_mid: {x.shape}")
        
        # Decoder Path (3 layers)
        # Layer 1
        x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv_up1(x)
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.dconv1(x))
        
        # Layer 2
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv_up2(x)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.dconv2(x))
        
        # Layer 3
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv_up3(x)
        x = torch.cat([x, x0], dim=1)
        x = F.relu(self.dconv3(x))
        
        # Final convolution
        x = self.conv_out(x)
        if should_debug:
            print(f"Final output shape: {x.shape}")
            print("==== END DEBUGGING ====\n")
            # Mark that we've printed debug info
            self.debug_printed = True
        
        return x


class DiffusionModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', debug=DEBUG_MODE):
        self.device = device
        self.debug = debug
        self.model = UNet(debug=debug).to(device)
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.timesteps = 1000
        self.loss_fn = nn.MSELoss()
        
        # Define the noise schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # Initialize model parameters
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Load MNIST dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.mnist_data = None
        self.dataloader = None

    def train_step(self):
        """Train the model for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(self.dataloader):
            images = images.to(self.device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
            
            # Add noise to the images
            x_noisy, noise = self.get_noise_for_timestep(t, images)
            
            # Predict the noise
            noise_pred = self.model(x_noisy, t.float() / self.timesteps)
            
            # Compute loss
            loss = self.loss_fn(noise_pred, noise)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, epochs=10):
        """Train the model for multiple epochs"""
        if self.dataloader is None:
            self.load_data()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            avg_loss = self.train_step()
            print(f"Average loss: {avg_loss:.4f}")
            
            # Generate sample images
            # if (epoch + 1) % 5 == 0:
            #     self.generate_samples(f"epoch_{epoch+1}")
    
    def load_data(self, batch_size=64):
        """Load the MNIST dataset"""
        self.mnist_data = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=self.transform
        )
        self.dataloader = DataLoader(self.mnist_data, batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"MNIST dataset loaded with {len(self.mnist_data)} samples")
        
    def load_data_by_digit(self, digit, batch_size=64):
        """Load only MNIST data for a specific digit"""
        # First ensure the dataset is loaded
        if self.mnist_data is None:
            self.mnist_data = torchvision.datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=self.transform
            )
            
        # Filter dataset for the specific digit
        indices = [i for i, (_, label) in enumerate(self.mnist_data) if label == digit]
        filtered_dataset = torch.utils.data.Subset(self.mnist_data, indices)
        
        # Create dataloader for the filtered dataset
        self.dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"MNIST dataset loaded with {len(filtered_dataset)} samples of digit {digit}")
        
    def show_data_samples(self, samples_per_digit=4):
        """Show sample images from the MNIST dataset"""
        if self.mnist_data is None:
            print("Dataset not loaded. Call load_data() first.")
            return
        
        # Create directory to save samples
        os.makedirs(os.path.join(OUTPUT_DIR, "mnist_samples"), exist_ok=True)
        
        # Group images by digit
        digit_samples = {i: [] for i in range(10)}
        
        # Collect samples for each digit
        for img, label in self.mnist_data:
            label_int = label.item() if isinstance(label, torch.Tensor) else label
            if len(digit_samples[label_int]) < samples_per_digit:
                digit_samples[label_int].append(img)
            
            # Check if we have enough samples for all digits
            if all(len(samples) >= samples_per_digit for samples in digit_samples.values()):
                break
        
        # Create a figure to display samples
        fig, axes = plt.subplots(10, samples_per_digit, figsize=(samples_per_digit * 2, 20))
        fig.suptitle("MNIST Dataset Samples", fontsize=16)
        
        # Plot samples for each digit
        for digit, samples in digit_samples.items():
            for i, img in enumerate(samples):
                if samples_per_digit > 1:
                    ax = axes[digit, i]
                else:
                    ax = axes[digit]
                
                # Normalize image for display
                img_display = img.squeeze().numpy()
                img_display = (img_display + 1) / 2  # Normalize from [-1, 1] to [0, 1]
                
                ax.imshow(img_display, cmap='gray')
                ax.set_title(f"Digit: {digit}")
                ax.axis('off')
        
        # Save the figure
        plt.tight_layout()
        sample_path = os.path.join(OUTPUT_DIR, "mnist_samples", "mnist_samples.png")
        plt.savefig(sample_path)
        print(f"MNIST sample images saved to {sample_path}")
        plt.close()
    
    def get_noise_for_timestep(self, t, x_0):
        """Add noise to the input image according to the noise schedule"""
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Add noise to the input according to the noise schedule
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    

    
    def sample(self, num_samples=1, digit=None):
        """Sample images from the model"""
        self.model.eval()
        
        with torch.no_grad():
            # Start with random noise
            x = torch.randn(num_samples, 1, 28, 28).to(self.device)
            
            # Iteratively denoise
            for i in reversed(range(self.timesteps)):
                t = torch.ones(num_samples, device=self.device).long() * i
                
                # Predict the noise
                predicted_noise = self.model(x, t.float() / self.timesteps)
                
                # Get alpha values for current timestep
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]
                
                # No noise at the last step (i=0)
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # Update x using the formula from DDPM paper
                # Key formula
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
        
        # Normalize images to [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        
        return x
    
    def generate_samples(self, name="sample", num_samples=10, digit=None):
        """Generate and save sample images"""
        samples = self.sample(num_samples)
        samples = samples.cpu().numpy()
        
        # Create directory to save samples
        if digit is not None:
            # Save in the digit-specific directory
            save_dir = os.path.join(OUTPUT_DIR, f"digit_{digit}", "generated_samples")
        else:
            # Save in the general output directory
            save_dir = os.path.join(OUTPUT_DIR, "generated_samples")
            
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
        sample_path = os.path.join(save_dir, f"{name}.png")
        plt.savefig(sample_path)
        print(f"Generated samples saved to {sample_path}")
        plt.close()
        
        return samples
    
    def save_model(self, path):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a saved model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # print(f"Model loaded from {path}")


# Function to integrate with the Pygame interface
def generate_digit_image(digit_text, model_dir=None):
    """Generate an image based on text input"""
    # Try to parse the input as a digit
    try:
        digit = int(digit_text)
        if digit < 0 or digit > 9:
            return None  # Only support digits 0-9
    except ValueError:
        return None  # Not a digit
    
    # Initialize the diffusion model
    diffusion = DiffusionModel(device='cpu')  # Use CPU for inference
    
    # Determine the model path
    if model_dir:
        model_path = os.path.join(model_dir, f"digit_{digit}", "diffusion_model.pt")
    else:
        model_path = os.path.join(OUTPUT_DIR, f"digit_{digit}", "diffusion_model.pt")
    
    # Load the digit-specific model if available
    try:
        if os.path.exists(model_path):
            diffusion.load_model(model_path)
        else:
            print(f"Model for digit {digit} not found at {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Generate the image
    sample = diffusion.sample(num_samples=1)
    
    # Convert tensor to numpy array in the right format for Pygame
    image_array = (sample[0, 0].numpy() * 255).astype(np.uint8)
    
    return image_array


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load full dataset once to show samples
    print("Loading MNIST dataset for samples...")
    diffusion_temp = DiffusionModel(device=DEVICE)
    diffusion_temp.load_data(batch_size=BATCH_SIZE)
    diffusion_temp.show_data_samples(samples_per_digit=4)
    
    # Train a separate model for each digit
    for digit in range(10):
        print(f"\n{'='*50}")
        print(f"Training model for digit {digit}")
        print(f"{'='*50}\n")
        
        # Create a directory for this digit's model and samples
        digit_dir = os.path.join(OUTPUT_DIR, f"digit_{digit}")
        os.makedirs(digit_dir, exist_ok=True)
        
        # Initialize a new diffusion model for this digit
        diffusion = DiffusionModel(device=DEVICE)
        
        # Load only data for this specific digit
        print(f"Loading MNIST data for digit {digit}...")
        diffusion.load_data_by_digit(digit=digit, batch_size=BATCH_SIZE)
        
        # Check for a resume path for this specific digit
        digit_resume_path = os.path.join(digit_dir, "diffusion_model.pt") if RESUME_PATH else None
        if digit_resume_path and os.path.exists(digit_resume_path):
            print(f"Loading existing model for digit {digit} from {digit_resume_path}")
            diffusion.load_model(digit_resume_path)
        
        # Train the model
        print(f"Training model for digit {digit} for {EPOCHS} epochs...")
        diffusion.train(epochs=EPOCHS)
        
        # Save the trained model
        model_path = os.path.join(digit_dir,  f"diffusion_model_{digit}.pt")
        print(f"Saving model for digit {digit} to {model_path}")
        diffusion.save_model(model_path)
        
        # Generate and save samples
        print(f"Generating samples for digit {digit}...")
        samples = diffusion.generate_samples(name=f"digit_{digit}_samples", num_samples=10, digit=digit)
    
    print("\nTraining of all digit models complete!")

if __name__ == "__main__":
    main() 