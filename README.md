# PyTorch Text-to-Image Diffusion with MNIST

This repository contains implementations of different image generation models using the MNIST dataset:

- Text-to-Image generation with diffusion models
- GAN implementation for image generation
- Ollama integration for text prompts

## Setup

1. Install the required dependencies
2. Ensure you have Ollama running locally at port 11434 for the test_ollama_request.py script
3. Run the desired script based on your needs

## Scripts

- `test_ollama_request.py` - Interact with a local Ollama model for text-to-image prompts
- `train_diffusion.py` - Train a diffusion model on MNIST data
- `train_GAN.py` - Train a GAN model on MNIST data
- `MainGameT2I.py` - Main application interface 