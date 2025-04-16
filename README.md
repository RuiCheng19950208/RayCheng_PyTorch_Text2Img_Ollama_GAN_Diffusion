# PyTorch Text-to-Image Diffusion with MNIST
-Demos on Youtube: 
-https://youtu.be/ILq5gP-RLis
-https://youtu.be/A8Pkf4d6Q9o
![Image_20250416133026](https://github.com/user-attachments/assets/676f51f6-c2a5-41d7-9d5f-974a83857eb1)
![samples_8](https://github.com/user-attachments/assets/aab556f7-3763-4b37-98fb-1983c8900db3)


This repository contains implementations of different image generation models using the MNIST dataset:

- Text-to-Image generation with diffusion models
- GAN implementation for image generation
- Ollama integration for text prompts

## Setup

1. Install the required dependencies
2. You have to run  train_diffusion.py or train_GAN.py to get the model.
3. Ensure you have Ollama running locally at port 11434 for the test_ollama_request.py script
4. Run the desired script based on your needs


## Scripts
- `MainGameT2I.py` - Main application interface (set IS_USING_GAN = True/False to toggle between GAN and Diffusion model)
- `test_ollama_request.py` - Interact with a local Ollama model for basic chatting function. Based on the default prompt, Ollama shall response with a single digit.
- `train_diffusion.py` - Train a diffusion model on MNIST data
- `train_GAN.py` - Train a GAN model on MNIST data
