import pygame
import sys
import time
import asyncio
import aiohttp
import threading
from queue import Queue
import requests
import warnings
import os
import numpy as np
import torch
from train_diffusion import DiffusionModel,UNet
from train_GAN import GANModel

# Create queues for communication between threads
IS_USING_GAN = True
task_queue = Queue()
response_queue = Queue()

# Suppress specific PyTorch warning about weights_only
warnings.filterwarnings("ignore", category=FutureWarning, 
                       message="You are using `torch.load` with `weights_only=False`")

# Additionally, you can suppress all FutureWarnings if needed
# warnings.simplefilter(action='ignore', category=FutureWarning)

# Optional: Suppress all warnings from being printed (use with caution)
# os.environ["PYTHONWARNINGS"] = "ignore"

async def async_interact_with_ollama(prompt, model="llama3.2:3b", temperature=0.7, system_prompt=None):

    # print(f"Starting async_interact_with_ollama with model={model}")
    
    # Ollama API endpoint
    url = "http://localhost:11434/api/generate"
    # print(f"Using API endpoint: {url}")
    
    # Prepare the request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }
    
    # Add system prompt if provided
    if system_prompt:
        payload["system"] = system_prompt
        print(f"Added system prompt: {system_prompt[:30]}...")
    
    # print(f"Request payload prepared: {payload.keys()}")
    
    try:
        # print("Creating aiohttp session...")
        # Using aiohttp to make asynchronous requests
        async with aiohttp.ClientSession() as session:
            # print("Sending POST request to Ollama...")
            async with session.post(url, json=payload) as response:
                # print(f"Received response with status: {response.status}")
                if response.status == 200:
                    # Parse the JSON response
                    # print("Parsing JSON response...")
                    result = await response.json()
                    # print(f"Successfully parsed JSON, keys: {result.keys() if result else 'None'}")
                    return result
                else:
                    error_text = await response.text()
                    print(f"Error: Received status code {response.status}")
                    print(f"Error response: {error_text}")
                    return None
                
    except Exception as e:
        print(f"Exception in async_interact_with_ollama: {e}")
        return None

def parse_ollama_response(response):
    """
    Parse the response from Ollama to extract relevant information.
    """
    if not response or "response" not in response:
        return {"error": "Invalid response from Ollama"}
    
    # Extract the text response
    text_response = response["response"]
    
    # Extract other useful metadata
    metadata = {
        "model": response.get("model", "unknown"),
        "total_duration": response.get("total_duration", 0),
        "prompt_eval_duration": response.get("prompt_eval_duration", 0),
        "eval_duration": response.get("eval_duration", 0),
        "tokens_predicted": response.get("eval_count", 0),
    }
    
    return {
        "text": text_response,
        "metadata": metadata
    }

def ollama_worker_thread():
    """
    A worker thread that runs the asyncio event loop for Ollama API calls.
    """
    # print("Worker thread started")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while True:
        # print("Waiting for tasks...")
        # Wait for a task to be added to the queue
        task = task_queue.get()
        
        if task is None:  # None is our signal to exit
            print("Worker thread received exit signal")
            break
            
        # Unpack the task data
        prompt, model, system_prompt = task
        # print(f"Worker thread received task: prompt='{prompt[:30]}...', model='{model}'")
        
        try:
            # Run the async function and get the result
            # print("Calling Ollama API...")
            response = loop.run_until_complete(
                async_interact_with_ollama(prompt, model, system_prompt=system_prompt)
            )
            
            # Parse the response
            if response:
                # print(f"Received response from Ollama: {type(response)}")
                parsed_response = parse_ollama_response(response)
                # print(f"Parsed response: {parsed_response['text'][:50]}...")
                # Put the result in the response queue for the main thread to process
                response_queue.put(parsed_response)
                # print("Response added to queue")
            else:
                print("Received None response from Ollama")
                response_queue.put({"text": "Error: Failed to get response from Ollama", "metadata": {}})
                print("Error response added to queue")
        except Exception as e:
            print(f"Exception in worker thread: {e}")
            response_queue.put({"text": f"Error: Exception while processing - {str(e)}", "metadata": {}})
        
        # Mark the task as done
        task_queue.task_done()
        # print("Task completed")

class TextToImageGame:
    def __init__(self, width=800, height=600, title="Ray Cheng Text to Image"):
        # Initialize pygame
        pygame.init()
        
        # Screen dimensions
        self.WIDTH = width
        self.HEIGHT = height
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption(title)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.DARK_GRAY = (150, 150, 150)
        
        # Fonts
        self.font = pygame.font.SysFont('Arial', 24)
        self.input_font = pygame.font.SysFont('Arial', 28)
        self.title_font = pygame.font.SysFont('Arial', 36)  # Bigger font for title
        
        # Input box
        self.input_box = pygame.Rect(50, 500, 600, 40)
        self.send_button = pygame.Rect(660, 500, 100, 40)
        self.active = False
        self.user_text = ""
        self.cursor_pos = 0  # Current cursor position in the text
        self.cursor_visible = True  # For blinking effect
        self.last_cursor_toggle = 0  # Last time the cursor visibility was toggled
        self.cursor_blink_interval = 0.5  # Seconds
        
        # Key repeat settings
        self.key_repeat_delay = 400  # ms before key starts repeating
        self.key_repeat_interval = 30  # ms between repeats once started
        self.last_key_time = 0  # Last time a key was processed
        self.last_key = None  # Last key pressed
        
        # Message history
        self.messages = []
        self.max_messages = 10  # Maximum number of messages to display
        
        # Game clock
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Ollama API settings
        self.model = "llama3.2:3b"
        self.system_prompt = "User will ask you to response a digit. Answer in one character from 0 to 9. If that question is not related toa digit, please answer n"
        
        # Start the worker thread
        self.worker_thread = threading.Thread(target=ollama_worker_thread, daemon=True)
        self.worker_thread.start()
        
        # Processing state
        self.processing = False
        self.status_message = ""

        #Image
        self.IMAGE_SIZE = 280  # Size of the generated image display (10x MNIST)
        self.image_surface = pygame.Surface((self.IMAGE_SIZE, self.IMAGE_SIZE))
        self.current_image = None
        self.diffusion_model = DiffusionModel(device='cpu')
        self.GAN_model = GANModel(device='cpu')


    def load_diffusion_model(self, digit):
        # print(f"Loading model for digit {digit}...")
        model_path = os.path.join("output", f"digit_{digit}", f"diffusion_model_{digit}.pt")
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None
        try:
            self.diffusion_model.load_model(model_path)
            return self.diffusion_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
        
    def load_GAN_model(self, digit):
        # print(f"Loading model for digit {digit}...")
        model_path = os.path.join("gan_output", f"digit_{digit}", "models")
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None
        try:
            self.GAN_model.load_model(model_path)
            return self.GAN_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
        

    def generate_image_GAN(self,text_digit):
        if len(text_digit) == 1 and text_digit.isdigit():
            text_digit = int(text_digit)
            self.GAN_model = self.load_GAN_model(text_digit)
        else:
            return np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)
            
        
        # Generate the image
        samples = self.GAN_model.sample(num_samples=1)
        
        # Convert tensor to numpy array
        # 1. Get the raw numpy array
        # 2. Scale from [0,1] to [0,255]
        image_array = (samples[0, 0].cpu().numpy() * 255).astype(np.uint8)
        # Calculate scaling factor (assuming MNIST is 28x28)
        original_size = image_array.shape[0]  # Should be 28 for MNIST
        
        # Create a new black image with the target size
        resized_image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)
        
        # Scale the image by mapping coordinates
        scale_factor = self.IMAGE_SIZE / original_size
        
        for y in range(self.IMAGE_SIZE):
            for x in range(self.IMAGE_SIZE):
                # Map the coordinates back to the original image
                orig_y = min(int(y / scale_factor), original_size - 1)
                orig_x = min(int(x / scale_factor), original_size - 1)
                
                # Get the grayscale value and set the pixel
                resized_image[y, x] = image_array[orig_y, orig_x]
        
        return resized_image
    
    def generate_image_diffusion(self,text_digit):
        if len(text_digit) == 1 and text_digit.isdigit():
            text_digit = int(text_digit)
            self.diffusion_model = self.load_diffusion_model(text_digit)
        else:
            return np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)
            
        
        # Generate the image
        samples = self.diffusion_model.sample(num_samples=1)
        
        # Convert tensor to numpy array
        # 1. Get the raw numpy array
        # 2. Scale from [0,1] to [0,255]
        image_array = (samples[0, 0].cpu().numpy() * 255).astype(np.uint8)
        # Calculate scaling factor (assuming MNIST is 28x28)
        original_size = image_array.shape[0]  # Should be 28 for MNIST
        
        # Create a new black image with the target size
        resized_image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)
        
        # Scale the image by mapping coordinates
        scale_factor = self.IMAGE_SIZE / original_size
        
        for y in range(self.IMAGE_SIZE):
            for x in range(self.IMAGE_SIZE):
                # Map the coordinates back to the original image
                orig_y = min(int(y / scale_factor), original_size - 1)
                orig_x = min(int(x / scale_factor), original_size - 1)
                
                # Get the grayscale value and set the pixel
                resized_image[y, x] = image_array[orig_y, orig_x]
        
        return resized_image

    def draw(self):
        self.screen.fill(self.BLACK)
        
        # Draw title
        title_text = self.title_font.render("Ray Cheng Text to Image", True, self.WHITE)
        self.screen.blit(title_text, (self.WIDTH // 2 - title_text.get_width() // 2, 20))
        
        # Calculate position to center the image in the window
        image_x = (self.WIDTH - self.IMAGE_SIZE) // 2
        image_y = (self.HEIGHT - self.IMAGE_SIZE) // 2 - 50  # Shift up a bit to make room for input
        
        # Create and update image surface
        if self.current_image is not None:
            # If we have an image, blit it to our image surface
            for y in range(self.IMAGE_SIZE):
                for x in range(self.IMAGE_SIZE):
                    color_value = self.current_image[y, x]
                    self.image_surface.set_at((x, y), (color_value, color_value, color_value))
        else:
            # If no image, fill with black
            self.image_surface.fill(self.BLACK)
        
        # Draw white border around the image (3 pixels wide)
        border_rect = pygame.Rect(image_x - 3, image_y - 3, self.IMAGE_SIZE + 6, self.IMAGE_SIZE + 6)
        pygame.draw.rect(self.screen, self.WHITE, border_rect)
        
        # Draw the image surface
        self.screen.blit(self.image_surface, (image_x, image_y))
        
        # Draw input box
        pygame.draw.rect(self.screen, self.GRAY if not self.active else self.DARK_GRAY, self.input_box)
        
        # Calculate the visible portion of text
        text_left = self.user_text[:self.cursor_pos]
        text_right = self.user_text[self.cursor_pos:]
        text_left_surface = self.input_font.render(text_left, True, self.WHITE)
        text_right_surface = self.input_font.render(text_right, True, self.WHITE)
        
        # Calculate cursor position in pixels
        cursor_x = self.input_box.x + 5 + text_left_surface.get_width()
        
        # Draw text with cursor
        self.screen.blit(text_left_surface, (self.input_box.x + 5, self.input_box.y + 5))
        self.screen.blit(text_right_surface, (cursor_x, self.input_box.y + 5))
        
        # Draw cursor if active and visible
        if self.active and self.cursor_visible:
            cursor_height = self.input_font.get_height()
            pygame.draw.line(self.screen, self.WHITE, (cursor_x, self.input_box.y + 5), 
                            (cursor_x, self.input_box.y + 5 + cursor_height), 2)
        
        # Draw send button
        pygame.draw.rect(self.screen, self.DARK_GRAY, self.send_button)
        send_text = self.font.render("Send", True, self.WHITE)
        self.screen.blit(send_text, (self.send_button.x + (self.send_button.width - send_text.get_width()) // 2, 
                               self.send_button.y + (self.send_button.height - send_text.get_height()) // 2))
        
        # Draw processing status if active
        if self.processing:
            status_text = self.font.render(self.status_message, True, self.WHITE)
            # Always position near the bottom of the window
            status_y = self.HEIGHT - status_text.get_height() - 20  # 20px from bottom
            self.screen.blit(status_text, (self.WIDTH // 2 - status_text.get_width() // 2, status_y))

        # Draw message history
        message_y = 500
        
        # Draw the last few messages
        for msg in reversed(self.messages[-1:]):
            msg_surface = self.font.render(msg, True, self.WHITE)
            message_y -= 40
            self.screen.blit(msg_surface, (50, message_y))
        
        pygame.display.flip()

    def process_text(self):
        if self.user_text.strip():  # Only process non-empty messages
            prompt = self.user_text
            # print(f"Processing text: '{prompt}'")
            self.messages.append(f"You: {prompt}")
            self.status_message = "Processing..."
            self.processing = True
            
            # Add the task to the queue for the worker thread
            # print(f"Adding task to queue: model={self.model}")
            task_queue.put((prompt, self.model, self.system_prompt))
            # print("Task added to queue")
            
            # Clear input
            self.user_text = ""
            self.cursor_pos = 0

    def handle_key_repeat(self, key, event_type):
        """Handle key repeats for typing and backspace/delete."""
        current_time = pygame.time.get_ticks()
        
        # Initial key press or different key
        if event_type == pygame.KEYDOWN or key != self.last_key:
            self.process_key(key)
            self.last_key = key
            self.last_key_time = current_time
            return
        
        # Key repeat
        if current_time - self.last_key_time > self.key_repeat_delay:
            repeat_time = (current_time - self.last_key_time - self.key_repeat_delay) // self.key_repeat_interval
            if repeat_time > 0 and (current_time - self.last_key_time - self.key_repeat_delay) % self.key_repeat_interval < 20:
                self.process_key(key)
                self.last_key_time = current_time - self.key_repeat_delay - (current_time - self.last_key_time - self.key_repeat_delay) % self.key_repeat_interval

    def process_key(self, key):
        """Process a single key input."""
        if key == pygame.K_RETURN:
            self.process_text()
        elif key == pygame.K_BACKSPACE:
            if self.cursor_pos > 0:
                self.user_text = self.user_text[:self.cursor_pos-1] + self.user_text[self.cursor_pos:]
                self.cursor_pos -= 1
        elif key == pygame.K_DELETE:
            if self.cursor_pos < len(self.user_text):
                self.user_text = self.user_text[:self.cursor_pos] + self.user_text[self.cursor_pos+1:]
        elif key == pygame.K_LEFT:
            self.cursor_pos = max(0, self.cursor_pos - 1)
        elif key == pygame.K_RIGHT:
            self.cursor_pos = min(len(self.user_text), self.cursor_pos + 1)
        elif key == pygame.K_HOME:
            self.cursor_pos = 0
        elif key == pygame.K_END:
            self.cursor_pos = len(self.user_text)
        elif key == pygame.K_SPACE:
            # Handle spacebar directly
            self.user_text = self.user_text[:self.cursor_pos] + " " + self.user_text[self.cursor_pos:]
            self.cursor_pos += 1
        else:
            # Get the unicode representation of the key, which handles shift combinations
            # and special characters correctly
            unicode_char = pygame.key.name(key)
            
            # Handle shift key for uppercase and special characters
            mods = pygame.key.get_mods()
            
            # If a single character key was pressed (letter, number, symbol)
            if len(unicode_char) == 1:
                # Get the actual character to add based on keyboard state
                char_to_add = pygame.key.name(key)
                
                # Check for shift key
                if mods & pygame.KMOD_SHIFT:
                    # Handle special cases for numbers and symbols
                    shift_map = {
                        '1': '!', '2': '@', '3': '#', '4': '$', '5': '%',
                        '6': '^', '7': '&', '8': '*', '9': '(', '0': ')',
                        '-': '_', '=': '+', '[': '{', ']': '}', '\\': '|',
                        ';': ':', "'": '"', ',': '<', '.': '>', '/': '?',
                        '`': '~'
                    }
                    
                    if char_to_add in shift_map:
                        char_to_add = shift_map[char_to_add]
                    else:
                        # For letters, just convert to uppercase
                        char_to_add = char_to_add.upper()
                
                self.user_text = self.user_text[:self.cursor_pos] + char_to_add + self.user_text[self.cursor_pos:]
                self.cursor_pos += 1

    def handle_mouse_click(self, pos):
        """Handle mouse click to position cursor within text."""
        if self.input_box.collidepoint(pos):
            self.active = True
            # Calculate approximate cursor position based on click position
            click_x = pos[0] - self.input_box.x - 5
            
            # Find closest character position
            best_pos = 0
            min_distance = float('inf')
            
            for i in range(len(self.user_text) + 1):
                text_width = self.input_font.size(self.user_text[:i])[0]
                distance = abs(text_width - click_x)
                if distance < min_distance:
                    min_distance = distance
                    best_pos = i
            
            self.cursor_pos = best_pos
        elif self.send_button.collidepoint(pos) and not self.processing:
            self.process_text()
        else:
            self.active = False

    def run(self):
        # Enable key repeat
        pygame.key.set_repeat(self.key_repeat_delay, self.key_repeat_interval)
        
        while self.running:
            current_time = time.time()
            
            # Check for responses from the worker thread
            try:
                if not response_queue.empty():
                    # print("Response queue has items, retrieving...")
                    response_data = response_queue.get_nowait()
                    # print(f"Got response: {response_data['text'][:50]}...")
                    # self.messages.append(f"AI: {response_data['text']}")
                    print(f"AI Response: {response_data['text']}")

                    # Generate image
                    if IS_USING_GAN:
                        self.current_image = self.generate_image_GAN(response_data['text'])
                    else:
                        self.current_image = self.generate_image_diffusion(response_data['text'])

                    self.processing = False
                    self.status_message = ""
                    response_queue.task_done()
                    print("Response processed")
                else:
                    # Periodically print queue status (not too frequently)
                    # if int(current_time) % 5 == 0 and int(current_time * 10) % 10 == 0:  # Every 5 seconds
                    #     print(f"Response queue empty, processing={self.processing}")
                    pass
            except Exception as e:
                print(f"Error processing response: {e}")
            
            # Toggle cursor visibility for blinking effect
            if current_time - self.last_cursor_toggle > self.cursor_blink_interval:
                self.cursor_visible = not self.cursor_visible
                self.last_cursor_toggle = current_time
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Clean up worker thread before exiting
                    self.task_queue.put(None)  # Signal to exit
                    self.worker_thread.join(timeout=1)
                    self.running = False
                
                # Handle mouse clicks
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_click(event.pos)
                
                # Handle keyboard input
                elif event.type == pygame.KEYDOWN and self.active:
                    self.handle_key_repeat(event.key, pygame.KEYDOWN)
            
            # Handle key being held down
            keys = pygame.key.get_pressed()
            if self.active and self.last_key is not None and keys[self.last_key]:
                self.handle_key_repeat(self.last_key, pygame.KEYUP)
            else:
                self.last_key = None
            
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

    

if __name__ == "__main__":
    # Test Ollama connection
    print("Testing Ollama connection...")
    try:
        r = requests.get("http://localhost:11434/api/tags")
        if r.status_code == 200:
            models = r.json()
            # print(f"Ollama connection successful. Available models: {models}")
        else:
            print(f"Ollama responded with status code: {r.status_code}")
            print(f"Response: {r.text}")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running on http://localhost:11434")
        print("You can start it by running 'ollama serve' in a terminal")
        print("Continuing anyway, but expect errors...")
    
    # Start the game
    game = TextToImageGame()
    game.run()