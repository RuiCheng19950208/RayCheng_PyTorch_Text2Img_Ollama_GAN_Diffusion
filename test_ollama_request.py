import requests
import json
import asyncio
import aiohttp


MODEL = "llama3.2:3b"

def get_user_prompt():
    """
    Get a prompt from the user interactively.
    
    Returns:
        str: The user's prompt
    """
    print("\n=== Ollama Interaction ===")
    print("Type your prompt below (press Enter twice to submit):")
    
    lines = []
    while True:
        line = input()
        if line.strip() == "" and lines:  # Empty line and we have content
            break
        lines.append(line)
    
    # Join the lines to form the complete prompt
    prompt = "\n".join(lines)
    return prompt

def interact_with_ollama(prompt, model="llama3.2:3b", temperature=0.7, system_prompt=None):
    """
    Send a prompt to Ollama and get the response.
    
    Args:
        prompt (str): The user prompt to send
        model (str): The model to use (default: llama3)
        temperature (float): Controls randomness (default: 0.7)
        system_prompt (str, optional): System prompt to set context
        
    Returns:
        dict: The parsed response from Ollama
    """
    # Ollama API endpoint
    url = "http://localhost:11434/api/generate"
    
    # Prepare the request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False  # Set to True if you want streaming responses
    }
    
    # Add system prompt if provided
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        # Send the request to Ollama
        response = requests.post(url, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
            return result
        else:
            print(f"Error: Received status code {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return None

def parse_ollama_response(response):
    """
    Parse the response from Ollama to extract relevant information.
    
    Args:
        response (dict): The response from Ollama
        
    Returns:
        dict: Parsed information from the response
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

async def async_interact_with_ollama(prompt, model="llama3.2:3b", temperature=0.7, system_prompt=None):
    """
    Asynchronously send a prompt to Ollama and get the response.
    
    Args:
        prompt (str): The user prompt to send
        model (str): The model to use (default: llama3.2:3b)
        temperature (float): Controls randomness (default: 0.7)
        system_prompt (str, optional): System prompt to set context
        
    Returns:
        dict: The parsed response from Ollama
    """
    # Ollama API endpoint
    url = "http://localhost:11434/api/generate"
    
    # Prepare the request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False  # Set to True if you want streaming responses
    }
    
    # Add system prompt if provided
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        # Using aiohttp to make asynchronous requests
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    # Parse the JSON response
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    print(f"Error: Received status code {response.status}")
                    print(error_text)
                    return None
                
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return None

# Main function to run the interactive Ollama interface
def main():

    
    # Optional system prompt
    system_prompt = "The user will ask you a question about generating images of a digit. Answer in one character from '0' to '9'. If that question is not related to generating images of a digit, please answer with 'n'"
    
    while True:
        # Get user prompt interactively
        user_prompt = get_user_prompt()
        
        if user_prompt.lower() in ["exit", "quit"]:
            print("Exiting Ollama interaction.")
            break
            
        print(f"\nSending prompt to Ollama model '{MODEL}'...")
        
        # Get response from Ollama
        response = interact_with_ollama(
            prompt=user_prompt,
            model=MODEL,
            system_prompt=system_prompt
        )
        
        # Parse and display the response
        if response:
            parsed_data = parse_ollama_response(response)
            
            print("\n--- Response from Ollama ---")
            print(parsed_data["text"])
            
            print("\n--- Metadata ---")
            for key, value in parsed_data["metadata"].items():
                print(f"{key}: {value}")
        else:
            print("Failed to get a response from Ollama.")
        
        # Ask if user wants to continue
        continue_choice = input("\nWould you like to ask another question? (y/n): ").lower()
        if continue_choice not in ["y", "yes"]:
            print("Exiting Ollama interaction.")
            break

if __name__ == "__main__":
    main()
