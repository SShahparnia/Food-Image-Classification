import openai
import sys
import os

# Initialize OpenAI client with API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set.")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY  # Use the retrieved API key

# Function to read ingredients from classification_results.txt
def get_ingredients_from_file(file_path):
    ingredients_dict = {}
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return ingredients_dict

    # Read the file and parse it for food classes
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                # Extract the food class after the colon
                _, food_class = line.strip().split(": ", 1)
                # Map the food class to its ingredients using get_ingredients
                ingredients_dict[food_class] = get_ingredients(food_class)
            except ValueError:
                print(f"Skipping malformed line: {line.strip()}")

    return ingredients_dict

# Function to fetch ingredients for a given food class
def get_ingredients(food_class):
    # Predefined dictionary of ingredients for each food class
    ingredients_dict = {
        "Bread": ["flour", "water", "yeast", "salt"],
        "Dairy product": ["milk", "cream", "butter", "cheese"],
        "Dessert": ["sugar", "flour", "butter", "eggs", "vanilla"],
        "Egg": ["eggs", "salt", "pepper", "oil"],
        "Fried food": ["chicken", "flour", "oil", "spices"],
        "Meat": ["beef", "salt", "pepper", "garlic"],
        "Noodles-Pasta": ["pasta", "tomato sauce", "parmesan", "basil"],
        "Rice": ["rice", "water", "salt", "butter"],
        "Seafood": ["shrimp", "garlic", "butter", "lemon"],
        "Soup": ["broth", "chicken", "carrots", "celery"],
        "Vegetable-Fruit": ["lettuce", "spinach", "apple", "orange"],
    }
    return ingredients_dict.get(food_class, ["No ingredients found"])

# Function to generate a recipe using OpenAI's GPT-3.5 Turbo
def generate_recipe(food_class, ingredients):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Create a recipe for {food_class} using the following ingredients: {', '.join(ingredients)}"
                }
            ]
        )
        return response.choices[0].message['content']
    except openai.error.AuthenticationError as e:
        print(f"Authentication error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    classification_file = 'classification_results.txt'

    if not os.path.exists(classification_file):
        print("Classification results not found. Please run classify_image.py first.")
        sys.exit(1)

    # Get the ingredients dictionary from the classification results
    ingredients_dict = get_ingredients_from_file(classification_file)
    
    # Process each food class and generate recipes
    for food_class, ingredients in ingredients_dict.items():
        recipe = generate_recipe(food_class, ingredients)
        print(f"The food in the image is: {food_class}")
        print(f"Ingredients: {', '.join(ingredients)}")
        print(f"Recipe:\n{recipe}\n")
