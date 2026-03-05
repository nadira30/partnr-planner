import asyncio
import json
import os
import yaml
from agents import Agent, Runner, function_tool

# Use absolute paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALIZATION_DIR = os.path.dirname(SCRIPT_DIR)

episode_id = "100" 
furnitures_path = os.path.join(VISUALIZATION_DIR, "data", "furniture_handles_val_mini.json")
objects_list_file = os.path.join(VISUALIZATION_DIR, "objects", "object_categories_one_per_class.csv")


@function_tool
def example_yaml_file() -> str:
    example_yaml = """
    objects:
    # Add a laptop to the office
    - object_category: "laptop"
        room: "office_1"
        furniture: "table_36"

    # Add a book to the bedroom
    - object_category: "book"
        room: "bedroom_1"
        furniture: "table_25"

    # Add a mug to the kitchen counter
    - object_category: "mug"
        room: "kitchen_1"
        furniture: "counter_19"

    # Add a vase to the living room
    - object_category: "vase"
        room: "living_room_1"
        furniture: "table_30"

    # Add items to a cabinet (if articulated furniture)
    - object_category: "bowl"
        room: "kitchen_1"
        furniture: "cabinet_40"
        """
    return example_yaml


@function_tool
def get_furniture_info() -> str:
    """Read and return the furniture information from the furniture handles JSON file."""
    try:
        with open(furnitures_path, 'r') as f:
            furniture_data = json.load(f)
        # Format the furniture data nicely for the agent
        formatted = f"Furniture information for episode {episode_id}:\n"
        formatted += json.dumps(furniture_data, indent=2)
        return formatted
    except FileNotFoundError:
        return f"Error: Furniture file not found at {furnitures_path}"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in furniture file: {str(e)}"


@function_tool
def get_available_objects() -> str:
    """Read and return the list of available objects from the CSV file."""
    try:
        with open(objects_list_file, 'r') as f:
            lines = f.readlines()
        
        # Parse CSV header and data
        if len(lines) < 2:
            return "Error: Objects file is empty or invalid"
        
        header = lines[0].strip()
        formatted = f"Available objects (from {os.path.basename(objects_list_file)}):\n"
        formatted += f"{header}\n"
        formatted += "-" * 50 + "\n"
        
        for line in lines[1:]:
            formatted += line.strip() + "\n"
        
        return formatted
    except FileNotFoundError:
        return f"Error: Objects file not found at {objects_list_file}"
    except Exception as e:
        return f"Error reading objects file: {str(e)}"


agent = Agent(
    name="ObjectAdderAgent",
    instructions="You are an agent that adds objects to a scene based on a YAML specification. " \
    "Use the provided function tools to:\n" \
    "1. Call get_furniture_info() to see available furniture in the episode\n" \
    "2. Call get_available_objects() to see objects you can add (use the 'clean_category' column for object_category)\n" \
    "3. Call example_yaml_file() to see the expected YAML format\n" \
    "Then generate a YAML file that specifies which objects to add. " \
    "Make sure the room, furniture names match what's in the furniture info, and object categories match what's available. " \
    "Output ONLY valid YAML without any explanatory text.",
    tools=[example_yaml_file, get_furniture_info, get_available_objects],
)


async def main():
    result = await Runner.run(
        agent,
        "Generate a YAML file to add objects to the scene. " \
        f"Episode ID: {episode_id}. " \
        "First, get the furniture info and available objects, then generate appropriate YAML. " \
        "Output ONLY the YAML content, no explanations."
    )
    
    output_yaml = result.final_output
    print("Agent Output:")
    print(output_yaml)
    print("\n" + "="*60 + "\n")
    
    # Validate the YAML before saving
    try:
        # Try to parse the YAML to validate it
        parsed_yaml = yaml.safe_load(output_yaml)
        print("✓ YAML validation successful")
        print(f"Parsed structure: {type(parsed_yaml)}")
        
        # Save the validated YAML to a file
        output_file = os.path.join(SCRIPT_DIR, "generated_add_objects.yaml")
        with open(output_file, "w") as f:
            f.write(output_yaml)
        
        print(f"✓ YAML saved to: {output_file}")
        
    except yaml.YAMLError as e:
        print(f"✗ YAML validation failed: {str(e)}")
        print("Saving raw output to 'generated_add_objects_raw.txt' for debugging...")
        
        # Save the invalid output for debugging
        error_file = os.path.join(SCRIPT_DIR, "generated_add_objects_raw.txt")
        with open(error_file, "w") as f:
            f.write(output_yaml)
        
        print(f"Raw output saved to: {error_file}")
        raise


if __name__ == "__main__":
    asyncio.run(main())