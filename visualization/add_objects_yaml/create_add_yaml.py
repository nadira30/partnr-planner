import asyncio
import json
import os
import yaml
import warnings
from agents import Agent, Runner, function_tool

# Suppress async cleanup warnings (Python 3.13 compatibility)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    """Read and return the furniture information for the current episode only."""
    print("currently in get_furniture_info() tool")  # Debug log to confirm function is called
    try:
        with open(furnitures_path, 'r') as f:
            furniture_data = json.load(f)
        
        # Filter to current episode only
        episode_key = str(episode_id)  # ensure string key lookup
        if episode_key not in furniture_data:
            return f"Error: Episode ID '{episode_id}' not found in furniture data. Available IDs: {list(furniture_data.keys())}"
        
        episode_data = furniture_data[episode_key]
        
        formatted = f"Furniture information for episode {episode_id}:\n"
        formatted += json.dumps(episode_data, indent=2)
        print(formatted)  # Log the furniture info for debugging
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
    instructions=(
    "You are a scene-population agent. Your goal is to fill a virtual household scene with realistic objects "
    "based on a YAML specification. The scene should reflect how a real home looks — every room should feel "
    "lived-in and complete.\n\n"

    "## Step-by-step workflow\n"
    "Follow these steps IN ORDER before generating any output:\n"
    "1. Call get_furniture_info() — note every room name and furniture name exactly as they appear.\n"
    "2. Call get_available_objects() — use the 'clean_category' column as the object_category value.\n"
    "3. Call example_yaml_file() — study the expected format carefully.\n\n"

    "## Object placement rules\n"
    "- Populate EVERY room with appropriate objects. Do not leave any room empty.\n"
    "- Place objects on ALL furniture pieces where it makes sense (e.g., shelves, tables, counters, beds).\n"
    "- Reuse the same object category multiple times across different rooms or furniture when realistic.\n"
    "- Prioritize realism: think about what a person would actually keep in each room.\n\n"

    "## Strict constraints\n"
    "- Room names MUST exactly match those returned by get_furniture_info().\n"
    "- Furniture names MUST exactly match those returned by get_furniture_info().\n"
    "- object_category values MUST exactly match the 'clean_category' column from get_available_objects().\n"
    "- Do not invent room names, furniture names, or object categories that were not returned by the tools.\n\n"

    "CRITICAL: You MUST call get_furniture_info() before generating any YAML. "
    "Do NOT generate any output until all three tools have been called. "
    "Generating YAML without first calling the tools is a failure."

    "## Output format\n"
    "Output ONLY valid YAML — no explanations, no comments, no markdown code fences. "
    "Your entire response must be parseable as YAML."),
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
    # print(output_yaml)
    print("\n" + "="*60 + "\n")
    
    # Validate the YAML before saving
    try:
        # Try to parse the YAML to validate it
        parsed_yaml = yaml.safe_load(output_yaml)
        print("✓ YAML validation successful")
        print(f"Parsed structure: {type(parsed_yaml)}")
        
        # Save the validated YAML to a file
        output_file = os.path.join(SCRIPT_DIR, f"generated_add_objects_{episode_id}.yaml")
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
    try:
        asyncio.run(main())
    except (SystemExit, KeyboardInterrupt):
        pass  # Suppress cleanup errors