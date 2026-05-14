import asyncio
import csv
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
OBJECT_CATEGORY_ALIASES = {
    "alarm_clock": "clock",
}


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


def load_object_category_lookup() -> tuple:
    """Load valid clean categories and template-id aliases from the CSV."""
    valid_categories = set()
    template_to_category = {}

    with open(objects_list_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_category = (row.get('clean_category') or '').strip()
            template_id = (row.get('id') or '').strip()
            if clean_category:
                valid_categories.add(clean_category)
            if template_id and clean_category and template_id not in template_to_category:
                template_to_category[template_id.lower()] = clean_category

    return valid_categories, template_to_category


def normalize_generated_yaml(parsed_yaml: dict) -> tuple:
    """Normalize object categories and drop entries that cannot be resolved."""
    valid_categories, template_to_category = load_object_category_lookup()

    objects = parsed_yaml.get('objects', [])
    cleaned_objects = []
    skipped_objects = []

    for obj in objects:
        if not isinstance(obj, dict):
            skipped_objects.append(f"non-dict entry: {obj!r}")
            continue

        raw_category = str(obj.get('object_category', '')).strip()
        normalized_category = raw_category
        raw_category_lower = raw_category.lower()

        if raw_category in valid_categories:
            normalized_category = raw_category
        elif raw_category_lower in template_to_category:
            normalized_category = template_to_category[raw_category_lower]
        else:
            normalized_category = OBJECT_CATEGORY_ALIASES.get(raw_category_lower, raw_category)

        if normalized_category not in valid_categories:
            skipped_objects.append(
                f"{raw_category!r} -> unresolved clean_category"
            )
            continue

        cleaned_obj = dict(obj)
        cleaned_obj['object_category'] = normalized_category
        cleaned_objects.append(cleaned_obj)

    cleaned_yaml = dict(parsed_yaml)
    cleaned_yaml['objects'] = cleaned_objects
    return cleaned_yaml, skipped_objects


# agent = Agent(
#     name="ObjectAdderAgent",
#     instructions=(
#     "You are a scene-population agent. Your goal is to fill a virtual household scene with realistic objects "
#     "based on a YAML specification. The scene should reflect how a real home looks — every room should feel "
#     "lived-in and complete.\n\n"

#     "## Step-by-step workflow\n"
#     "Follow these steps IN ORDER before generating any output:\n"
#     "1. Call get_furniture_info() — note every room name and furniture name exactly as they appear.\n"
#     "2. Call get_available_objects() — use the 'clean_category' column as the object_category value.\n"
#     "3. Call example_yaml_file() — study the expected format carefully.\n\n"

#     "## Object placement rules\n"
#     "- Populate EVERY room with appropriate objects. Do not leave any room empty.\n"
#     "- Place objects on ALL furniture pieces where it makes sense (e.g., shelves, tables, counters, beds).\n"
#     "- Reuse the same object category multiple times across different rooms or furniture when realistic.\n"
#     "- Prioritize realism: think about what a person would actually keep in each room.\n\n"

#     "## Strict constraints\n"
#     "- Room names MUST exactly match those returned by get_furniture_info().\n"
#     "- Furniture names MUST exactly match those returned by get_furniture_info().\n"
#     "- object_category values MUST exactly match the 'clean_category' column from get_available_objects().\n"
#     "- Do not invent room names, furniture names, or object categories that were not returned by the tools.\n\n"

#     "CRITICAL: You MUST call get_furniture_info() before generating any YAML. "
#     "Do NOT generate any output until all three tools have been called. "
#     "Generating YAML without first calling the tools is a failure."

#     "## Output format\n"
#     "Output ONLY valid YAML — no explanations, no comments, no markdown code fences. "
#     "Your entire response must be parseable as YAML."),
#     tools=[example_yaml_file, get_furniture_info, get_available_objects],
# )

agent = Agent(
    name="ObjectAdderAgent",
    instructions=(
        "You are a scene-population agent. Your goal is to fill a virtual household scene with realistic "
        "objects based on a YAML specification. The scene should reflect how a real home looks — every "
        "room should feel lived-in and complete.\n\n"

        "## Workflow — follow IN ORDER, generate NO output until all 3 steps are done\n"
        "1. Call get_furniture_info() — build a mental map of EXACTLY which furniture belongs to which room. Treat this as the only source of truth for room-furniture associations. Your world knowledge about what furniture typically appears in a room is irrelevant and must be ignored.\n"
        "2. Call get_available_objects() — record every value in the 'clean_category' column exactly as returned.\n"
        "3. Call example_yaml_file() — extract the following: top-level structure, key names, "
        "nesting depth, and how rooms, furniture, and object_category are represented.\n\n"

        "## Object placement rules\n"
        "- Populate EVERY room returned by get_furniture_info(). No room may be left empty.\n"
        "- Place objects on ALL furniture pieces where placement makes sense "
        "(e.g., shelves, tables, counters, desks, beds).\n"
        "- Aim for 3–7 objects per furniture piece. Prefer the higher end for storage furniture "
        "(shelves, cabinets, drawers) and the lower end for surfaces (nightstands, coffee tables).\n"
        "- Reuse the same object category across different rooms or furniture when realistic.\n"
        "- Prioritize realism: think about what a person would actually keep in each room.\n\n"

        "## Strict constraints\n"
        "- Room names MUST exactly match those returned by get_furniture_info().\n"
        "- Furniture names MUST exactly match those returned by get_furniture_info().\n"
        "- object_category values MUST exactly match the 'clean_category' column from get_available_objects().\n"
        "- Do not invent room names, furniture names, or object categories not returned by the tools.\n"
        "- If a tool returns an error or empty data, stop and report the error. Do not guess or proceed.\n"
        "- Each furniture piece belongs to exactly one room as returned by get_furniture_info().\n" 
        "- NEVER place objects on a furniture piece under a room it was not listed under.\n"
        "- Before writing any furniture entry, verify: \"Did get_furniture_info() list this furniture under this room?\" If not, do not write it.\n\n"

        "## Output format\n"
        "- Output ONLY valid YAML — no explanations, no comments, no markdown code fences.\n"
        "- Your entire response must be parseable as YAML.\n"
        "- Use the structure, key names, and nesting depth from example_yaml_file() exactly."
    ),
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
        if not isinstance(parsed_yaml, dict):
            raise yaml.YAMLError("Top-level YAML must be a mapping with an 'objects' key")

        parsed_yaml, skipped_objects = normalize_generated_yaml(parsed_yaml)
        if skipped_objects:
            print("⚠ Skipped invalid object categories:")
            for item in skipped_objects:
                print(f"  - {item}")

        if not parsed_yaml.get('objects'):
            raise yaml.YAMLError("No valid objects remained after normalization")

        output_yaml = yaml.safe_dump(parsed_yaml, sort_keys=False, allow_unicode=False)
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