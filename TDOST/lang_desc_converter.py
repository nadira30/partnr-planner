import re
import os
import time
from datetime import datetime
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config ────────────────────────────────────────────────────────────────────
PROMPT_FILE   = "prompt_lang_descriptions.txt"
INPUT_FILE    = "files/human_room_trace_ava.txt"
OUTPUT_FILE   = "tdost_file/actual/tdost_llm_human_room_trace_ava.txt"
# ─────────────────────────────────────────────────────────────────────────────


def load_prompt(prompt_path: str, data_file_path: str) -> str:
    """Load the system prompt and inject the data file path."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    return prompt.replace('{FILE_PATH}', data_file_path)


def parse_line(line: str) -> dict | None:
    """  
    Parse one line of the raw trace file.

    Expected format:
        <Day> <HH:MM:SS> <HH:MM:SS> <float> <float> <float> <room_id> <label>

    Returns a dict with keys: day, time, room, activity
    Returns None if the line is malformed or empty.
    """
    line = line.strip()
    if not line:
        return None

    # 1) Common new format: space-separated columns
    #    day wall_time sim_time x z y_world room activity
    parts = line.split()
    if len(parts) >= 8:
        day = parts[0]
        time = parts[1]  # wall_time is the second token
        # Validate the time token; skip header or malformed rows
        if not re.match(r"^\d{2}:\d{2}:\d{2}$", time):
            return None
        room = parts[6]
        activity = " ".join(parts[7:])
        return {'day': day, 'time': time, 'room': room, 'activity': activity}

    # 2) Labeled format: look for `wall_time=` and `room=` tokens
    wall_pattern = (
        r'^(?P<day>\S+).*?wall_time=(?P<time>\d{2}:\d{2}:\d{2}).*?'
        r'room=(?P<room>\S+)\s+activity=(?P<activity>\S+)$'
    )
    match = re.match(wall_pattern, line)
    if match:
        return match.groupdict()

    # 3) Fallback to original unlabeled-with-keys format where time is the second token
    pattern = (
        r'^(?P<day>\S+)\s+'
        r'(?P<time>\d{2}:\d{2}:\d{2})\s+'
        r'x=\S+\s+'
        r'y=\S+\s+'
        r'y_world=\S+\s+'
        r'room=(?P<room>\S+)\s+'
        r'activity=(?P<activity>\S+)$'
    )
    match = re.match(pattern, line)
    if not match:
        return None

    return match.groupdict()


def get_time_period(time_str: str) -> str:
    """Map HH:MM:SS to a time-of-day label."""
    hour = datetime.strptime(time_str, "%H:%M:%S").hour
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Late Night"


def build_events(input_file: str) -> list[dict]:
    """
    Read the trace file line by line and detect sensor events.

    Emits:
      - ON  when a room is first entered (or file starts)
      - OFF then ON when a room transition occurs
    """
    events = []
    prev_room = None
    event_index = 1
    current_stay_start = None   # time the person entered the current room

    with open(input_file, 'r', encoding='utf-8') as f:
        for raw_line in f:
            # print(f"Parsing line: {raw_line.strip()}")
            parsed = parse_line(raw_line)
            if parsed is None:
                continue

            day      = parsed['day']
            time_str = parsed['time']
            room     = parsed['room']
            activity = parsed['activity']
            period   = get_time_period(time_str)

            if prev_room is None:
                # First valid line → emit ON
                events.append({
                    'sensor_key': f"M_{room}_{event_index}",
                    'state':      'ON',
                    'time':       time_str,
                    'day':        day,
                    'period':     period,
                    'room':       room,
                    'activity':   activity,
                })
                event_index += 1
                current_stay_start = time_str

            elif room != prev_room:
                # Room transition → emit OFF for previous room, ON for new room
                events.append({
                    'sensor_key': f"M_{prev_room}_{event_index}",
                    'state':      'OFF',
                    'time':       time_str,
                    'day':        day,
                    'period':     get_time_period(current_stay_start),
                    'room':       prev_room,
                    'activity':   activity,
                })
                event_index += 1

                events.append({
                    'sensor_key': f"M_{room}_{event_index}",
                    'state':      'ON',
                    'time':       time_str,
                    'day':        day,
                    'period':     period,
                    'room':       room,
                    'activity':   activity,
                })
                event_index += 1
                current_stay_start = time_str

            # Same room → no new event (contiguous stay already captured by ON)
            prev_room = room

    return events


def room_to_human(room_id: str) -> str:
    """Convert raw room IDs to human-readable names."""
    mapping = {
        'living_room_1': 'living room',
        'dining_room_1': 'dining room',
        'kitchen_1':     'kitchen',
        'hallway_1':     'hallway',
        'bedroom_1':     'bedroom',
        'bathroom_1':    'bathroom',
        'office_1':      'office',
        'workout_1':     'gym',
        "workout":       'gym',
        "gym":           'gym',
    }
    return mapping.get(room_id, room_id.replace('_', ' '))


def build_user_message(event: dict) -> str:
    """Build the prompt for one sensor event."""
    human_room = room_to_human(event['room'])
    sensor_key = event['sensor_key']
    state = event['state']
    time_str = event['time']
    day = event['day']
    period = event['period']

    return (
        f"Generate a TDOST sensor description for the following event:\n\n"
        f"Sensor Key : {sensor_key} {state} {time_str}\n"
        f"Day        : {day}\n"
        f"Time-of-day: {period}\n"
        f"Room       : {human_room}\n"
        f"State      : {state}\n\n"
        f"Output format:\n"
        f"{sensor_key} {state} {time_str}\n"
        f"({day}, {period}, Motion, in {human_room}, {state})\n"
        f'["<Sentence1>", "<Sentence2>", "<Sentence3>"]'
    )


def request_tdost_description(client: OpenAI, system_prompt: str, event: dict, max_attempts: int = 3) -> str:
    """Request one TDOST description with simple retry support."""
    user_message = build_user_message(event)

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(2 ** (attempt - 1))


def count_nonempty_lines(file_path: str) -> int:
    """Count existing completed rows in the output file."""
    if not os.path.exists(file_path):
        return 0

    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def generate_tdost_descriptions(events: list[dict], system_prompt: str) -> list[str]:
    """
    Send each sensor event to GPT and collect TDOST-style output.
    """
    client = OpenAI()   # reads OPENAI_API_KEY from env
    results = []

    for event in events:
        results.append(request_tdost_description(client, system_prompt, event))

    return results


def format_description_row(description: str, activity: str = '') -> str:
    """
    Flatten a multiline TDOST response into a single tab-separated row with activity.
    Always outputs exactly 4 columns: raw_event, structured_event, descriptions, label
    """
    lines = description.splitlines()
    
    # Initialize all 4 columns with empty strings
    raw_event = ''
    structured_event = ''
    descriptions = ''
    
    # First column: sensor_key state time (line 1)
    if lines and lines[0].strip():
        raw_event = lines[0].strip()
    
    # Second column: structured event (line 2)
    if len(lines) > 1 and lines[1].strip():
        structured_event = lines[1].strip()
    
    # Third column: JSON array descriptions (everything from line 3 onwards, kept as single column)
    if len(lines) > 2:
        # Join remaining lines into a single column, removing newlines and extra tabs
        descriptions = " ".join(" ".join(lines[2:]).split())
    
    # Ensure we never output empty values - use placeholder if needed
    if not raw_event and not structured_event and not descriptions:
        # Fallback for completely empty/malformed responses
        descriptions = description.strip()
    
    # Always output exactly 4 columns with tabs
    return "\t".join([raw_event, structured_event, descriptions, activity])


def process(prompt_file: str, input_file: str, output_file: str) -> None:
    """End-to-end pipeline: load prompt → detect events → generate → save."""
    prompt_path = os.path.join(BASE_DIR, prompt_file)
    input_path = os.path.join(BASE_DIR, input_file)
    output_path = os.path.join(BASE_DIR, output_file)

    print(f"Loading prompt from : {prompt_path}")
    system_prompt = load_prompt(prompt_path, input_path)

    print(f"Parsing trace file  : {input_path}")
    events = build_events(input_path)
    print(f"  {len(events)} events detected.")

    completed_rows = count_nonempty_lines(output_path)
    if completed_rows > len(events):
        completed_rows = 0

    if completed_rows:
        print(f"Resuming from row   : {completed_rows}")
    else:
        print("Generating TDOST descriptions ...")

    if completed_rows == len(events):
        print(f"Done. Output already complete: {output_path}")
        return

    client = OpenAI()
    output_mode = 'a' if completed_rows else 'w'

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, output_mode, encoding='utf-8') as f_out:
        for event in events[completed_rows:]:
            description = request_tdost_description(client, system_prompt, event)
            f_out.write(format_description_row(description, event['activity']) + '\n')
            f_out.flush()

    print(f"Done. Output saved to: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    process(PROMPT_FILE, INPUT_FILE, OUTPUT_FILE)