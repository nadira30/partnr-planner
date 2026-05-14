#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Any, Dict


# append the path of the
# parent directory
sys.path.append("..")

import omegaconf
import hydra

from hydra.utils import instantiate

from habitat_llm.utils import cprint, setup_config, fix_config

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)

from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat_llm.utils.sim import init_agents
from habitat_llm.examples.example_utils import execute_skill, DebugVideoUtil
from habitat_llm.utils.world_graph import (
    print_all_entities,
    print_furniture_entity_handles,
    print_object_entity_handles,
)


def _load_commands_from_file(
    commands_file: str,
    valid_skill_names: List[str],
    valid_control_commands: List[str],
) -> List[str]:
    """Load scripted commands from a text file and skip non-command lines."""
    commands_path = Path(commands_file).expanduser()
    if not commands_path.exists():
        raise FileNotFoundError(f"Command file not found: {commands_path}")

    valid_skill_set = set(valid_skill_names)
    valid_control_set = set(valid_control_commands)

    commands: List[str] = []
    skipped_line_count = 0
    for line in commands_path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        # support bullets/numbered lists (e.g., "- Navigate ..." or "1. Navigate ...")
        normalized_line = re.sub(r"^([-*]\s+|\d+[.)]\s+)", "", stripped_line).strip()
        if not normalized_line:
            continue

        first_token = normalized_line.split(" ")[0]

        if first_token in valid_control_set and " " not in normalized_line:
            commands.append(normalized_line)
            continue

        if first_token in valid_skill_set:
            commands.append(normalized_line)
            continue

        skipped_line_count += 1

    if skipped_line_count > 0:
        print(
            f"Skipped {skipped_line_count} non-command lines while reading command file: {commands_path}"
        )

    return commands


def _should_follow_human_navigate(
    human_command: str, human_agent_uid: int = 1
) -> bool:
    """
    Return True when command is a human Navigate command.

    :param human_command: The human's skill command (e.g. "Navigate 1 floor_bedroom_1")
    :param human_agent_uid: The UID of the human agent (default 1)
    :return: True iff command is human Navigate
    """
    components = human_command.split()
    if len(components) < 3:
        return False

    skill_name = components[0]
    try:
        agent_uid = int(components[1])
    except (ValueError, IndexError):
        return False

    return skill_name == "Navigate" and agent_uid == human_agent_uid


def _get_robot_room_anchor_from_human_location(
    env_interface: EnvironmentInterface,
    human_agent_uid: int = 1,
) -> str:
    """
    Resolve predefined robot target anchor in the human's current room.

    Uses room of human agent in world graph and maps it to floor anchor:
    <room_name> -> floor_<room_name>
    """
    try:
        human_wg = env_interface.world_graph[human_agent_uid]
        human_node = human_wg.get_human()
        room_node = human_wg.get_room_for_entity(human_node)
        anchor_name = f"floor_{room_node.name}"
        human_wg.get_node_from_name(anchor_name)
        return anchor_name
    except Exception:
        return ""


def _get_room_anchor_from_human_navigate_target(
    env_interface: EnvironmentInterface,
    navigate_target_name: str,
    human_agent_uid: int = 1,
) -> str:
    """Resolve floor anchor for the room containing the human Navigate target entity."""
    try:
        human_wg = env_interface.world_graph[human_agent_uid]
        target_node = human_wg.get_node_from_name(navigate_target_name)
        room_node = human_wg.get_room_for_entity(target_node)
        anchor_name = f"floor_{room_node.name}"
        human_wg.get_node_from_name(anchor_name)
        return anchor_name
    except Exception:
        return ""


def _orient_robot_toward_human(
    env_interface: EnvironmentInterface,
    robot_agent_uid: int = 0,
    human_agent_uid: int = 1,
) -> bool:
    """Rotate robot base in-place to face the human's current position."""
    try:
        robot_agent = env_interface.sim.agents_mgr[robot_agent_uid].articulated_agent
        human_agent = env_interface.sim.agents_mgr[human_agent_uid].articulated_agent

        robot_pos = np.array(robot_agent.base_pos)
        human_pos = np.array(human_agent.base_pos)
        delta = human_pos - robot_pos
        delta_xz = delta[[0, 2]]
        norm = np.linalg.norm(delta_xz)
        if norm < 1e-6:
            return False

        unit_dir = delta_xz / norm
        target_yaw = float(np.arctan2(-unit_dir[1], unit_dir[0]))
        robot_agent.base_rot = target_yaw
        return True
    except Exception:
        return False


def _enforce_robot_human_separation(
    env_interface: EnvironmentInterface,
    min_distance: float,
    robot_agent_uid: int = 0,
    human_agent_uid: int = 1,
) -> bool:
    """Ensure robot and human keep at least min_distance in the XZ plane."""
    try:
        robot_agent = env_interface.sim.agents_mgr[robot_agent_uid].articulated_agent
        human_agent = env_interface.sim.agents_mgr[human_agent_uid].articulated_agent

        robot_pos = np.array(robot_agent.base_pos, dtype=float)
        human_pos = np.array(human_agent.base_pos, dtype=float)

        delta = robot_pos - human_pos
        delta_xz = delta[[0, 2]]
        cur_dist = float(np.linalg.norm(delta_xz))

        if cur_dist >= min_distance:
            return False

        if cur_dist < 1e-6:
            direction_xz = np.array([1.0, 0.0], dtype=float)
        else:
            direction_xz = delta_xz / cur_dist

        new_robot_pos = robot_pos.copy()
        new_robot_pos[0] = human_pos[0] + direction_xz[0] * min_distance
        new_robot_pos[2] = human_pos[2] + direction_xz[1] * min_distance

        try:
            snapped = env_interface.sim.pathfinder.snap_point(new_robot_pos)
            if snapped is not None and np.all(np.isfinite(snapped)):
                new_robot_pos = np.array(snapped, dtype=float)
        except Exception:
            pass

        robot_agent.base_pos = new_robot_pos
        return True
    except Exception:
        return False


def _is_robot_at_anchor(
    env_interface: EnvironmentInterface,
    anchor_name: str,
    robot_agent_uid: int = 0,
    distance_thresh: float = 0.35,
) -> bool:
    """Return True when robot base is already close to the anchor in XZ plane."""
    try:
        robot_agent = env_interface.sim.agents_mgr[robot_agent_uid].articulated_agent
        robot_pos = np.array(robot_agent.base_pos, dtype=float)

        robot_wg = env_interface.world_graph[robot_agent_uid]
        anchor_node = robot_wg.get_node_from_name(anchor_name)
        anchor_translation = anchor_node.properties.get("translation", None)
        if anchor_translation is None:
            return False
        anchor_pos = np.array(anchor_translation, dtype=float)

        dist_xz = float(np.linalg.norm((robot_pos - anchor_pos)[[0, 2]]))
        return dist_xz <= distance_thresh
    except Exception:
        return False


# Method to load agent planner from the config
@hydra.main(
    config_path="../conf", config_name="examples/skill_runner_default_config.yaml"
)
def run_skills(config: omegaconf.DictConfig) -> None:
    """
    The main function for executing the skill_runner tool. A default config is provided.
    See the `main` function for example CLI command to run the tool.

    :param config: input is a habitat-llm config from Hydra. Can contain CLI overrides.
    """
    fix_config(config)
    # Setup a seed
    seed = 47668090
    # Setup some hardcoded config overrides (e.g. the metadata path)
    with omegaconf.open_dict(config):
        config_dict = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
        )
        config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
        config.habitat = config_dict
    config = setup_config(config, seed)

    assert config.env == "habitat", "Only valid for Habitat skill testing."

    # whether or not to show blocking videos after each command call
    show_command_videos = (
        config.skill_runner_show_videos
        if hasattr(config, "skill_runner_show_videos")
        else True
    )
    # make videos only if showing or saving them
    make_video = config.evaluation.save_video or show_command_videos

    if not make_video:
        remove_visual_sensors(config)

    # We register the dynamic habitat sensors
    register_sensors(config)

    # We register custom actions
    register_actions(config)

    # We register custom measures
    register_measures(config)

    # create the dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    print(f"Loading EpisodeDataset from: {config.habitat.dataset.data_path}")
    # Initialize the environment interface for the agent
    env_interface = EnvironmentInterface(config, dataset=dataset)

    ##########################################
    # select and initialize the desired episode by index or id
    # NOTE: use "+skill_runner_episode_index=2" in CLI to set the episode index ( e.g. episode 2)
    # NOTE: use "+skill_runner_episode_id=<id>" in CLI to set the episode id ( e.g. episode "")
    assert not (
        hasattr(config, "skill_runner_episode_index")
        and hasattr(config, "skill_runner_episode_id")
    ), "Episode selection options are mutually exclusive."
    if hasattr(config, "skill_runner_episode_index"):
        episode_index = config.skill_runner_episode_index
        print(f"Loading episode_index = {episode_index}")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_index(
            episode_index
        )
    elif hasattr(config, "skill_runner_episode_id"):
        episode_id = config.skill_runner_episode_id
        print(f"Loading episode_id = {episode_id}")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(
            str(episode_id)
        )
    env_interface.reset_environment()
    ###########################################

    # Initialize planner(s)
    agent_config = config.evaluation.agents
    initialized_agents = init_agents(agent_config, env_interface)
    agents_by_uid = {agent.uid: agent for agent in initialized_agents}

    planner = None
    planners_by_uid = {}
    if hasattr(config.evaluation, "planner"):
        # Centralized setup
        planner_conf = config.evaluation.planner
        planner = instantiate(planner_conf)
        planner = planner(env_interface=env_interface)
        planner.agents = initialized_agents
        planner.reset()
    else:
        # Decentralized setup: each agent has its own planner config
        for agent_name, agent_eval_conf in config.evaluation.agents.items():
            if not hasattr(agent_eval_conf, "planner"):
                raise ValueError(
                    f"Missing planner config for {agent_name} in decentralized evaluation setup."
                )
            planner_conf = agent_eval_conf.planner
            agent_planner = instantiate(planner_conf)
            agent_planner = agent_planner(env_interface=env_interface)
            agent_uid = int(agent_eval_conf.uid)
            if agent_uid not in agents_by_uid:
                raise ValueError(f"No initialized agent found for uid {agent_uid}.")
            agent_planner.agents = [agents_by_uid[agent_uid]]
            agent_planner.reset()
            planners_by_uid[agent_uid] = agent_planner

    sim = env_interface.sim

    ############################
    # done with setup, prompt the user and start running skills

    # available skills
    skills = {
        "Navigate": "Navigate <agent_index> <entity_name>",
        "Open": "Open <agent_index> <entity_name>",
        "Close": "Close <agent_index> <entity_name>",
        "Pick": "Pick <agent_index> <entity_name>",
        # Place skill requires 5 arguments, comma separated, no spaces:
        "Place": "Place <agent_index> <entity_name_0,relation_0,entity_name_1,relation_1,entity_name_2>",
        "Wait": "Wait <agent_index> <wait_time_seconds>",
    }
    exit_skill = "exit"
    help_skill = "help"
    entity_skill = "entities"
    pdb_skill = "debug"
    cumulative_video_skill = "make_video"
    control_commands = [
        exit_skill,
        help_skill,
        entity_skill,
        pdb_skill,
        cumulative_video_skill,
    ]

    cprint("Welcome to skill_runner!", "green")
    cprint(
        f"Current Episode (id=={sim.ep_info.episode_id}) is running in scene {sim.ep_info.scene_id} with info: {sim.ep_info.info}.",
        "green",
    )

    print_all_entities(env_interface.perception.gt_graph)
    print_furniture_entity_handles(env_interface.perception.gt_graph)
    print_object_entity_handles(env_interface.perception.gt_graph)

    help_text = f"Available skills are {skills}. Type a skill to begin.\n alternatively type one of: \n  '{exit_skill}' - exit the program \n  '{help_skill}' - display help text \n  '{entity_skill}' - display all available entities \n  '{pdb_skill}' - enter pdb breakpoint for interactive debugging \n  '{cumulative_video_skill}' - make a single cumulative video out of all individual command clips"
    cprint(help_text, "green")

    # setup a sequence of commands to run immediately without manual input
    scripted_commands: List[str] = []
    has_scripted_command_list = hasattr(config, "skill_runner_scripted_commands")
    has_scripted_command_file = hasattr(config, "skill_runner_commands_file")
    assert not (
        has_scripted_command_list and has_scripted_command_file
    ), "skill_runner_scripted_commands and skill_runner_commands_file are mutually exclusive."

    if has_scripted_command_file:
        scripted_commands = _load_commands_from_file(
            config.skill_runner_commands_file,
            valid_skill_names=list(skills.keys()),
            valid_control_commands=control_commands,
        )
        print(
            f"Loaded {len(scripted_commands)} scripted commands from file: {config.skill_runner_commands_file}"
        )

    if hasattr(config, "skill_runner_scripted_commands"):
        scripted_commands = config.skill_runner_scripted_commands
        # we need special handling for "Place" skill because arguements are comma separated and need to be joined
        place_indices = [i for i, x in enumerate(scripted_commands) if "Place" in x]
        for i, place_ix in enumerate(place_indices):
            corrected_ix = place_ix - i * 4  # account for removed elements
            for j in range(1, 5):
                # concat the elements
                scripted_commands[corrected_ix] += (
                    "," + scripted_commands[corrected_ix + j]
                )
            scripted_commands = (
                scripted_commands[: corrected_ix + 1]
                + scripted_commands[corrected_ix + 5 :]
            )
    print(scripted_commands)

    scripted_commands_from_file = has_scripted_command_file
    interactive_playback = show_command_videos and not scripted_commands_from_file
    write_individual_skill_videos = make_video and not scripted_commands_from_file
    if scripted_commands_from_file and show_command_videos:
        cprint(
            "skill_runner_commands_file detected: disabling interactive video playback for non-interactive run.",
            "yellow",
        )
    if scripted_commands_from_file and make_video:
        cprint(
            "skill_runner_commands_file detected: deferring video encoding to final cumulative output.",
            "yellow",
        )

    # show the topdown map if requested
    if hasattr(config, "skill_runner_show_topdown"):
        dbv = DebugVisualizer(sim, config.paths.results_dir)
        dbv.create_dbv_agent(resolution=(1000, 1000))
        top_down_map = dbv.peek("stage")
        if interactive_playback:
            top_down_map.show()
        if config.evaluation.save_video:
            top_down_map.save(output_path=config.paths.results_dir, prefix="topdown")
        dbv.remove_dbv_agent()
        dbv.create_dbv_agent()
        dbv.remove_dbv_agent()

    auto_save_scripted_video = (
        config.skill_runner_auto_save_scripted_video
        if hasattr(config, "skill_runner_auto_save_scripted_video")
        else True
    )

    robot_human_min_distance = (
        float(config.skill_runner_robot_human_min_distance)
        if hasattr(config, "skill_runner_robot_human_min_distance")
        else 1.0
    )

    # collect debug frames to create a final video
    cumulative_frames: List[Any] = []

    # Pending robot follow anchor; executed concurrently with human commands without blocking human completion
    pending_robot_anchor: str = ""

    command_index = 0
    # history of skill commands and their responses
    command_history: List[Tuple[str, str]] = []

    def _print_history() -> None:
        print("==========================")
        print("Exiting. Command History:")
        for ix, t in enumerate(command_history):
            print(f" [{ix}]: '{t[0]}' -> '{t[1]}'")
        print("==========================")

    def _maybe_save_cumulative_video() -> None:
        if not auto_save_scripted_video:
            return
        if len(cumulative_frames) == 0:
            print("No cumulative frames recorded; skipping final video generation.")
            return
        dvu = DebugVideoUtil(env_interface, env_interface.conf.paths.results_dir)
        dvu.frames = cumulative_frames
        dvu._make_video(postfix="cumulative", play=interactive_playback)

    while True:
        cprint("Enter Command", "blue")
        if scripted_commands_from_file and len(scripted_commands) <= command_index:
            cprint(
                "Reached end of scripted command file. Saving final cumulative video and exiting.",
                "green",
            )
            _maybe_save_cumulative_video()
            _print_history()
            return

        high_level_skill_actions = {}  # Reset for each command
        if len(scripted_commands) > command_index:
            user_input = scripted_commands[command_index]
            print(user_input)
        else:
            user_input = input("> ")

        selected_skill = None

        if user_input == exit_skill:
            _maybe_save_cumulative_video()
            _print_history()
            return
        elif user_input == help_skill:
            cprint(help_text, "green")
        elif user_input == entity_skill:
            print_all_entities(env_interface.perception.gt_graph)
        elif user_input == pdb_skill:
            # peek an entity
            dbv = DebugVisualizer(sim, config.paths.results_dir)
            dbv.create_dbv_agent()
            # NOTE: do debugging calls here
            # example to peek an entity: dbv.peek(env_interface.world_graph.get_node_from_name('table_50').sim_handle).show()
            breakpoint()
            dbv.remove_dbv_agent()
        elif user_input == cumulative_video_skill:
            # create a video of all accumulated frames thus far and play it
            if len(cumulative_frames) > 0:
                dvu = DebugVideoUtil(
                    env_interface, env_interface.conf.paths.results_dir
                )
                dvu.frames = cumulative_frames
                dvu._make_video(postfix="cumulative", play=interactive_playback)
        elif user_input in skills:
            # fill information piece by piece
            selected_skill = user_input
            # get the agent index
            agent_ix = input("Agent Index (0=robot, 1=human) = ")
            if agent_ix not in ["0", "1"]:
                cprint("... invalid Agent Index, aborting.", "red")
                command_index += 1
                continue
            target_entity_name = input("Target Entity = ")
            high_level_skill_actions = {
                int(agent_ix): (selected_skill, target_entity_name, None)
            }
        elif user_input.split(" ")[0] in skills:
            # attempt to parse full skill definition from string
            skill_components = user_input.split(" ")
            selected_skill = skill_components[0]
            agent_ix = skill_components[1]
            if agent_ix not in ["0", "1"]:
                cprint("... invalid Agent Index, aborting.", "red")
                command_index += 1
                continue
            target_entity_name = skill_components[2]
            high_level_skill_actions = {
                int(agent_ix): (selected_skill, target_entity_name, None)
            }
        else:
            cprint("... invalid command.", "red")
            command_index += 1
            continue

        # configure and run the skill
        if high_level_skill_actions:

            ############################
            # run the skill
            try:
                # Get the agent for this command
                agent_idx = list(high_level_skill_actions.keys())[0]
                skill_name = high_level_skill_actions[agent_idx][0]
                should_follow_human = _should_follow_human_navigate(
                    user_input, human_agent_uid=1
                )

                # Step 1: Execute the human/user command first
                # For decentralized planning, route through the agent's own planner
                if planners_by_uid:
                    if agent_idx not in planners_by_uid:
                        cprint(
                            f"... no planner found for agent {agent_idx}, aborting.",
                            "red",
                        )
                        command_index += 1
                        continue
                    active_planner = planners_by_uid[agent_idx]
                else:
                    # Centralized planner
                    active_planner = planner

                actions_to_run = dict(high_level_skill_actions)
                if agent_idx == 1 and pending_robot_anchor:
                    if _is_robot_at_anchor(
                        env_interface,
                        pending_robot_anchor,
                        robot_agent_uid=0,
                        distance_thresh=0.35,
                    ):
                        cprint(
                            f"Robot already at anchor {pending_robot_anchor}; clearing pending navigation.",
                            "yellow",
                        )
                        separated = _enforce_robot_human_separation(
                            env_interface,
                            min_distance=robot_human_min_distance,
                            robot_agent_uid=0,
                            human_agent_uid=1,
                        )
                        if separated:
                            cprint(
                                f"Adjusted robot position to keep distance >= {robot_human_min_distance:.2f}m from human.",
                                "yellow",
                            )
                        oriented = _orient_robot_toward_human(
                            env_interface, robot_agent_uid=0, human_agent_uid=1
                        )
                        if oriented:
                            cprint("Auto-oriented robot to face human.", "yellow")
                        pending_robot_anchor = ""
                    else:
                        actions_to_run[0] = ("Navigate", pending_robot_anchor, None)
                        cprint(
                            f"Robot continuing non-blocking navigation to: {pending_robot_anchor}",
                            "yellow",
                        )

                responses, _, frames = execute_skill(
                    actions_to_run,
                    active_planner,
                    vid_postfix=f"{command_index}_",
                    make_video=make_video,
                    play_video=interactive_playback,
                    write_video=write_individual_skill_videos,
                    decentralized_planners=planners_by_uid if planners_by_uid else None,
                    blocking_agent_ids=[agent_idx],
                )
                cumulative_frames.extend(frames)

                skill_name = high_level_skill_actions[agent_idx][0]
                skill_response = responses.get(agent_idx, "")
                target_entity_name = high_level_skill_actions[agent_idx][1]

                pick_not_close_failure = (
                    skill_name == "Pick"
                    and isinstance(skill_response, str)
                    and "Failed to pick" in skill_response
                    and "Not close enough to the object" in skill_response
                )

                if pick_not_close_failure:
                    cprint(
                        f"Pick failed because agent is not close to {target_entity_name}. Auto-recovering with Navigate + Pick retry.",
                        "yellow",
                    )

                    nav_actions = {agent_idx: ("Navigate", target_entity_name, None)}
                    nav_responses, _, nav_frames = execute_skill(
                        nav_actions,
                        active_planner,
                        vid_postfix=f"{command_index}_autonav_",
                        make_video=make_video,
                        play_video=interactive_playback,
                        write_video=write_individual_skill_videos,
                        decentralized_planners=planners_by_uid if planners_by_uid else None,
                        blocking_agent_ids=[agent_idx],
                    )
                    cumulative_frames.extend(nav_frames)
                    print(
                        f"Auto Navigate completed. Response = '{nav_responses.get(agent_idx, '')}'"
                    )

                    retry_actions = {agent_idx: ("Pick", target_entity_name, None)}
                    retry_responses, _, retry_frames = execute_skill(
                        retry_actions,
                        active_planner,
                        vid_postfix=f"{command_index}_autoretry_pick_",
                        make_video=make_video,
                        play_video=interactive_playback,
                        write_video=write_individual_skill_videos,
                        decentralized_planners=planners_by_uid if planners_by_uid else None,
                        blocking_agent_ids=[agent_idx],
                    )
                    cumulative_frames.extend(retry_frames)
                    skill_response = retry_responses.get(agent_idx, skill_response)

                # Place skill recovery: navigate to furniture if not close enough, then retry place
                place_not_close_failure = (
                    skill_name == "Place"
                    and isinstance(skill_response, str)
                    and "Failed to place" in skill_response
                    and "Not close enough to" in skill_response
                )

                if place_not_close_failure:
                    # Extract furniture name from place command
                    # Place command format: (Place, "object,on,furniture,spatial_relation,spatial_constraint")
                    place_args = high_level_skill_actions[agent_idx][1]
                    furniture_name = None
                    if isinstance(place_args, str) and "," in place_args:
                        parts = place_args.split(",")
                        if len(parts) >= 3:
                            furniture_name = parts[2]  # furniture is the 3rd element
                    
                    if furniture_name:
                        cprint(
                            f"Place failed because agent is not close to {furniture_name}. Auto-recovering with Navigate + Place retry.",
                            "yellow",
                        )

                        nav_actions = {agent_idx: ("Navigate", furniture_name, None)}
                        nav_responses, _, nav_frames = execute_skill(
                            nav_actions,
                            active_planner,
                            vid_postfix=f"{command_index}_autonav_",
                            make_video=make_video,
                            play_video=interactive_playback,
                            write_video=write_individual_skill_videos,
                            decentralized_planners=planners_by_uid if planners_by_uid else None,
                            blocking_agent_ids=[agent_idx],
                        )
                        cumulative_frames.extend(nav_frames)
                        print(
                            f"Auto Navigate completed. Response = '{nav_responses.get(agent_idx, '')}'"
                        )

                        retry_actions = {agent_idx: ("Place", place_args, None)}
                        retry_responses, _, retry_frames = execute_skill(
                            retry_actions,
                            active_planner,
                            vid_postfix=f"{command_index}_autoretry_place_",
                            make_video=make_video,
                            play_video=interactive_playback,
                            write_video=write_individual_skill_videos,
                            decentralized_planners=planners_by_uid if planners_by_uid else None,
                            blocking_agent_ids=[agent_idx],
                        )
                        cumulative_frames.extend(retry_frames)
                        skill_response = retry_responses.get(agent_idx, skill_response)

                command_history.append((user_input, skill_response))
                print(f"{skill_name} completed. Response = '{skill_response}'")

                if agent_idx == 1 and pending_robot_anchor and responses.get(0):
                    print(
                        f"Robot anchor navigate completed. Response = '{responses[0]}'"
                    )
                    separated = _enforce_robot_human_separation(
                        env_interface,
                        min_distance=robot_human_min_distance,
                        robot_agent_uid=0,
                        human_agent_uid=1,
                    )
                    if separated:
                        cprint(
                            f"Adjusted robot position to keep distance >= {robot_human_min_distance:.2f}m from human.",
                            "yellow",
                        )

                    oriented = _orient_robot_toward_human(
                        env_interface, robot_agent_uid=0, human_agent_uid=1
                    )
                    if oriented:
                        cprint("Auto-oriented robot to face human.", "yellow")
                    else:
                        cprint(
                            "Could not orient robot toward human at this step.",
                            "yellow",
                        )
                    pending_robot_anchor = ""

                # After human Navigate, send robot to predefined anchor in human's room.
                if should_follow_human:
                    robot_anchor = _get_robot_room_anchor_from_human_location(
                        env_interface, human_agent_uid=1
                    )
                    if robot_anchor:
                        pending_robot_anchor = robot_anchor
                        cprint(
                            f"Queued robot room-anchor navigation (non-blocking): Navigate 0 {robot_anchor}",
                            "yellow",
                        )
                    else:
                        cprint(
                            "Could not resolve human room anchor for robot navigation.",
                            "yellow",
                        )
            except Exception as e:
                failure_string = f"Failed to execute skill with exception: {str(e)}"
                print(failure_string)
                command_history.append((user_input, failure_string))
        command_index += 1


##########################################
# CLI Example:
# HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.skill_runner hydra.run.dir="."
# or
# python habitat_llm/examples/skill_runner.py
#
# NOTE: conf/examples/skill_runner_default_config.yaml is consumed to initialize parameters
# NOTE: use --config-name examples/skill_runner_decentralized_config.yaml to run decentralized planner setup
##########################################
# Script Specific CLI overrides:
#
# (mutually exclusive)
# - '+skill_runner_episode_index=0' - initialize the episode with the specified index within the dataset
# - '+skill_runner_episode_id=' - initialize the episode with the specified "id" within the dataset
#
# - '+skill_runner_show_topdown=True' - (default False) show a topdown view of the scene upon initialization for context
#
# (output control options)
# - '+skill_runner_show_videos=False' - (default True) turn off showing videos immediately after running a command
# - 'evaluation.save_video=False' - (default True) option to save videos to files. Also affects cumulative videos produced with "make_video" command.
# NOTE: videos are made only if either of the above options are True
# - 'paths.results_dir=<relative_path>' (default './results/') relative path to desired output directory for evaluation
# - '+skill_runner_auto_save_scripted_video=False' - (default True) disable automatic cumulative video generation when scripted commands complete
#
# (scripted input options)
# - '+skill_runner_commands_file=<path_to_text_file>' - run one command per line from a file (ignores empty lines and lines starting with '#').
#   At the end of the file, skill_runner automatically saves a cumulative video and exits.
#
##########################################
# Other useful CLI overrides:
#
# - 'habitat.dataset.data_path="<path to dataset .json.gz>"' - set the desired episode dataset
#
if __name__ == "__main__":
    cprint(
        "\nStart of the example program to run custom skill commands in a CollaborationEpisode.",
        "blue",
    )

    # Run the skills
    run_skills()

    cprint(
        "\nEnd of the example program to run custom skill commands in a CollaborationEpisode.",
        "blue",
    )
