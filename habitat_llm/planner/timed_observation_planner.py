#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
Timed Observation Planner - Robot navigates to observation points every 10 seconds
based on the human's current location.
"""
from habitat_llm.planner.scripted_centralized_planner import ScriptedCentralizedPlanner
from habitat_llm.world_model import Room


class TimedObservationPlanner(ScriptedCentralizedPlanner):
    """
    A planner where the robot navigates to observation points every 10 seconds
    based on the human's current room location.
    """

    def __init__(self, plan_config, env_interface):
        super().__init__(plan_config, env_interface)

        # Action-based observation parameters (every N human actions)
        self.observation_interval_actions = (
            3  # Navigate every 3 human actions (balanced frequency)
        )
        self.human_actions_since_last_nav = 0

        # Track human's current room
        self.human_current_room = None
        self.robot_target_furniture = None

        # Define predefined observation furniture for each room
        self.room_observation_furniture = {
            "bedroom_1": "chair_16",
            "bathroom_1": "chair_23",
            "living_room_1": "stand_55",
            "bedroom_2": "table_13",
            "bathroom_2": "unknown_14",
            "kitchen_1": "counter_19",
            "laundryroom_1": "washer_dryer_11",
            "entryway_1": "bench_51",
            "hallway_1": "table_30",
        }

    def reset(self):
        """Reset planner for a new episode."""
        super().reset()
        self.human_actions_since_last_nav = 0
        self.human_current_room = None
        self.robot_target_furniture = None
        self.robot_is_navigating = False  # Track if robot is currently navigating

    def get_human_current_room(self, environment_graph):
        """
        Determine which room the human (agent_1) is currently in based on their position.
        """
        # Get human agent position from the environment
        try:
            sim = self.env_interface.env.env.env._env.sim
            human_agent_idx = 1  # agent_1 is the human
            sim.agents_mgr[human_agent_idx].articulated_agent.base_pos

            # Find the closest room to the human's position
            all_rooms = environment_graph.get_all_rooms()

            if not all_rooms:
                return None

            # For simplicity, we'll check which floor the human is on and infer the room
            # In a more sophisticated implementation, you'd check spatial proximity
            # For now, we'll track based on the human's last action target

            return self.human_current_room

        except Exception as e:
            print(f"Error getting human current room: {e}")
            return None

    def update_human_room_from_action(self, action):
        """
        Update the tracked human room based on their action.
        """
        if not isinstance(action, tuple) or len(action) < 2:
            return

        action_type = action[0]
        target = action[1]

        old_room = self.human_current_room

        # Get world graph from env_interface
        world_graph = self.env_interface.full_world_graph

        # Get object-to-furniture mapping
        sim = self.env_interface.env.env.env._env.sim
        object_to_furniture = {}
        if hasattr(sim, "ep_info") and hasattr(sim.ep_info, "name_to_receptacle"):
            sim_handle_to_name = self.env_interface.perception.sim_handle_to_name
            for obj_handle, fur_handle in sim.ep_info.name_to_receptacle.items():
                if obj_handle in sim_handle_to_name:
                    obj_name = sim_handle_to_name[obj_handle]
                    if fur_handle in sim_handle_to_name:
                        fur_name = sim_handle_to_name[fur_handle]
                        object_to_furniture[obj_name] = fur_name
                    elif "floor" in fur_handle:
                        object_to_furniture[obj_name] = fur_handle

        location = None

        if action_type == "Navigate":
            location = target
        elif action_type == "Place" and "," in target:
            parts = target.split(",")
            if len(parts) >= 3:
                location = parts[2].strip()

        # Find room for this location
        if location:
            try:
                location_node = world_graph.get_node_from_name(location)
                if location_node in world_graph.graph:
                    neighbors = world_graph.graph[location_node]

                    for neighbor_node, edge_label in neighbors.items():
                        if edge_label == "inside" and isinstance(neighbor_node, Room):
                            self.human_current_room = neighbor_node.name
                            return

                    # Check if it's an object, find its furniture, then room
                    if location in object_to_furniture:
                        furniture_name = object_to_furniture[location]
                        if "floor" in furniture_name:
                            self.human_current_room = furniture_name.replace(
                                "floor_", ""
                            )
                            return
                        elif furniture_name.startswith("rec_"):
                            base_furniture_name = furniture_name[4:].rsplit("_", 1)[0]
                            try:
                                furniture_node = world_graph.get_node_from_name(
                                    base_furniture_name
                                )
                                if furniture_node in world_graph.graph:
                                    for room_neighbor, room_edge in world_graph.graph[
                                        furniture_node
                                    ].items():
                                        if room_edge == "inside" and isinstance(
                                            room_neighbor, Room
                                        ):
                                            self.human_current_room = room_neighbor.name
                                            return
                            except ValueError:
                                pass
            except ValueError:
                pass

        # Debug output if room changed
        if self.human_current_room != old_room:
            print(
                f"[Room Update] Human room changed from {old_room} to {self.human_current_room} (action: {action_type}[{target}])"
            )

    def should_robot_navigate(self):
        """
        Check if enough human actions have passed since the last observation.
        Navigate every N human actions.
        """
        return self.human_actions_since_last_nav >= self.observation_interval_actions

    def get_robot_observation_target(self):
        """
        Get the furniture target for robot observation in the human's current room.
        """
        if (
            self.human_current_room
            and self.human_current_room in self.room_observation_furniture
        ):
            return self.room_observation_furniture[self.human_current_room]
        return None

    def get_plan_dag(self):
        """
        Generate task plan with human doing all tasks, robot observing on a timer.
        """
        curr_episode = self.env_interface.env.env.env._env.current_episode

        prop_dependencies = curr_episode.evaluation_proposition_dependencies
        propositions = curr_episode.evaluation_propositions

        # Force all eval tasks to human (agent_1)
        self.actions_per_agent = self.prop_res.solve_dag(
            propositions,
            curr_episode.evaluation_constraints,
            prop_dependencies,
            self.env_interface.full_world_graph,
            self.env_interface,
            force_human_only=True,
        )

        # Robot (agent_0) will have empty actions initially
        # Actions will be generated dynamically based on timer
        self.actions_per_agent[0] = []

        print(f"\nHuman actions ({len(self.actions_per_agent[1])} total):")
        for i, act in enumerate(self.actions_per_agent[1]):
            print(f"  {i}: {act}")
        print(
            f"\nRobot will navigate to observation points every {self.observation_interval_actions} human actions"
        )

    def get_next_action(self, instruction, observations, world_graph):
        """
        Get next action for each agent, with robot navigating every N human actions.
        """

        should_stop = False
        replan_required = {agent.uid: False for agent in self.agents}

        if self.trace == "":
            pass

        if self.actions_per_agent is None:
            self.get_plan_dag()
            # Initialize human room from first action
            if len(self.actions_per_agent[1]) > 0:
                first_human_action = self.actions_per_agent[1][0]
                self.update_human_room_from_action(first_human_action)

        if len(self.curr_hist) == 0:
            # Initialize history
            pass

        planner_info = {"replanned": dict.fromkeys(replan_required, False)}
        print_str = ""

        replanned = dict.fromkeys(self.agent_indices, False)

        # Build high level actions for both agents
        high_level_actions = {}

        # Handle human (agent_1) actions
        human_done = self.plan_indx[1] >= len(self.actions_per_agent[1])

        if not human_done:
            if self.next_skill_agents[1]:
                self.plan_indx[1] += 1
                self.next_skill_agents[1] = False
                replanned[1] = True  # Mark that human has a new action
                self.human_actions_since_last_nav += 1  # Count human actions

            if self.plan_indx[1] < len(self.actions_per_agent[1]):
                human_action = self.actions_per_agent[1][self.plan_indx[1]]

                # Update human's room based on their action
                self.update_human_room_from_action(human_action)

                # Add to high level actions (always add, even if it's Wait)
                high_level_actions[1] = human_action
                self.last_high_level_actions[1] = human_action
            else:
                # Human is done, set wait action
                self.last_high_level_actions[1] = ("Wait", "", "")
        else:
            # Human is done, set wait action
            self.last_high_level_actions[1] = ("Wait", "", "")

        # Handle robot (agent_0) actions with action-based navigation
        # Check if robot's previous action completed (next_skill_agents[0] is True when action finished)
        if self.next_skill_agents[0] and self.robot_is_navigating:
            print(
                f"[Navigation Complete] Robot finished navigation (after {self.human_actions_since_last_nav} human actions)"
            )
            self.robot_is_navigating = False
            self.next_skill_agents[0] = False  # Reset for next action

        # Only start new navigation if robot is not currently navigating and enough actions passed
        if not self.robot_is_navigating and self.should_robot_navigate():
            # Time to navigate to observation point
            observation_target = self.get_robot_observation_target()

            if observation_target:
                print(
                    f"[Action Count: {self.human_actions_since_last_nav}] Robot starting navigation to: {observation_target} in room {self.human_current_room}"
                )
                robot_action = ("Navigate", observation_target, "")
                high_level_actions[0] = robot_action
                self.last_high_level_actions[0] = robot_action
                replanned[0] = True  # Mark that robot has a new action
                self.robot_is_navigating = True  # Mark that navigation started
                self.robot_target_furniture = observation_target  # Store target

                # Reset action counter
                self.human_actions_since_last_nav = 0
            else:
                # No valid observation target, robot waits
                print(
                    f"[Action Count: {self.human_actions_since_last_nav}] Robot waiting - no observation target (human_room: {self.human_current_room})"
                )
                robot_action = ("Wait", "0", "")
                high_level_actions[0] = robot_action
                self.last_high_level_actions[0] = robot_action
                replanned[0] = True
        elif self.robot_is_navigating:
            # Robot is currently navigating - continue the navigation action
            # MUST re-add the navigation action so it continues executing
            if hasattr(self, "robot_target_furniture") and self.robot_target_furniture:
                robot_action = ("Navigate", self.robot_target_furniture, "")
                high_level_actions[0] = robot_action
                self.last_high_level_actions[0] = robot_action
                replanned[0] = False  # Not a new action, continuing existing one
        else:
            # Not enough actions yet - robot waits
            robot_action = ("Wait", "0", "")
            high_level_actions[0] = robot_action
            self.last_high_level_actions[0] = robot_action
            replanned[0] = True

        # Execute high level actions using parent class method
        if len(high_level_actions) == 0:
            should_stop = human_done
            responses = {}
            low_level_actions = {}
        else:
            # Remove Wait actions before processing (Wait is not executable)
            high_level_actions_to_execute = {
                agent: agent_action
                for agent, agent_action in high_level_actions.items()
                if agent_action[0].lower() != "wait"
            }

            # Process high level actions to get responses
            low_level_actions, responses = self.process_high_level_actions(
                high_level_actions_to_execute, observations
            )

            # Update next_skill_agents based on responses
            for agent_id, resp in responses.items():
                self.next_skill_agents[int(agent_id)] = len(resp) > 0

        # Add responses for agents that don't have them (e.g., agents doing Wait)
        # Also ensure no response is None or empty string
        for agent_uid in self.agent_indices:
            if agent_uid not in responses or not responses[agent_uid]:
                hl_action = self.last_high_level_actions.get(agent_uid)
                if hl_action and isinstance(hl_action, tuple) and len(hl_action) > 0:
                    response = f"Action {hl_action[0]}[{hl_action[1] if len(hl_action) > 1 else ''}] is still in progress."
                else:
                    response = "Action in progress."
                responses[agent_uid] = response

        # Check if task is done
        if human_done:
            should_stop = True

        # Build output string
        obs_str = ""
        for agent_uid in sorted(self.last_high_level_actions.keys()):
            agent_name = f"Agent_{agent_uid}"
            hl_action = self.last_high_level_actions[agent_uid]
            if isinstance(hl_action, tuple) and len(hl_action) >= 2:
                action_str = f"{hl_action[0]}[{hl_action[1]}]"
            else:
                action_str = str(hl_action)

            # Debug: Show detailed action info
            in_hl_actions = agent_uid in high_level_actions
            was_replanned = replanned[agent_uid]
            is_navigating = (
                "(navigating)" if (agent_uid == 0 and self.robot_is_navigating) else ""
            )
            action_count_info = (
                f"(human_actions={self.human_actions_since_last_nav})"
                if agent_uid == 0
                else ""
            )
            print_str += f"{agent_name}_Action: {action_str} {is_navigating}{action_count_info} [in_hl={in_hl_actions}, replanned={was_replanned}]\n"

            # Add response if available
            if agent_uid in responses:
                response = responses[agent_uid]
                obs_str += f"{agent_name}_observation: {response}\n"
            else:
                obs_str += f"{agent_name}_observation: Action in progress\n"

        self.curr_hist = self.curr_hist + print_str + obs_str

        if should_stop:
            print("Task completed!")

        planner_info["responses"] = responses
        if len(print_str + obs_str) > 0:
            planner_info["chat"] = print_str + obs_str
        planner_info["high_level_actions"] = self.last_high_level_actions
        planner_info["prompts"] = {agent.uid: self.curr_hist for agent in self.agents}
        planner_info["replan_required"] = replan_required
        planner_info["is_done"] = {agent.uid: should_stop for agent in self.agents}
        planner_info["traces"] = {agent.uid: self.trace for agent in self.agents}
        planner_info["replanned"] = replanned
        planner_info["actions_per_agent"] = self.actions_per_agent

        # Update world based on actions
        self.update_world(responses)

        return low_level_actions, planner_info, should_stop
