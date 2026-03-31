#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import imageio
import numpy as np

from habitat_llm.agent.env import EnvironmentInterface


class DebugVideoUtil:
    """
    This class provides an interface wrapper for creating, saving, and viewing third person videos of individual skill runs using the EnvironmentInterface API.

    For example, see `execute_skill` function below.
    NOTE: This code was largely adapted from the evaluation_runner.py
    """

    def __init__(
        self, env_interface_arg: EnvironmentInterface, output_dir: str
    ) -> None:
        """
        Construct the DebugVideoUtil instance from an EnvironmentInterface.

        :param env_interface_arg: The EnvironmentInterface instance.
        :param output_dir: The desired directory for saving output frames and videos.
        """

        self.env_interface = env_interface_arg

        # Declare container to store frames used for generating video
        self.frames: List[Any] = []

        self.output_dir = output_dir

        self.num_agents = 0
        for _agent_conf in self.env_interface.conf.evaluation.agents.values():
            self.num_agents += 1

    def __get_combined_frames(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        For each agent, extract the observation from the "third_rgb" sensor and merge them into a single split-screen image.

        :param batch: A dict mapping observation names to values.
        :return: The composite image as a numpy array.
        """
        # Extract first agent frame
        images = []
        for obs_name, obs_value in batch.items():
            if "third_rgb" in obs_name:
                if self.num_agents == 1:
                    if "0" in obs_name or "main_agent" in obs_name:
                        images.append(obs_value)
                else:
                    images.append(obs_value)

        # Extract dimensions of the first image
        height, width = images[0].shape[1:3]

        # Create an empty canvas to hold the concatenated images
        concat_image = np.zeros((height, width * len(images), 3), dtype=np.uint8)

        # Iterate through the images and concatenate them horizontally
        for i, image in enumerate(images):
            concat_image[:, i * width : (i + 1) * width] = image.cpu()

        return concat_image

    def _store_for_video(
        self, observations: Dict[str, Any], hl_actions: Dict[int, Any]
    ) -> None:
        """
        Store a video with observations and text from an observation dict and an agent to action metadata dict.
        NOTE: Could probably go into utils?

        :param observations: A dict mapping observation names to values.
        :param hl_actions: A dict mapping agent action indices to actions.
        """
        frames_concat = self.__get_combined_frames(observations)
        frames_concat = np.ascontiguousarray(frames_concat)

        for idx, action in hl_actions.items():
            # text = f"Agent_{id}:{action[0]}[{action[1]}]"
            agent_name = "Human" if str(idx) == "1" else "Robot"
            text = f"{agent_name}: {action[0]}[{action[1]}]"
            frames_concat = cv2.putText(
                frames_concat,
                text,
                (20, (int(idx) + 1) * 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

        self.frames.append(frames_concat)
        return

    def _make_video(self, play: bool = True, postfix: str = "") -> None:
        """
        Makes a video from a pre-processed set of frames using imageio and saves it to the output directory.

        :param play: Whether or not to play the video immediately.
        :param postfix: An optional postfix for the video file name.
        """
        if len(self.frames) == 0:
            print("No frames available for video generation, skipping.")
            return

        out_file = f"{self.output_dir}/videos/video.mp4"  # -{postfix}
        print(f"Saving video to {out_file}")
        os.makedirs(f"{self.output_dir}/videos", exist_ok=True)

        # Ensure every frame has consistent shape and dtype before writing.
        reference_frame = np.asarray(self.frames[0])
        if reference_frame.ndim == 2:
            reference_frame = cv2.cvtColor(reference_frame, cv2.COLOR_GRAY2RGB)
        if reference_frame.ndim == 3 and reference_frame.shape[2] > 3:
            reference_frame = reference_frame[:, :, :3]
        reference_h, reference_w = reference_frame.shape[:2]

        writer = imageio.get_writer(
            out_file,
            fps=30,
            quality=4,
        )
        for frame in self.frames:
            processed_frame = np.asarray(frame)
            if processed_frame.ndim == 2:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
            if processed_frame.ndim != 3:
                continue
            if processed_frame.shape[2] > 3:
                processed_frame = processed_frame[:, :, :3]
            if processed_frame.shape[:2] != (reference_h, reference_w):
                processed_frame = cv2.resize(
                    processed_frame,
                    (reference_w, reference_h),
                    interpolation=cv2.INTER_AREA,
                )
            if processed_frame.dtype != np.uint8:
                processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
            processed_frame = np.ascontiguousarray(processed_frame)
            writer.append_data(processed_frame)

        writer.close()
        if play:
            print("     ...playing video, press 'q' to continue...")
            self.play_video(out_file)

    def play_video(self, filename: str) -> None:
        """
        Play and loop video from a filepath with cv2.

        :param filename: The filepath of the video.
        """
        cap = cv2.VideoCapture(filename)
        last_time = time.time()
        while cap.isOpened():
            if time.time() - last_time > 1.0 / 30:
                last_time = time.time()
                ret, frame = cap.read()
                # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

                if ret:
                    cv2.imshow("Image", frame)
                else:
                    # looping
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


def execute_skill(
    high_level_skill_actions: Dict[Any, Any],
    llm_env,
    make_video: bool = True,
    vid_postfix: str = "",
    play_video: bool = True,
    write_video: bool = True,
    decentralized_planners: Dict[int, Any] = None,
) -> Tuple[Dict[Any, Any], Dict[Any, Any], List[Any]]:
    """
    Execute a high-level skill from a string (e.g. as produced by the planner).
    Can create and display a video of the running skill.

    :param high_level_skill_actions: The map of agent indices to actions. TODO: typing
    :param llm_env: The planner instance. TODO: typing
    :param make_video: whether or not to create, save, and display a video of the skill.
    :param vid_postfix: An optional postfix for the video file. For example, the action name.
    :param play_video: Whether or not to immediately play the generated video.
    :param write_video: Whether or not to write a per-skill video file. Frame collection is still controlled by make_video.
    :param decentralized_planners: Optional map of agent uid -> planner for decentralized multi-agent stepping.
    :return: A tuple with two dict(the first contains responses per-agent skill, the second contains the number of skill steps taken) and a list of frames.
    """
    dvu = DebugVideoUtil(
        llm_env.env_interface, llm_env.env_interface.conf.paths.results_dir
    )

    # Get the env observations
    observations = llm_env.env_interface.get_observations()
    agent_idx = list(high_level_skill_actions.keys())[0]
    skill_name = high_level_skill_actions[agent_idx][0]
    assigned_agent_ids = list(high_level_skill_actions.keys())

    # Set up the variables
    skill_steps = 0
    max_skill_steps = 1500
    skill_done = None

    # While loop for executing skills
    while not skill_done:
        # Check if the maximum number of steps is reached
        assert (
            skill_steps < max_skill_steps
        ), f"Maximum number of steps reached: {skill_name} skill fails."

        # Get low level actions and responses
        if decentralized_planners:
            low_level_actions = {}
            responses = {}
            for agent_id, agent_action in high_level_skill_actions.items():
                if agent_id not in decentralized_planners:
                    responses[agent_id] = "No planner found for this agent"
                    continue
                agent_planner = decentralized_planners[agent_id]
                agent_low_level_actions, agent_responses = (
                    agent_planner.process_high_level_actions(
                        {agent_id: agent_action}, observations
                    )
                )
                low_level_actions.update(agent_low_level_actions)
                responses.update(agent_responses)
        else:
            low_level_actions, responses = llm_env.process_high_level_actions(
                high_level_skill_actions, observations
            )

        # Check if all targeted agents are done
        if all(responses.get(agent_id) for agent_id in assigned_agent_ids):
            skill_done = True

        if len(low_level_actions) == 0:
            assert skill_done, f"No low level actions returned. Response: {responses.values()}"
            break

        # Get the observations
        obs, reward, done, info = llm_env.env_interface.step(low_level_actions)
        observations = llm_env.env_interface.parse_observations(obs)

        if make_video:
            dvu._store_for_video(observations, high_level_skill_actions)

        # Increase steps
        skill_steps += 1

    if make_video and write_video and skill_steps > 1:
        dvu._make_video(postfix=vid_postfix, play=play_video)

    return responses, {"skill_steps": skill_steps}, dvu.frames
