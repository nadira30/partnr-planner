# skill runner command: 
HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.skill_runner hydra.run.dir="." +skill_runner_show_topdown=True habitat.dataset.data_path=visualization/data/episode_100_modified_2026-03-05_16-02-45.json.gz +skill_runner_episode_id="100"

# run skill runner with command file: 
HYDRA_FULL_ERROR=1 /home/nadira/miniconda3/envs/habitat/bin/python -m habitat_llm.examples.skill_runner hydra.run.dir="." +skill_runner_show_topdown=True habitat.dataset.data_path=visualization/data/episode_100_modified_2026-03-05_16-02-45.json.gz +skill_runner_episode_id="100" +skill_runner_commands_file=skill_runner_commands_example.txt

This will automatically run all commends in the txt file andd save the resulting video. 

# add objects to house 
 curl -X POST http://localhost:5002/api/episode/100/add-objects-batch   -F "file=@generated_add_objects_100.yaml"

# remap command skill before excuting 
the generated skill command assign objects id in order for each category. However, in the skill runner, the objects are renamed by assign an id for all the objects in aorder. 
To-Do: include this step directly in the skill command generator. 
python visualization/episode_info/remap_skill_runner_commands.py --commands-file skill_runner_commands.txt --dataset-path visualization/data/episode_100_modified_2026-03-30_13-51-13.json.gz --episode-id 100 --output-file skill_runner_commands_mapped.txt

# run decentralized planner
python -m habitat_llm.examples.skill_runner --config-name examples/skill_runner_decentralized_config.yaml hydra.run.dir="." +skill_runner_show_topdown=True habitat.dataset.data_path=visualization/data/episode_100_modified_2026-03-30_13-51-13.json.gz +skill_runner_episode_id="100" +skill_runner_commands_file=skill_runner_commands_mapped.txt

# run decentralized planner with offset and robot facing human 
python -m habitat_llm.examples.skill_runner --config-name examples/skill_runner_decentralized_config.yaml hydra.run.dir="." evaluation.save_video=False +skill_runner_show_topdown=True habitat.dataset.data_path=visualization/data/episode_100_modified_2026-03-30_13-51-13.json.gz +skill_runner_episode_id="100" +skill_runner_commands_file=skill_runner_commands_mapped.txt +skill_runner_robot_human_min_distance=0.6 

# running previous and save human loc and pose to file 
 python -m habitat_llm.examples.skill_runner_human_trace --config-name examples/skill_runner_decentralized_config.yaml hydra.run.dir="." +skill_runner_show_topdown=True habitat.dataset.data_path=visualization/data/episode_100_modified_2026-03-30_13-51-13.json.gz +skill_runner_episode_id="100" +skill_runner_commands_file=skill_runner_commands_mapped.txt +skill_runner_robot_human_min_distance=0.6 evaluation.save_video=False

 # GPU running command 
 CUDA_VISIBLE_DEVICES=0 /home/nadira/miniconda3/envs/habitat/bin/python -m habitat_llm.examples.skill_runner_human_trace --config-name examples/skill_runner_decentralized_config.yaml hydra.run.dir="." device=cuda habitat_baselines.torch_gpu_id=0 habitat.simulator.habitat_sim_v0.gpu_device_id=0 +skill_runner_show_topdown=True habitat.dataset.data_path=visualization/data/episode_100_modified_2026-03-30_13-51-13.json.gz +skill_runner_episode_id="100" +skill_runner_commands_file=skill_runner_quick_check.txt +skill_runner_robot_human_min_distance=0.6 evaluation.save_video=False 

 CUDA_VISIBLE_DEVICES=0 /home/nadira/miniconda3/envs/habitat/bin/python -m habitat_llm.examples.skill_runner_human_trace --config-name examples/skill_runner_decentralized_config.yaml hydra.run.dir="." device=cuda habitat_baselines.torch_gpu_id=0 habitat.simulator.habitat_sim_v0.gpu_device_id=0 habitat.dataset.data_path=visualization/data/episode_100_modified_2026-04-27_18-07-46.json.gz +skill_runner_episode_id="100" +skill_runner_commands_file=skill_runner_quick_check.txt +skill_runner_robot_human_min_distance=0.6 evaluation.save_video=False +skill_runner_show_videos=False

SKILL_RUNNER_HUMAN_TRACE_FILE=/home/nadira/partnr-planner/outputs/human_room_trace_quick_check_dual_clock_amy.txt CUDA_VISIBLE_DEVICES=0 /home/nadira/miniconda3/envs/habitat/bin/python -m habitat_llm.examples.skill_runner_human_trace --config-name examples/skill_runner_decentralized_config.yaml hydra.run.dir="." device=cuda habitat_baselines.torch_gpu_id=0 habitat.simulator.habitat_sim_v0.gpu_device_id=0 habitat.dataset.data_path=visualization/data/episode_100_modified_2026-04-27_18-07-46.json.gz +skill_runner_episode_id="100" +skill_runner_commands_file=skill_runner_commands/Amy_mapped.txt +skill_runner_robot_human_min_distance=0.6 evaluation.save_video=False +skill_runner_show_videos=False

python visualization/episode_info/remap_skill_runner_commands.py --commands-file skill_runner_commands/Malik_weekly_routine.txt --dataset-path visualization/data/episode_100_modified_2026-04-27_18-07-46.json.gz --episode-id 100 --output-file skill_runner_commands/Malik_mapped.txt
