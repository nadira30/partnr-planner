# skill runner command: 
HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.skill_runner hydra.run.dir="." +skill_runner_show_topdown=True habitat.dataset.data_path=visualization/data/episode_100_modified_2026-03-05_16-02-45.json.gz +skill_runner_episode_id="100"

# run skill runner with command file: 
HYDRA_FULL_ERROR=1 /home/nadira/miniconda3/envs/habitat/bin/python -m habitat_llm.examples.skill_runner hydra.run.dir="." +skill_runner_show_topdown=True habitat.dataset.data_path=visualization/data/episode_100_modified_2026-03-05_16-02-45.json.gz +skill_runner_episode_id="100" +skill_runner_commands_file=skill_runner_commands_example.txt

This will automatically run all commends in the txt file andd save the resulting video. 

# remap command skill before excuting 
the generated skill command assign objects id in order for each category. However, in the skill runner, the objects are renamed by assign an id for all the objects in aorder. 
To-Do: include this step directly in the skill command generator. 

# run decentralized planner
python -m habitat_llm.examples.skill_runner --config-name examples/skill_runner_decentralized_config.yaml hydra.run.dir="." +skill_runner_show_topdown=True habitat.dataset.data_path=visualization/data/episode_100_modified_2026-03-30_13-51-13.json.gz +skill_runner_episode_id="100" +skill_runner_commands_file=skill_runner_commands_mapped.txt