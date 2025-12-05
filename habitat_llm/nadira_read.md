Progress:
Played around with simulator to see different configurations
Modify the LLM planner file to make the human perform actions and robot waits
Tried to make the agents perform actions simulataneously but this is not permitted by the skill runner currently -> Potential Solution: I could modify the data json file to make all the actions go to the human and make the robot navigate. Check the scripted_centralized planner and others
Changed most conf files for full observability

Plan:
- Check the solve_dag function in the lanner instead of assign_dag to make robot waits while agents perform actions.
- Create a dataset file(AgentSense data mapping might be a good starting point) -- started in LLM_smarthome repo

Time-based to action-based observation:

Robot navigates every 3 human actions (configurable via observation_interval_actions)
Counts human actions instead of trying to track simulation time
More reliable and predictable behavior
This should work much better! Run the simulation again and you should see the robot navigating multiple times throughout the episode.
