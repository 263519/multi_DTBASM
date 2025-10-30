Instructions for editing, building, and running the robot swarm simulation.

---

### 1. Edit the Controller

The main robot controller logic file can be found at:

<your_workspace>/multi_DTBASM/src/sphero_simulation/sphero_stage/sphero_stage/robot_controller.py

---


### 2. Build and Run the Controller

To build only the 'sphero_stage' package and run your controller node:

    cd <your_workspace>/multi_DTBASM
    colcon build --packages-select sphero_stage
    source install/setup.bash
    ros2 run sphero_stage robot_controller

---

### 3. Run the World (Simulation)

Use this command to run the example world in Stage:

    ros2 launch sphero_stage start.launch.py

NOTE: If you encounter problems launching (e.g., after changing branches), try rebuilding the project with 'colcon build' first.

---



### 4. Configuration (Maps, Number of Robots)

Parameters like the map or number of robots can be modified in the file:

<your_workspace>/multi_DTBASM/src/sphero_simulation/sphero_stage/launch/launch_params.yaml