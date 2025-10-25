# ROS 2 Workspace Commands - sphero_stage

This file contains frequently used commands for working with the sphero_stage package in ROS 2.  
Short descriptions are provided. Add new commands below as needed.

---

## Configuration

```bash
#Change number of robots here
code /multi_DTBASM/src/sphero_simulation/sphero_stage/launch/launch_params.yaml
```

---

## Launching

```bash
#Start the sphero_stage simulation
ros2 launch sphero_stage start.launch.py
```

---

## Building & Running

```bash
Stage_build
Stage
ros2 run sphero_stage robot_controller
```

```bash

#cd ~/multi_DTBASM
#colcon build --packages-select sphero_stage
#. install/setup.bash
```
