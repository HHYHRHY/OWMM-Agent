# Install Habitat
1. **Preparing conda env**

   Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
   ```bash
   # We require python>=3.9 and cmake>=3.14
   conda create -n habitat python=3.9 cmake=3.14.0
   conda activate habitat
   ```

1. **conda install habitat-sim**
    To install habitat-sim with bullet physics
      
      ```bash
      # We fix the habitat-sim version to 0.3.1. 
      # You might have compatibility issue for other versions since habitat is under active development
      conda install habitat-sim=0.3.1 withbullet -c conda-forge -c aihabitat
      ```

      Note, for newer features added after the most recent release, you may need to install `aihabitat-nightly`. See Habitat-Sim's [installation instructions](https://github.com/facebookresearch/habitat-sim#installation) for more details.

1. **pip install habitat-lab stable version**.

      ```bash
      cd habitat-mas
      pip install -e habitat-lab  # install habitat_lab
      # NOTE: You need to install our modified habitat-lab package rather than the original repo.
      # git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
      # cd habitat-lab
      # pip install -e habitat-lab  # install habitat_lab
      ```
1. **Install habitat-baselines**.

    The command above will install only core of Habitat-Lab. To include habitat_baselines along with all additional requirements, use the command below after installing habitat-lab:

      ```bash
      pip install -e habitat-baselines  # install habitat_baselines
      ```

## Testing

1. Let's download some 3D assets using Habitat-Sim's python data download utility:
   - Download (testing) 3D scenes:
      ```bash
      python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
      ```
      Note that these testing scenes do not provide semantic annotations.

   - Download point-goal navigation episodes for the test scenes:
      ```bash
      python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
      ```

1. **Non-interactive testing**: Test the Pick task: Run the example pick task script
    <!--- Please, update `examples/example.py` if you update example. -->
    ```bash
    python examples/example.py
    ```

    which uses [`habitat-lab/habitat/config/benchmark/rearrange/skills/pick.yaml`](habitat-lab/habitat/config/benchmark/rearrange/skills/pick.yaml) for configuration of task and agent. The script roughly does this:

    ```python
    import gym
    import habitat.gym

    # Load embodied AI task (RearrangePick) and a pre-specified virtual robot
    env = gym.make("HabitatRenderPick-v0")
    observations = env.reset()

    terminal = False

    # Step through environment with random actions
    while not terminal:
        observations, reward, terminal, info = env.step(env.action_space.sample())
    ```

    To modify some of the configurations of the environment, you can also use the `habitat.gym.make_gym_from_config` method that allows you to create a habitat environment using a configuration.

    ```python
    config = habitat.get_config(
      "benchmark/rearrange/skills/pick.yaml",
      overrides=["habitat.environment.max_episode_steps=20"]
    )
    env = habitat.gym.make_gym_from_config(config)
    ```

    If you want to know more about what the different configuration keys overrides do, you can use [this reference](habitat-lab/habitat/config/CONFIG_KEYS.md).

    See [`examples/register_new_sensors_and_measures.py`](examples/register_new_sensors_and_measures.py) for an example of how to extend habitat-lab from _outside_ the source code.



1. **Interactive testing**: Using you keyboard and mouse to control a Fetch robot in a ReplicaCAD environment:
    ```bash
    # Pygame for interactive visualization, pybullet for inverse kinematics
    pip install pygame==2.0.1 pybullet==3.0.4

    # Interactive play script
    python examples/interactive_play.py --never-end
    ```

   Use I/J/K/L keys to move the robot base forward/left/backward/right and W/A/S/D to move the arm end-effector forward/left/backward/right and E/Q to move the arm up/down. The arm can be difficult to control via end-effector control. More details in documentation. Try to move the base and the arm to touch the red bowl on the table. Have fun!

   Note: Interactive testing currently fails on Ubuntu 20.04 with an error: `X Error of failed request:  BadAccess (attempt to access private resource denied)`. We are working on fixing this, and will update instructions once we have a fix. The script works without errors on MacOS.