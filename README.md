## Code for Paper: "Learning to Play Soccer From Scratch: Sample-Efficient Emergent Coordination Through Curriculum-Learning and Competition"


Code for Paper: "Learning to Play Soccer from Scratch: Sample-Efficient Emergent
Coordination through Curriculum-Learning and Competition", accepted in the 2021
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2021).


To download this repository (and its submodules):

```shell
git clone --recurse-submodules https://github.com/Pavan-Samtani/RLSoccer.git
```

Pip can be used to install both packages:

```shell
pip install dm_soccer2gym
pip install soccer-spinningup
```

This project uses Deepmind's (walker-based) Soccer Environment, available in:

```
https://github.com/deepmind/dm_control/tree/master/dm_control/locomotion/soccer
```

Installation instructions for the dm_control package are available in the same
link.

Saved models for each stage can be found in: https://drive.google.com/drive/folders/1Ek0pcSswHkap48YHuFRBjRUbIKsnn-7w?usp=sharing

To load the soccer environment with our wrapper run:

```python
import dm_soccer2gym

# Returns 2v2 soccer environment, with a dense reward
env = dm_soccer2gym.make('2vs2', task_kwargs={"rew_type": "dense", "time_limit": 45.,
"disable_jump": True, "dist_thresh": 0.03,  'control_timestep': 0.05, 'observables': 'all'})

```

To load a policy, simply run:

```python
from spinup.utils.test_policy import load_policy_and_env

_, get_action = load_policy_and_env("path_to_models", model_identifier_num,
                                    two_p=two_player_policy)
```

Where:
- get_action is a function that receive an observation array and outputs a
two-dimensional action output.
- path_to_models is the path where the Tensorflow checkpoints are saved.
- model_identifier_num is the integer number next to XX
- two_player_policy is a boolean indicating if the policy corresponds to that
of a team of two robot agents. If true, get_action is a two-input function, that
receives the observations of both team agents, and outputs a tuple of two
two-dimensional action outputs.

If you use our work, please cite:

```
@inproceedings{samtani2021learning,
  title={Learning to Play Soccer From Scratch: Sample-Efficient Emergent Coordination
    Through Curriculum-Learning and Competition},
  author={Samtani, Pavan and Leiva, Francisco and Ruiz-del-Solar, Javier},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2021}
}
```

This project used:

- https://github.com/martinseilair/dm_control2gym/ as a skeleton for the
dm_soccer2gym wrapper, which wraps Deepmind's Control Suite Environment, as a
Gym Environment.
- OpenAI's SpinningUp (https://github.com/openai/spinningup/), as a skeleton for
the basic multi-agent extension of TD3 used in our work.
