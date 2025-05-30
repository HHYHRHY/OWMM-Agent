import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
import pdb
from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from habitat_mas.agents.vlm_agent import VLMAgent,VLMAgentSingle
from collections import OrderedDict
import json
# class Context:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if not cls._instance:
#             cls._instance = super(Context, cls).__new__(cls, *args, **kwargs)
#             cls._instance.data = {}
#         return cls._instance

#     def update(self, data):
#         self.data = data

#     def get_data(self):
#         return self.data
# def get_context():
#     return Context()
# com = get_context()
# print("com_id",id(com))
import multiprocessing
def produce_data(queue,data):
    queue.put(data)

class HabitatMASEvaluator(Evaluator):
    """
    Evaluator for Habitat environments.
    """

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        temp = 0
        observations = envs.reset()
        observations = envs.post_step(observations)
        # queue = multiprocessing.Queue()
        # producer_process = multiprocessing.Process(target=produce_Data,args=(queue,))
        # producer_process.start()
        # # collect numerical observations and non-numerical observations
        # numerical_observations = {
        #     key: value
        #     for key, value in observations[0].items()
        #     if isinstance(value, np.ndarray)
        # }
        # non_numerical_observations = {
        #     key: value
        #     for key, value in observations.items()
        #     if not isinstance(value, np.ndarray)
        # }

        # batch = batch_obs(numerical_observations, device=device)
        # print("observations:",observations)
        name_to_remove = {'agent_0_obj_list_info','agent_1_obj_list_info'}
        obj_info_item_disk = []
        name_to_remove = {'agent_0_obj_list_info','agent_1_obj_list_info'}
        for obs in observations:
            for key in name_to_remove:
                if key in obs:
                    obj_info_item_disk.append(obs[key])
                    obs[key] = np.array([1])
        try:
            obj_info_item = obj_info_item_disk[0]
        except:
            print("skip")
        batch = batch_obs(observations, device=device) 
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore
        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.image_option) > 0:
            os.makedirs(config.habitat_baselines.image_dir, exist_ok=True)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            # for item in batch.items():
            #     print(item)
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}, config, # if k != "agent_0_fourth_rgb" and k != "agent_1_fourth_rgb"
                        0,
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
            if config.habitat_baselines.eval.generate_fourth_rgb:
                rgb_frames_fourth: List[List[np.ndarray]] = [
                    [
                        observations_to_image(
                            {k: v[env_idx] for k, v in batch.items()if
                                 k == "agent_0_fourth_rgb"}, {}, config, 0,
                        )
                    ]
                    for env_idx in range(config.habitat_baselines.num_environments)
                ]
        else:
            rgb_frames_fourth = None
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)


        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"
        envs_text_context = {}
        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()
        if config.habitat_baselines.eval.vlm_eval or config.habitat_baselines.eval.vlm_compare:
            vlm_agent = VLMAgentSingle(agent_num = 2,image_dir = './video_dir/image',
                                 json_dir = './video_dir/image_dir/episode_91/episode_91.json',
                                 url = "http://0.0.0.0:10077/robot-chat")

        cur_ep_id = -1
        dataset_info = config.habitat.dataset.data_path
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()
            # If all prev_actions are zero, meaning this is the start of an episode
            # Then collect the context of the episode
            if current_episodes_info[0].episode_id != cur_ep_id:
                cur_ep_id = current_episodes_info[0].episode_id
                print("===============================================================================")
                print("=================================Episode ID====================================")
                print("Current Episode ID: ", cur_ep_id)
                print("=================================Episode ID====================================")
                print("===============================================================================")
                envs_text_context = envs.call(["get_task_text_context"] * envs.num_envs)
                if 'pddl_text_goal' in batch:
                    envs_pddl_text_goal_np = batch['pddl_text_goal'].cpu().numpy()
                    for i in range(envs.num_envs):
                        pddl_text_goal_np = envs_pddl_text_goal_np[i, ...]
                        envs_text_context[i]['pddl_text_goal'] = ''.join(str(pddl_text_goal_np, encoding='UTF-8'))
                
                for i in range(envs.num_envs):
                    # also add the debug/ logging info to the text context for convenience
                    envs_text_context[i]['episode_id'] = current_episodes_info[i].episode_id

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            # print("start:---------------------------------------------------------------------------")
            # print("---------------------------------------------------------------------------")
            filter_action = {}
            if config.habitat_baselines.eval.vlm_eval:
                eval_jump = True
            else:
                eval_jump = False
            
            if config.habitat_baselines.eval.vlm_eval:
                
                with inference_mode():
                    check_info = agent.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                        envs_text_context=envs_text_context,
                        check_info = True,
                        **space_lengths,
                    )
                print("check_info:",check_info)
                agent_query = [1 if item else 0 for item in check_info.should_inserts[0]]
                print("agent_query:",agent_query)
                if agent_query[0] == 1:
                    agent_0_image = batch["agent_0_head_rgb"].cpu()
                    agent_1_image = batch["agent_1_head_rgb"].cpu()
                    agent_0_depth_info = batch["agent_0_depth_inf"].cpu()
                    # print("depthinfoshape:",agent_0_depth_info.shape)
                    # agent_0_trans = batch["agent_0_robot_trans_martix"].cpu()
                    # agent_1_trans = batch["agent_1_robot_trans_martix"].cpu()
                    # image = [agent_0_image,agent_1_image]
                    agent_0_depth_rot = batch["agent_0_depth_rot"].cpu()
                    agent_0_depth_trans = batch["agent_0_depth_trans"].cpu()
                    image = [agent_0_image]
                    agent_trans = []
                    # print("image_shape:",image[0].shape,flush = True)
                    filter_action = vlm_agent.answer_vlm(agent_trans_list = agent_trans,
                                                         agent_query = agent_query,image = image,
                                                         episode_id = int(current_episodes_info[0].episode_id),depth_info=agent_0_depth_info,depth_rot = agent_0_depth_rot,depth_trans = agent_0_depth_trans)
                    print("__________________")
                    print("filter_action",filter_action)
                    for agent_id,info in filter_action.items():
                        if info["name"] == "pick":
                            position = info["position"]
                            mindistance = 100000
                            min_key = None
                            for key,value in obj_info_item.items():
                                if not str(key).startswith("robot_"):
                                    distance = np.linalg.norm(position - value)
                                    if distance < mindistance:
                                        min_key = key
                                        mindistance = distance
                                        # print()
                            info["position"] = str(min_key)
                    stored_path ='./data_temp_10.json'
                    if os.path.exists(stored_path) and os.path.getsize(stored_path) == 0:
                        with open(stored_path,'w') as f:
                            json.dump(filter_action,f)
                    else:
                        with open(stored_path,'r') as f:
                            data = json.load(f)
                        for key in filter_action:
                            if key in data:
                                data[key] = filter_action[key]
                        with open(stored_path,'w') as f:
                            json.dump(data,f)
            # print("current_episodes_info",current_episodes_info)
            # agent_0_image = batch["agent_0_head_rgb"].cpu()
            # agent_0_depth_info = batch["agent_0_depth_inf"].cpu()
            # agent_0_depth_rot = batch["agent_0_depth_rot"].cpu()
            # agent_0_depth_trans = batch["agent_0_depth_trans"].cpu()
            # print("agent_0_depth_info.shape:",agent_0_depth_info.shape)
            # print("agent_0_depth_rot.shape:",agent_0_depth_rot)
            # print("agent_0_depth_trans.shape:",agent_0_depth_trans)

            with inference_mode():
                ep_info = [int(cur_ep_id),dataset_info]
                # print("ep_info:",ep_info)
                # agent_0_loc = batch["agent_0_localization_sensor"].cpu()
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    ep_info,
                    # deterministic=False,
                    envs_text_context=envs_text_context,
                    output = filter_action,
                    **space_lengths,
                )
                if config.habitat_baselines.eval.vlm_compare:
                    agent_0_image = batch["agent_0_head_rgb"].cpu()
                    agent_1_image = batch["agent_1_head_rgb"].cpu()
                    agent_0_depth_info = batch["agent_0_depth_inf"].cpu()
                    # agent_0_trans = batch["agent_0_robot_trans_martix"].cpu()
                    # agent_1_trans = batch["agent_1_robot_trans_martix"].cpu()
                    # image = [agent_0_image,agent_1_image]
                    print("image_shape",agent_0_image.shape,flush=True)
                    image = [agent_0_image]

                    agent_trans = []
                    filter_action = vlm_agent.send_and_receive(image_list = image,episode_id=26)
                    print("__________________")
                    print(f"{temp}_action",filter_action,flush = True)
                    temp+=1
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # print(f"----------------action_data\n{action_data}\n-----------------------------")
            if is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]
            # pdb.set_trace()
            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()if
                         k != "agent_0_fourth_rgb" and k != "agent_1_fourth_rgb"}, disp_info,
                        config, len(rgb_frames[0]),
                        episode_id=current_episodes_info[i].episode_id,
                    )
                    if config.habitat_baselines.eval.generate_fourth_rgb:
                        frame_fourth = observations_to_image(
                            {k: v[i] for k, v in batch.items() if
                             k == "agent_0_fourth_rgb"}, infos[i],
                            config, len(rgb_frames_fourth[0]),
                            episode_id=current_episodes_info[i].episode_id,
                        )
                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()if
                             k != "agent_0_fourth_rgb" and k != "agent_1_fourth_rgb" and ("camera_info" not in k)},
                            disp_info, config,
                            frame_id=len(rgb_frames[0]),
                            episode_id=current_episodes_info[i].episode_id,
                        )
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            final_frame_fourth = observations_to_image(
                                {k: v[i] for k, v in batch.items() if
                                 k == "agent_0_fourth_rgb"}, infos[i],
                                config, len(rgb_frames_fourth[0]),
                                episode_id=current_episodes_info[i].episode_id,
                            )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            final_frame_fourth = overlay_frame(final_frame_fourth, infos[i])
                            rgb_frames_fourth[i].append(final_frame_fourth)
                            rgb_frames_fourth[i].append(frame_fourth)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            frame_fourth = overlay_frame(frame_fourth, infos[i])
                            rgb_frames_fourth[i].append(frame_fourth)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    # clear the prev_actions and recurrent_hidden_states
                    prev_actions[i] = 0
                    test_recurrent_hidden_states[i] = 0

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            generate_video(
                                video_option=config.habitat_baselines.eval.video_option,
                                video_dir=config.habitat_baselines.video_dir,
                                images=rgb_frames_fourth[i][:-1],
                                episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}_fourth",
                                checkpoint_idx=checkpoint_index,
                                metrics=extract_scalars_from_info(disp_info),
                                fps=config.habitat_baselines.video_fps,
                                tb_writer=writer,
                                keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                            )
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            rgb_frames_fourth[i] = rgb_frames_fourth[i][-1:]
                        rgb_frames[i] = rgb_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)
