import json, jsonlines
import glob, pdb, re
import csv,gzip
import os, shutil,gzip,pdb
import random
import process_qa
from process_qa import QA_process
import copy
import argparse
def process_array(arr):
    processed_arr = [str(int(x * 100)) for x in arr]
    return '[' + ','.join(processed_arr) + ']'
def recursive_delete_folder(folder_path):
    if not os.path.exists(folder_path):
        return
    if not os.path.isdir(folder_path):
        return
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error when delete {file_path}: {e}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
            except Exception as e:
                print(f"Error when delete {dir_path}: {e}")
    try:
        os.rmdir(folder_path)
    except Exception as e:
        print(f"Error when delete {folder_path}: {e}")
def get_green_point_list(data,episode,step):
    return data.get(episode, {}).get(step, None)
def count_points_in_bounding_box(points, bounding_box):
    x_min = bounding_box[0]
    y_min = bounding_box[1]
    x_max = x_min + bounding_box[2]
    y_max = y_min + bounding_box[3]
    count = 0
    for point in points:
        x, y, _ = point
        if (x_min <= x <= x_max and y_min <= y <= y_max):
            count += 1
    return count
def check_green_point(bbox_info,action,ispick,green_point_list):
    if action == "place":
        goal_bbox = bbox_info["goal_bbox"][0]
        if count_points_in_bounding_box(green_point_list,goal_bbox)<13:
            raise ValueError("Green Point CHECK ERROR--place")
    if action == "pick":
        obj_bbox = bbox_info["obj_bbox"][0]
        if count_points_in_bounding_box(green_point_list,obj_bbox)<1:
            raise ValueError("Green Point CHECK ERROR--pick")
    if action == "nav_to_point" and ispick:
        goal_bbox = bbox_info["goal_bbox"][0]
        if count_points_in_bounding_box(green_point_list,goal_bbox)>5:
            return False
    elif action == "nav_to_point":
        obj_bbox = bbox_info["obj_bbox"][0]
        if count_points_in_bounding_box(green_point_list,obj_bbox)>=1:
            return False
    return True
def process_image_name(image_name_in_json,desired_name):
    name_wo_frame = image_name_in_json.removeprefix("frame_")
    name_wo_frame_match = re.sub(r'(agent_\d+)(_head)',r'\1',name_wo_frame)
    if desired_name =="arm_workspace_rgb":
        name_wo_frame_match = name_wo_frame_match.replace("_head_rgb","_arm_workspace_rgb")
    elif desired_name == "head_rgb":
        pass
    return name_wo_frame_match
def parse_args():
    parser = argparse.ArgumentParser(description='Scene graph generation parameters')
    parser.add_argument('--getting_anno_only', action='store_true',
                        help='Select if get the annotation only')
    parser.add_argument('--store_scene_graph_only', action='store_true',
                        help='Select if you want to get the scene_graph_annotation_only')
    parser.add_argument('--target_goal_same', action='store_true',
                        help='Target goal same flag')
    parser.add_argument('--notseen_at_search', action='store_true',
                        help='Ensure scene graph not seen flag')
    parser.add_argument('--meta_json_name', type=str, default='robotdemo_meta.json',
                        help='Metadata JSON filename')
    parser.add_argument('--scene_graph_num', type=int, default=8,
                        help='The scene graph frames number')
    parser.add_argument('--dataset_name', type=str, default='TEST_GOOGLE_30scene',
                        help='Dataset name')
    parser.add_argument('--gz_dir_name', type=str, default='hssd_scene_google_test',
                        help='scene config directory name')
    parser.add_argument('--desired_robot_image_name', type=str, default='head_rgb',
                        choices=['arm_workspace_rgb', 'head_rgb'],
                        help='Type of robot image to use')
    
    args = parser.parse_args()
    args.output_anno_name = f'robotdata_{args.desired_robot_image_name}.jsonl'
    args.output_dir_name = f'sat_{args.dataset_name}_{args.desired_robot_image_name}'
    args.anno_name = f'robotdata_{args.desired_robot_image_name}.jsonl'
    args.scene_annotation_name = "scene_annotation_gpt"
    return args
def main(args):
    getting_anno_only = args.getting_anno_only       #select if get the annotation only
    store_scene_graph_only = args.store_scene_graph_only     #select if you want to get the scene_graph_annotation_only
    target_goal_same = args.target_goal_same
    notseen_at_search = args.notseen_at_search
    meta_json_name = args.meta_json_name
    dataset_name = args.dataset_name    #origin dataset name, generate by dataset_generation
    gz_dir_name = args.gz_dir_name
    desired_robot_image_name = args.desired_robot_image_name  #use arm_workspace_rgb or head_rgb
    output_dir_name = args.output_dir_name
    output_anno_name = args.output_anno_name
    output_anno_path = os.path.join(output_dir_name, output_anno_name)
    abs_output_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),output_dir_name)
    abs_output_anno_path = os.path.join(abs_output_dir_path,output_anno_path)
    scene_annotation_name = args.scene_annotation_name
    scene_graph_num = args.scene_graph_num
    meta_info = {
        "robotdata_demo": 
        {
            "root": abs_output_dir_path,
            "annotation": abs_output_anno_path,
            "data_augment": False,
            "repeat_time": 1,
            "length": 0
            }
    }
    num_step = 0
    num_id = 0
    combined_data = []
    target_goal_same = False
    divide_dataset_name = [os.path.join(dataset_name, name) 
                           for name in os.listdir(dataset_name) 
                           if os.path.isdir(os.path.join(dataset_name, name))] #in our dataset sampling pipeline, we put origin datasets generated from different GPUS to one folder
    directorys_name = []
    for divide_dataset in divide_dataset_name:
        divide_dataset_directorys_name = [os.path.join(divide_dataset, name) 
        for name in os.listdir(divide_dataset) if os.path.isdir(os.path.join(divide_dataset, name)) and name[0].isdigit()]
        directorys_name.extend(divide_dataset_directorys_name)
    print("directorys_name:",directorys_name)
    for directory_name in directorys_name:
        gz_files_path = os.path.join(gz_dir_name, os.path.basename(os.path.normpath(directory_name)))
        os.makedirs(output_dir_name, exist_ok=True)
        for process_folder in os.listdir(directory_name):
            # print(f"process folder:{process_folder}-dir:{directory_name}",flush = True)
            process_path = os.path.join(directory_name, process_folder)
            if os.path.isdir(process_path) and process_folder.startswith('data_'):
                with open(os.path.join(gz_files_path,f"{process_folder}.json"), 'r') as file:
                    gz_data = json.load(file)
                # print(f"{num_id}_gz_Data:{gz_data}")
                episodes_info = []
                for episode in gz_data.get('episodes', []):
                    episode_info = {
                        'episode_id': episode.get('episode_id'),
                        'target_receptacles': episode.get('target_receptacles'),
                        'goal_receptacles': episode.get('goal_receptacles'),
                        'target_object': episode.get('name_to_receptacle',{})
                    }
                    episodes_info.append(episode_info)
                green_point_sample_path = os.path.join(process_path, 'metadata_greenpoint.json')
                try:
                    with open(green_point_sample_path, 'r') as gp_sample_file:
                        gp_sample_data = json.load(gp_sample_file)
                    gp_sample_optimized_data = {}  #
                except:
                    print(f"not find green_point_json")
                    continue
                for item in gp_sample_data:
                    # print("item:",item)
                    episode = item['episode_id']
                    step = item['step']
                    if episode not in gp_sample_optimized_data:
                        gp_sample_optimized_data[episode] = {}
                    gp_sample_optimized_data[episode][step] = item['green_points']
                for episode_folder in os.listdir(process_path):
                    try:
                        episode_path = os.path.join(process_path, episode_folder)
                        if os.path.isdir(episode_path) and episode_folder.startswith('episode_'):
                            json_path = os.path.join(episode_path, f'{episode_folder}.json')
                            information_path = os.path.join(episode_path, 'sum_data.json')
                            if os.path.exists(json_path) and os.path.exists(information_path):
                                with open(json_path, 'r') as file:
                                    data = json.load(file)
                                with open(information_path, 'r') as file:
                                    information_data = json.load(file)
                                image_exist_flag = True
                                for item in data:
                                    # image_1 = item.get("image_1")
                                    image_name = process_image_name(item.get("image"),desired_robot_image_name)
                                    if image_name:
                                        image_exists = os.path.exists(os.path.join(episode_path, image_name))
                                    else:
                                        image_exists = False
                                    if not image_exists:
                                        image_exist_flag = False
                                        break
                                if image_exist_flag:
                                    match = re.search(r'episode_(\d+)', episode_folder)
                                    if match:
                                        now_episode = int(match.group(1))
                                    matched_episode = None
                                    for episode in episodes_info:
                                        if int(episode['episode_id']) == now_episode:
                                            matched_episode = episode
                                            break
                                    goal_item = matched_episode['goal_receptacles'][0][0].split('_')[0]
                                    target_item = matched_episode['target_receptacles'][0][0].split('_')[0]
                                    if target_item == goal_item:
                                        if not target_goal_same:
                                            raise ValueError("target and goal receptacles are same")
                                    obj_item = ""
                                    for key in matched_episode['target_object'].keys():
                                        key_replace_front = re.sub(r'^\d+_', '', key)
                                        key_replace_back = re.sub(r'[:_]\d+$', '', key_replace_front)
                                        obj_item = key_replace_back.replace('_', ' ')
                                    scene_name_path = os.path.join(scene_annotation_name,f"{os.path.basename(os.path.normpath(directory_name))}.json")
                                    goal_name = ""
                                    target_name = ""
                                    with open(scene_name_path, 'r') as f:
                                        scene_name_data = json.load(f)
                                    result_find = []
                                    for sc_name in scene_name_data:
                                        if goal_item == sc_name["template_name"]:
                                            goal_name = sc_name["name"]
                                            break
                                    for sc_name in scene_name_data:
                                        if target_item == sc_name["template_name"]:
                                            target_name = sc_name["name"]
                                            break
                                    if len(target_name) ==0 or len(obj_item) == 0 or len(goal_name) == 0:
                                        raise ValueError("Invalid object match")
                                    scene_graph_info = []
                                    for sup in data:
                                        agent_0_action = sup["action"]["name"]
                                        agent_0_pos = sup["action"]["position"]
                                        if agent_0_action == "pick" or agent_0_action == "place":
                                            x, y, w, h = agent_0_pos[0]
                                            if x < 0 or y < 0 or w < 0 or h < 0:
                                                raise ValueError("Invalid position")
                                    scene_image_list = []
                                    target_rec_scene_image = ""
                                    goal_rec_scene_image = ""
                                    for i,sup in enumerate(data):
                                        agent_0_action = sup["action"]["name"]
                                        if agent_0_action == "search_for_object_rec" or agent_0_action =="search_for_goal_rec":
                                            image_name = process_image_name(data[i+1]["image"],desired_robot_image_name)
                                            rec_append_name_path = ""
                                            if agent_0_action == "search_for_object_rec":
                                                rec_append_name_path = f"image/{num_id}/target_rec.png"
                                                target_rec_scene_image = rec_append_name_path
                                            else:
                                                rec_append_name_path = f"image/{num_id}/goal_rec.png"
                                                goal_rec_scene_image = rec_append_name_path
                                            match = re.search(r'(\d+)_agent_0', image_name)
                                            if match:
                                                frame_number = match.group(1)
                                            scene_graph_loc = information_data["entities"][int(frame_number)-1]['data']['agent_0_localization_sensor']
                                            scene_graph_info.append({
                                                "episode_id":str(now_episode),
                                                "obs_files":[image_name],
                                                "localization_sensor":scene_graph_loc,
                                                "annotation": agent_0_action
                                            })
                                            scene_image_list.append(rec_append_name_path)
                                    for i in range(2,scene_graph_num):
                                        scene_image_list.append(f"image/{num_id}/random_scene_graph_{i}.png")
                                    flag_has_picked = False
                                    process_qa = QA_process(target_name,goal_name,obj_item,scene_graph_num,arm_or_head=desired_robot_image_name)
                                    temp_data = []
                                    for i,sup in enumerate(data):
                                        item = {}
                                        item['id'] = f"{directory_name}_{process_folder}_{episode_folder}_{sup['step']}"
                                        agent_0_action = sup["action"]["name"]
                                        image_list = []
                                        random.shuffle(scene_image_list)
                                        rec_index = [scene_image_list.index(target_rec_scene_image),scene_image_list.index(goal_rec_scene_image)]
                                        image_list += scene_image_list
                                        new_image_name = process_image_name(sup['image'],desired_robot_image_name)
                                        image_list.append(f"image/{num_id}/{new_image_name}")
                                        match = re.search(r'(\d+)_agent_0', new_image_name)
                                        if match:
                                            frame_number = match.group(1)
                                        else:
                                            raise ValueError("Invalid image name")
                                        item['image'] = image_list
                                        item['height'] = [512, 512]
                                        item['weight'] = [512, 512]
                                        agent_0_pos = sup["action"]["position"]
                                        nav_point_bbox = []
                                        target_bbox = []
                                        goal_bbox = []
                                        if agent_0_action == "nav_to_point":
                                            x, y = agent_0_pos
                                            agent_0_pos = [[int((x-10) * 1000 / 512) if x-10>=0 else 0, int((y-10) * 1000 / 512) if y-10 >= 0 else 0,
                                                            int(20 * 1000 / 512) if x + 10 < 512 else int((512 - (x-10)) * 1000 / 512),
                                                            int(20 * 1000 / 512) if y + 10 < 512 else int((512 - (y-10)) * 1000 / 512)]]
                                            nav_point_bbox = [[(x-10) if x-10>=0 else 0, (y-10) if y-10 >= 0 else 0,20 if x + 10 < 512 else 512 - (x-10),
                                                                20 if y + 10 < 512 else 512 - (y-10)]]
                                            target_bbox = information_data["entities"][int(frame_number)-1]['data']['agent_0_rec_bounding_box']
                                            goal_bbox = information_data["entities"][int(frame_number)-1]['data']['agent_0_target_bounding_box']
                                        if agent_0_action == "pick" or agent_0_action == "place":
                                            if agent_0_action == "pick":
                                                flag_has_picked = True
                                                target_bbox = information_data["entities"][int(frame_number)-1]['data']['agent_0_rec_bounding_box']
                                                goal_bbox = [[-1,-1,-1,-1]]
                                            if agent_0_action == "place":
                                                target_bbox = [[-1,-1,-1,-1]]
                                                goal_bbox = agent_0_pos
                                            x, y, w, h = agent_0_pos[0]
                                            agent_0_pos = [[int(x * 1000 / 512), int(y * 1000 / 512), int(w * 1000 / 512),
                                                            int(h * 1000 / 512)]]
                                        if notseen_at_search:
                                            if agent_0_action == "search_for_object_rec":
                                                x, y, w, h = information_data["entities"][int(frame_number)-1]['data']['agent_0_rec_bounding_box'][0]
                                            if agent_0_action == "search_for_goal_rec":
                                                x, y, w, h = information_data["entities"][int(frame_number)-1]['data']['agent_0_target_bounding_box'][0]
                                            if w*h > 400:
                                                raise ValueError("Rec can seen at search action")
                                        obj_bbox = information_data["entities"][int(frame_number)-1]['data']['agent_0_obj_bounding_box']
                                        bbox_info = {
                                            "target_bbox":target_bbox,
                                            "goal_bbox":goal_bbox,
                                            "obj_bbox":obj_bbox,
                                            "nav_point_bbox":nav_point_bbox,
                                            "agent_0_pos":agent_0_pos
                                        }
                                        green_point_list = get_green_point_list(gp_sample_optimized_data,str(now_episode),int(frame_number))
                                        if not check_green_point(bbox_info,agent_0_action,flag_has_picked,green_point_list):
                                            if flag_has_picked:
                                                agent_0_action = "place"
                                                x, y, w, h = goal_bbox[0]
                                            else:
                                                agent_0_action = "pick"
                                                x, y, w, h = obj_bbox[0]
                                            agent_0_pos_new = [[int(x * 1000 / 512), int(y * 1000 / 512), int(w * 1000 / 512),
                                                            int(h * 1000 / 512)]]
                                            bbox_info["agent_0_pos"] = agent_0_pos_new
                                        conversations = process_qa.get_conversations(
                                            action_name=agent_0_action,bbox_dict=bbox_info,rec_index=rec_index,flag_if_picked=flag_has_picked)
                                        item["conversations"] = conversations
                                        for file_name in os.listdir(episode_path):
                                            if process_image_name(sup['image'],desired_robot_image_name) in file_name and not getting_anno_only:
                                                source_file = os.path.join(episode_path, file_name)
                                                destination_folder_path = os.path.join(output_dir_name,'image',str(num_id))
                                                os.makedirs(destination_folder_path, exist_ok=True)
                                                destination_file = os.path.join(destination_folder_path, file_name)
                                                shutil.copy(source_file, destination_file)
                                        if store_scene_graph_only:
                                            if agent_0_action in ["search_for_goal_rec","search_for_object_rec"]:
                                                temp_data.append(item)
                                        else:
                                            temp_data.append(item)
                                    combined_data.extend(temp_data)
                                    if not getting_anno_only:
                                        store_gz_episode = []
                                        store_gz = copy.deepcopy(gz_data)
                                        for episode in gz_data.get('episodes', []):
                                            if f"episode_{episode.get('episode_id')}" == episode_folder:
                                                store_gz_episode.append(episode)
                                                break
                                        store_gz["episodes"] = store_gz_episode
                                        scene_graph_sample_info_gz_path = os.path.join(output_dir_name,'image',str(num_id),'scene_graph.gz')
                                        with gzip.open(scene_graph_sample_info_gz_path,"wt") as f:
                                            f.write(json.dumps(store_gz))  #store scene_graph_sampling_gz
                                        scene_graph_info_path = os.path.join(output_dir_name,'image',str(num_id),'scene_graph_info.json')
                                        with open(scene_graph_info_path, "w") as f:
                                            json.dump(scene_graph_info, f,indent=4)  #store_scene_graph_info
                                    num_id += 1

                    except Exception as e:
                        if not getting_anno_only:
                            destination_folder_path = os.path.join(output_dir_name,'image',str(num_id))
                            recursive_delete_folder(destination_folder_path)
                        print(f"ERROR:{e}")
                        continue

    meta_info["robotdata_demo"]["length"] = len(combined_data)
    with jsonlines.open(output_anno_path, mode='w') as writer:
        writer.write_all(combined_data)
    with open(os.path.join(output_dir_name, meta_json_name), 'w') as file:
        json.dump(meta_info, file, indent=4)
        return 
if __name__ == "__main__":
    args = parse_args()
    main(args)





