import json
import numpy as np
import cv2
import os
import magnum
import re
import shutil
import random
import magnum as mn
# project_size = 2.0
projection_matrix = np.array([[1, 0, 0, 0],
       [0, 1, 0, 0],
      [0, 0, -1.00002, -0.0200002],
       [0, 0, -1, 0]])
viewport = [512,512]   #because the new sampling range is [512,512]
def _3d_to_2d(matrix, point_3d):
        # get the scene render camera and sensor object
        W, H = viewport[0], viewport[1]

        # use the camera and projection matrices to transform the point onto the near plane
        project_mar = projection_matrix
        # print("mar:",np.append(point_3d,1),np.array(matrix).reshape(4, 4))s
        cam_mat = mn.Matrix4(matrix)
        # point_transform = cam_mat.transform_point(point_3d)
        point_transform = np.dot(matrix,np.append(point_3d,1))
        point_transs = mn.Vector3(point_transform[:3])
        point_mat = mn.Matrix4(project_mar)
        projected_point_3d = point_mat.transform_point(point_transs)
        # convert the 3D near plane point to integer pixel space
        point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
        point_2d = point_2d / 2.0
        point_2d += mn.Vector2(0.5)
        point_2d *= mn.Vector2(W,H)
        out_bound = 10
        point_2d = np.nan_to_num(point_2d, nan=W+out_bound, posinf=W+out_bound, neginf=-out_bound)
        return point_2d.astype(int).tolist()

def _2d_to_3d(matrix, point_2d, depth=10000.0):
    W, H = viewport[0], viewport[1]
    
    point_2d_normalized = (np.array(point_2d) / np.array([W, H])) * 2.0 - 1.0
    point_2d_normalized = np.array([point_2d_normalized[0], -point_2d_normalized[1], 1.0])  # Z为1.0
    
    point_2d_normalized = mn.Vector3(point_2d_normalized)
    inv_proj_matrix = mn.Matrix4(projection_matrix).inverted()
    point_3d_camera = inv_proj_matrix.transform_point(point_2d_normalized)
    
    inv_camera_matrix = mn.Matrix4(matrix).inverted()
    point_3d_world = inv_camera_matrix.transform_point(point_3d_camera * depth)
    
    return list(point_3d_world)
def process_directory(base_dir,skip_len):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            json_path = os.path.join(dir_path, 'sum_data.json')
            
            if os.path.exists(json_path):
                # try:
                # print("dir_name---------------------:",dir_name)
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                    result = []
                    data_trans = []
                    action = ["turn","nav_to_point", "pick","turn","nav_to_point", "place"] 
                    #now 'turn' also include the answer that the robot can not see the target object
                    agent_0_action = 0
                    nav_slide = []
                    i = 0
                    while(i < len(data['entities'])):
                        if data['entities'][i]['data']['agent_0_has_finished_oracle_nav'] == [1.0]:
                            nav_slide.append(i+1)
                            i+=1
                        i+=1
                    place_index = 0
                    for i in range(0, len(data['entities'])):
                        step_data = data['entities'][i]
                        # agent_0_trans_matrix = step_data['data']['agent_0_robot_trans_martix']
                        agent_0_nowloc = step_data['data']['agent_0_localization_sensor']
                        agent_0_obj = step_data['data']['agent_0_obj_bounding_box']
                        # agent_1_objpos = step_data['data']['agent_1_obj_pos']
                        agent_0_camera = step_data['data']['agent_0_camera_extrinsic']
                        if agent_0_action == 5:
                            agent_0_target = data['entities'][place_index]['data']['agent_0_target_bounding_box']
                        else:
                            agent_0_target = step_data['data']['agent_0_target_bounding_box']
                        agent_0_rec = step_data['data']['agent_0_rec_bounding_box']
                        if agent_0_action == 0:
                            x,y,w,h = agent_0_rec[0]
                            if w*h > 12000 and agent_0_nowloc[:3]!= data['entities'][i+1]['data']['agent_0_localization_sensor'][:3]:
                                agent_0_action+=1
                        elif agent_0_action == 3:
                            x,y,w,h = agent_0_target[0]
                            # print("target",agent_0_target[0])
                            if w*h > 12000 and agent_0_nowloc[:3]!= data['entities'][i+1]['data']['agent_0_localization_sensor'][:3]:
                                agent_0_action+=1
                                
                        elif (agent_0_action == 1 or agent_0_action == 4) and step_data['data']['agent_0_has_finished_oracle_nav'] == [1.0]:
                            if agent_0_action == 4:
                                place_index = i
                            agent_0_action += 1
                        elif agent_0_action == 2 and data['entities'][i+1]['data']['agent_0_localization_sensor'] != data['entities'][i]['data']['agent_0_localization_sensor']:
                            agent_0_action += 1
                        # annotation of the agent_0's action
                        # if (agent_1_action == 0 or agent_1_action == 2) and step_data['data']['agent_1_has_finished_oracle_nav'] == [1.0]:
                        #     agent_1_action += 1
                        # if agent_1_action == 1 and data['entities'][i]['data']['agent_1_localization_sensor'] != data['entities'][i-1]['data']['agent_1_localization_sensor']:
                        #     agent_1_action += 1

                        result = {
                            "step": i + 1,
                            "agent_0_now_worldloc":agent_0_nowloc,
                            "agent_0_obj": agent_0_obj,
                            "agent_0_rec": agent_0_rec,
                            "agent_0_target": agent_0_target,
                            "agent_0_martix": agent_0_camera,
                            # "agent_1_pre_worldloc":agent_1_pre_worldloc,
                            # "agent_0_pre_robotloc": agent_0_pre_robotloc.tolist(),
                            # "agent_0_trans_matrix": agent_0_trans_matrix,
                            # "agent_1_trans_matrix": agent_1_trans_matrix,
                            # "agent_1_nowloc": agent_1_nowloc,
                            # "agent_1_pre_robotloc": agent_1_pre_robotloc.tolist(),
                            # "agent_0_obj_ro": agent_0_obj_ro.tolist(),
                            # "agent_1_obj_ro": agent_1_obj_ro.tolist(),
                            "agent_0_action": action[agent_0_action],
                            # "agent_1_action": action[agent_1_action]
                        }
                        data_trans.append(result)
                    data_dir_path = os.path.join(dir_path, 'data')
                    if not os.path.exists(data_dir_path):
                        os.makedirs(data_dir_path)
                    output_json_path = os.path.join(data_dir_path, 'data_trans.json')
                    # output_json_path = os.path.join(dir_path, 'data_trans.json')
                    with open(output_json_path, 'w') as file:
                        json.dump(data_trans, file, indent=2)
                # except Exception as e:
                #     print(f"ERROR:{e}")
def check_if_in_range(t):
    if t>=0 and t<512:
        return True
def check_bounding_box(data):
    if len(data) != 1:
        return False
    x,y,w,h = data[0]
    if check_if_in_range(x) and check_if_in_range(y) and check_if_in_range(x+w) and check_if_in_range(y+h):
        return True
    return False
def limit_to_range(num):
    if num<0:
        return 0
    elif num>511:
        return 511
    else:
        return num
def datatrans_2_end_sat_waypoint_closer(process_dir:str,skip_len:int,pick_place_sample_num=3,sample_clip=800) -> list:
    find_episode = []
    skip_len_start = skip_len
    process_dir_path = process_dir
    for folder_name in os.listdir(process_dir_path):
        json_path = os.path.join(process_dir_path,folder_name,"sum_data.json")
        if os.path.exists(json_path):
            with open(json_path,'r',encoding='utf-8') as f:
                data = json.load(f)
                if 'entities' in data:
                    entity_count = len(data['entities'])
                    if 5< entity_count < sample_clip-5:
                        find_episode.append(folder_name)
    temp_q = 0
    sample_info = []
    for name in find_episode:
        try:
            with open(os.path.join(process_dir_path,name,"data/data_trans.json"), 'r') as file:
                data = json.load(file)
            data_final_0 = []
            action = ["turn","nav_to_point","pick","turn","nav_to_point","place"]
            action_point_index = []
            i = 1
            result_agent_0 = []
            flag = 0
            late_action = data[0]["agent_0_action"]
            while i < len(data):
                if (data[i]["agent_0_action"] != late_action):
                    action_point_index.append(i)
                    late_action = data[i]["agent_0_action"]
                i+=1
            # print("action_point_index:",action_point_index)
            assert len(action_point_index) == 5,"Wrong episode"
            turn1 = {
                "step":1,
                "action":{
                    "name":"search_for_object_rec",
                    "position":None
                },
                "image":f"frame_1"+"_agent_0_head_rgbFetchRobot_head_rgb.png",
            }
            data_final_0.append(turn1)
            nav_1_point = [action_point_index[0]]
            i = action_point_index[0]
            skip_len = skip_len_start +random.randint(-3,3)
            while i +skip_len < action_point_index[1]:
                now_step = i
                if i+skip_len+14 >= action_point_index[1]:
                    i = action_point_index[1]
                else:
                    i +=skip_len
                skip_len = skip_len_start +random.randint(-3,3)
                test_step = i
                for a in range(now_step,test_step):
                    test_point = _3d_to_2d(matrix=data[now_step]["agent_0_martix"],
                                                 point_3d=data[test_step]["agent_0_now_worldloc"][:3])
                    x,y = test_point
                    if not (0 <= x < 512 and 0 <= y < 512):
                        test_step -=1
                    else:
                        break
                nav_1_point.append(test_step)
                i = test_step
            if nav_1_point[-1] != action_point_index[1]:
                nav_1_point.append(action_point_index[1])
            for i in range(len(nav_1_point)):
                if i+1< len(nav_1_point):
                    x,y = _3d_to_2d(matrix=data[nav_1_point[i]]["agent_0_martix"],
                                                 point_3d=data[nav_1_point[i+1]]["agent_0_now_worldloc"][:3])
                    x = limit_to_range(x)
                    y = limit_to_range(y)
                    nav_temp = {
                        "step":data[nav_1_point[i]]["step"],
                        "action":{
                            "name":"nav_to_point",
                            "position":[x,y]
                        },
                        "image":f"frame_"+str(data[nav_1_point[i]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                    }
                    data_final_0.append(nav_temp)
            for i in range(action_point_index[1]-2,action_point_index[1]+1):  #TODO:check out the right data[i]["step"]
                pick_temp = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"pick",
                        "position":data[i]["agent_0_obj"]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                }
                data_final_0.append(pick_temp)
            turn2 = {
                "step":action_point_index[2],
                "action":{
                    "name":"search_for_goal_rec",
                    "position":None
                },
                "image":f"frame_"+str(action_point_index[2])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
            }
            data_final_0.append(turn2)
            nav_2_point = [action_point_index[3]]
            i = action_point_index[3]
            skip_len = skip_len_start +random.randint(-3,3)
            while i +skip_len < action_point_index[4]:
                now_step = i
                if i+skip_len+14 >= action_point_index[4]:
                    i = action_point_index[4]
                else:
                    i +=skip_len
                skip_len = skip_len_start +random.randint(-3,3)
                test_step = i
                for a in range(now_step,test_step):
                    test_point = _3d_to_2d(matrix=data[now_step]["agent_0_martix"],
                                                 point_3d=data[test_step]["agent_0_now_worldloc"][:3])
                    x,y = test_point
                    if not (0 <= x < 512 and 0 <= y < 512):
                        test_step -=1
                    else:
                        break
                nav_2_point.append(test_step)
                i = test_step
            if nav_2_point[-1] != action_point_index[4]:
                nav_2_point.append(action_point_index[4])
            for i in range(len(nav_2_point)):
                if i+1< len(nav_2_point):
                    x,y = _3d_to_2d(matrix=data[nav_2_point[i]]["agent_0_martix"],
                                                 point_3d=data[nav_2_point[i+1]]["agent_0_now_worldloc"][:3])
                    x = limit_to_range(x)
                    y = limit_to_range(y)
                    nav_temp = {
                        "step":data[nav_2_point[i]]["step"],
                        "action":{
                            "name":"nav_to_point",
                            "position": [x,y]
                            },
                        "image":f"frame_"+str(data[nav_2_point[i]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                    }
                    data_final_0.append(nav_temp)
            for i in range(action_point_index[4]-2,action_point_index[4]+1):
                place_temp = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"place",
                        "position":data[i]["agent_0_target"]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                }
                data_final_0.append(place_temp)
            temp_info = {
                    "episode_id":int(name.replace('episode_', '')),
                    "sample_frame":[],
                }
            for i in range(len(data_final_0)):
                match = re.search(r"frame_(\d+)_agent_(\d+)", data_final_0[i]["image"])
                if match:
                    frame_number = match.group(1)
                    agent_number = match.group(2)
                result_0 = [int(frame_number), int(agent_number)]
                temp_info["sample_frame"].append(result_0)   
            sample_info.append(temp_info)
            with open(os.path.join(process_dir_path,name,f"{name}.json"), 'w') as file:
                json.dump(data_final_0, file, indent=2)
        except:
            continue
    return sample_info
