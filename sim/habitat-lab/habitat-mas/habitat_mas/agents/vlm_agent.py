import matplotlib.pyplot as plt
import os
import numpy as np
import json
import re
import base64
import requests
import time
from PIL import Image
from io import BytesIO
class VLMAgent():
    def __init__(self,agent_num,image_dir,url,json_dir=None) -> None:
        self.image_dir = image_dir
        self.step_num = 0
        self.json_dir = json_dir
        self.url = url
    def vlm_inference(self,image_dir):
        output = {}
        return output
    def test_inference(self,json_dir):
        with open(json_dir,'r') as file:
            data = json.load(file)
        step = self.step_num
        return data[step]
    def pos_trans(self,trans,pos):
        poss = np.append(pos,1)
        coord_pos = np.dot(trans, poss)
        return np.array(coord_pos)[:3]
    def send_and_receive(self,image_list,episode_id):
        images = []
        headers = {
            "Content-Type": "application/json"
        }
        for image in image_list:
            image = image.squeeze(0).numpy().astype(np.uint8)
            image_PIL = Image.fromarray(image)
            buffered = BytesIO()
            image_PIL.save(buffered,format = 'PNG')
            images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        data = {
            "id": f"{episode_id}_{self.step_num}",
            "width": [256, 256],
            "height": [256, 256],
            "image": images,
            "prompt": (
                "You are an AI visual assistant that can manage a single robot. You receive the robot's task, one image representing the robot's current view and what the robot has completed so far. You need to output the robot's next action. Actions the robot can perform are \"nav_to_point\", \"turn\", \"pick\" and \"place\". If you cannot determine the next action based on the robot's current view, you can command the robot to \"turn\". Your output format should be either \"turn\" or \"action_name<box>[[x1, y1, x2, y2]]</box>\".Robot's Task: The robot need to navigate to the dresser where the box is located, pick it up,navigate to the kitchen and place the box.\n    Robot's current view: <image>\n    ."
            )
        }
        self.step_num+=1
        while True:
            response = requests.post(self.url, headers=headers, json=data)
            if response.status_code >= 200 and response.status_code < 300:
                return_json = response.json()
                try:
                    response_json = response.json()
                    if self.check_vlm_ans(response_json):
                        return response_json
                except:
                    print("invalid json")
            else:
                print("Wrong status_code")
            time.sleep(0.2)

    def answer_vlm(self,agent_trans_list,agent_query,image,episode_id):
        image = image
        output = self.send_and_receive(image_list= image,episode_id=episode_id)
        result_dict = {}
        pattern = re.compile(r'robot_(\d+):(\w+)_\[(\-?\d+),(\-?\d+)\]')
        pattern = re.compile(r'robot_(\d+):(\w+)_\[(\-?\d+),(\-?\d+)(?:,(\-?\d+))?\]')
        matches = re.findall(pattern,output)
        for match in matches:
            robotw,action,x,y,z = match
            z = z if z else None
            robot_num = int(robotw) - 1
            print("action",action)
            a = int(x)
            b = int(y)
            if z:
                c = int(z)
            if a == 0 and b == 0 and action == "nav_to_point":
                agent_key = f"agent_{robot_num}"
                result_dict[agent_key] = {
                    "name": action,
                    "position": [a,b] 
                }
                
            else:
                x = float(x) * 0.01
                y = float(y) * 0.01
                if z:
                    z = float(z) * 0.01
                agent_key = f"agent_{robot_num}"
                if z:
                    result_dict[agent_key] = {
                        "name": action,
                        "position": [x,y,z] 
                        }
                else:
                    result_dict[agent_key] = {
                        "name": action,
                        "position": [x,y] 
                        }
        num_query = 0
        output = {}
        print("agent___:",agent_query)
        print("vlm_out:",result_dict)
        for item in result_dict:
            print("agent_query[num_query]:",agent_query[num_query])
            print("item:",result_dict[item])
            if agent_query[num_query] == 1:
                pos = result_dict[item]['position']
                print(f"pos_0:{pos[0]}_pos_1:{pos[1]}")
                if pos == [0,0]:
                    pos.append(0)
                    output[item] = result_dict[item]
                else:
                    if len(pos) == 2:
                        pos.append(0.0)
                    print("pos:",pos)
                    if len(pos) == 3:
                        match = re.search(r'\d+',item)
                        num = -1
                        if match:
                            num = int(match.group())
                            result_dict[item]['position'] = self.pos_trans(pos=pos,
                                                                                trans=agent_trans_list[num]).flatten()[:3].tolist()
                    output[item] = result_dict[item]
                print("temp:",output[item])
            num_query+=1       
        return output 
    def check_vlm_ans(self,json):
        return True
    def answer(self,agent_trans_list,agent_query,image_list = None):
        if image_list != None:
            for i, image in enumerate(image_list):
                obs_k = image
                obs_k = obs_k.squeeze(0)
                if not isinstance(obs_k, np.ndarray):
                    obs_k = obs_k.cpu().numpy()
                if obs_k.dtype != np.uint8:
                    obs_k = obs_k * 255.0
                    obs_k = obs_k.astype(np.uint8)
                if obs_k.shape[2] == 1:
                    obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)
                plt.imshow(obs_k)
                plt.axis('off')
                plt.savefig(os.path.join(self.image_dir, str(self.step_num)+'_agent_'+str(i)+'.png'),
                            bbox_inches='tight', pad_inches=0)
            # vlm_output = self.vlm_inference(self.image_dir)
        test_output = self.test_inference(self.json_dir)
        self.step_num+=1
        query = [0] + agent_query
        filter_output = {key:value for i ,(key,value) in enumerate(test_output.items()) if 
                         i < len(query) and query[i] == 1}
        for item in filter_output:
            pos = filter_output[f"{item}"]['position']
            
            if len(pos) == 2:
                pos.append(0.0)
            print("pos:",pos)
            if len(pos) == 3:
                
                match = re.search(r'\d+',item)
                num = -1
                if match:
                    num = int(match.group())
                    filter_output[f"{item}"]['position'] = self.pos_trans(pos=pos,trans=agent_trans_list[num]).flatten()[:3].tolist()
        return filter_output

class VLMAgentSingle():
    def __init__(self,agent_num,image_dir,url,json_dir=None) -> None:
        self.image_dir = image_dir
        self.step_num = 0
        self.json_dir = json_dir
        self.url = url
        self.history = []
    def vlm_inference(self,image_dir):
        output = {}
        return output
    def test_inference(self,json_dir):
        with open(json_dir,'r') as file:
            data = json.load(file)
        step = self.step_num
        return data[step]
    def get_episode_info(self, episode_id):
        return True
    def send_and_receive(self,image_list,episode_id):
        images = []
        target_name = ""
        goal_name = ""
        headers = {
            "Content-Type": "application/json"
        }
        for image in image_list:
            image = image.squeeze(0).numpy().astype(np.uint8)
            image_PIL = Image.fromarray(image)
            buffered = BytesIO()
            image_PIL.save(buffered,format = 'PNG')
            images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        data = {
            "id": f"{episode_id}_{self.step_num}",
            "width": [256, 256],
            "height": [256, 256],
            "image": images,
            "prompt": (
                # f"describe what is in the picture."
                f"You are an AI visual assistant that can manage a single robot. You receive the robot's task, one image representing the robot's current view and what the robot has completed so far. You need to output the robot's next action. Actions the robot can perform are \"nav_to_point\", \"turn\", \"pick\" and \"place\". If you cannot determine the next action based on the robot's current view, you can command the robot to \"turn\". Your output format should be either \"turn\" or \"action_name<box>[[x1, y1, x2, y2]]</box>\".Robot's Task: The robot need to navigate to the {target_name} where the box is located, pick it up,navigate to the {goal_name} and place the box.\n    Robot's current view: <image>\n    ."
                # f"You are an AI visual assistant that can manage a single robot. You receive the robot's task, one image representing the robot's current view and what the robot has completed so far. You need to output the robot's next action. Actions the robot can perform are \"nav_to_point\", \"turn\", \"pick\" and \"place\". If you cannot determine the next action based on the robot's current view, you can command the robot to \"turn\". Your output format should be either \"turn\" or \"action_name<box>[[x1, y1, x2, y2]]</box>\".Robot's Task: The robot need to navigate to the {target_name} where the box is located, pick it up,navigate to the {goal_name} and place the box.\n    Robot's current view: <image>\n    .The robot has finished: navigate to the kitchen,pick the box,navigate to the chair."

            )
        }
        self.step_num+=1
        while True:
            response = requests.post(self.url, headers=headers, json=data)
            if response.status_code >= 200 and response.status_code < 300:
                return_json = response.json()
                try:
                    response_json = response.json()
                    if self.check_vlm_ans(response_json):
                        return response_json
                except:
                    print("invalid json")
            else:
                print("Wrong status_code")
            time.sleep(0.2)
    def _2d_to_3d_single_point(self, depth_obs, depth_rot,depth_trans,pixel_x, pixel_y):
        # depth_camera = self._sim._sensors[depth_name]._sensor_object.render_camera

        # hfov = float(self._sim._sensors[depth_name]._sensor_object.hfov) * np.pi / 180.
        # W, H = depth_camera.viewport[0], depth_camera.viewport[1]
        W = 256
        H = 256
        hfov = 1.5707963267948966
        # Intrinsic matrix K
        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0., 1, 0],
            [0., 0., 0, 1]
        ])
        
        # Normalize pixel coordinates
        xs = (2.0 * pixel_x / (W - 1)) - 1.0  # normalized x [-1, 1]
        ys = 1.0 - (2.0 * pixel_y / (H - 1))  # normalized y [1, -1]

        # Depth value at the pixel
        depth = depth_obs[0, pixel_x,pixel_y,0]

        # Create the homogeneous coordinates for the pixel in camera space
        xys = np.array([xs * depth, ys * depth, -depth, 1.0]).reshape(4, 1)
        
        # Apply the inverse of the intrinsic matrix to get camera space coordinates
        xy_c = np.matmul(np.linalg.inv(K), xys)

        # Get the rotation and translation of the camera
        depth_rotation = np.array(depth_rot)
        depth_translation = np.array(depth_trans)

        # Get camera-to-world transformation
        T_world_camera = np.eye(4)
        T_world_camera[0:3, 0:3] = depth_rotation
        T_world_camera[0:3, 3] = depth_translation

        # Apply transformation to get world coordinates
        T_camera_world = np.linalg.inv(T_world_camera)
        points_world = np.matmul(T_camera_world, xy_c)

        # Get non-homogeneous points in world space
        points_world = points_world[:3, :] / points_world[3, :]
        return points_world.flatten()
    def answer_vlm(self,agent_trans_list,agent_query,image,episode_id,depth_info,depth_rot, depth_trans,debug=False,debug_str=""):
        if debug:
            output = debug_str
        else:
            output = self.send_and_receive(image_list= image,episode_id=episode_id)
        result_dict = {}
        print("vlm_output:",output)
        if "turn" in output:
            return {"agent_0": {"name": "nav_to_point", "position": [0, 0, 0]}, "agent_1": {"name": "wait", "position": [0.0, 0.0, 0]}}    
        # pattern = re.compile(r'robot_(\d+):(\w+)_\[(\-?\d+),(\-?\d+)\]')
        match = re.search(r'(\w+)\s*\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]', output)
        if match:
            func_name = match.group(1)
            x = int(match.group(2))
            y = int(match.group(3))
            w = int(match.group(4))
            h = int(match.group(5))
            print(f"x:{x},y:{y},w:{w},h:{h},func:{func_name}")
            point_2d = (int((x+(w/2))*256/1000),int((y+(h/2))*256/1000))
            print("point2d:",point_2d,flush = True)
            point_3d = self._2d_to_3d_single_point(depth_info,depth_rot, depth_trans,point_2d[0],point_2d[1])
            print("point3d:",point_3d,flush = True)
            return {"agent_0": {"name": func_name, "position": point_3d.tolist()}, "agent_1": {"name": "wait", "position": [0.0, 0.0, 0]}}
        else:
            return {"agent_0": {"name": "nav_to_point", "position": [0, 0, 0]}, "agent_1": {"name": "wait", "position": [0.0, 0.0, 0]}}    
    def check_vlm_ans(self,json):
        return True
