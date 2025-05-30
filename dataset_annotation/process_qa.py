import json
class QA_process():
    def __init__(self,target_rec_name,goal_rec_name,obj_item,scene_frame_num,arm_or_head:str):
        self.summary = ""
        self.target_rec = target_rec_name
        self.goal_rec = goal_rec_name
        self.obj_item = obj_item
        self.scene_frame_num = scene_frame_num
        self.arm_or_head = arm_or_head
    def check_bbox(self, bbox):
        x,y,w,h = bbox
        # print(f"x:{x}/y:{y}/w:{w}/h:{h}")
        if x < 0 or y < 0 or x + w > 512 or y + h > 512:
            return False
        return True
    def determine_position(self,bbox, image_size=(512, 512)):
        if not self.check_bbox(bbox):
            raise ValueError(f"Invalid bounding box:{bbox}")
        x, y, w, h = bbox
        img_w, img_h = image_size

        # 计算中心点
        center_x = x + w / 2
        center_y = y + h / 2

        if center_x < img_w / 3:
            horizontal = 'left'
        elif center_x > 2 * img_w / 3:
            horizontal = 'right'
        else:
            horizontal = 'center'

        if center_y < img_h / 3:
            vertical = 'top'
        elif center_y > 2 * img_h / 3:
            vertical = 'bottom'
        else:
            vertical = 'center'

        if horizontal == 'center' and vertical == 'center':
            return 'center'
        return f"{vertical} {horizontal}"
    def determine_position_for_obj(self, bbox, image_size=(512, 512)):
        if not self.check_bbox(bbox):
            return None
        x, y, w, h = bbox
        img_w, img_h = image_size

        center_x = x + w / 2
        center_y = y + h / 2

        if center_x < img_w / 3:
            horizontal = 'left'
        elif center_x > 2 * img_w / 3:
            horizontal = 'right'
        else:
            horizontal = 'center'

        if center_y < img_h / 3:
            vertical = 'top'
        elif center_y > 2 * img_h / 3:
            vertical = 'bottom'
        else:
            vertical = 'center'

        if horizontal == 'center' and vertical == 'center':
            return 'center'
        return f"{vertical} {horizontal}"


    def process_questions(self,arm_or_head:str):
        if len(self.summary) == 0:
            robot_history = ""
        else:
            robot_history = f"Robot's history:\"{self.summary}\""
        json_format_info = {
            "reasoning":f"Based on the current image information and history, think and infer the actions that need to be executed and action's information.",
            "action":"The action name of your reasoning result.",
            "action_information":"If the action is \"nav_to_point\",\"pick\" or \"place\", the information format should be \"<box>[[x1, y1, x2, y2]]</box>\" to indicate location information,where x1, y1, x2, y2 are the coordinates of the bounding box;if the action is \"search_scene_frame\",the information format should be \"id\",where id is the value of the frame index you choose.",
            "summarization":"summarize based on the current action information and history."
        }
        question_prompt_align_with_ovmm = f"""You are an AI visual assistant that can manage a single robot. You receive the robot's task,one image representing the robot's current view and eight frames of the scene from the robot's tour. You need to output the robot's next action. Actions the robot can perform are "search_scene_frame","nav_to_point","pick" and "place".
These frames are from the robot's tour of the scene:
Image-1:<image>
Image-2:<image>
Image-3:<image>
Image-4:<image>
Image-5:<image>
Image-6:<image>
Image-7:<image>
Image-8:<image>
If you can not find the target you need to identify,you should find the frame that the robot should navigate to complete the task,and output "search_scene_frame" action and the id of frame.
Robot's current view is: Image-9:<image>.If you can find the position that you should navigate to, pick or place,you should output your action information.In robot's current view, some green points may appear,indicating the positions that the robot's arm can reach. When there are green points on the object that needs to be picked, it means the robot's arm can pick up the object. When there are enough green points on the goal container where the object needs to be placed, it means the robot's arm can place the object into the goal container.
Besides,you need to explain why you choose this action in your output and summarize by combining your chosen action with historical information.
Robot's Task: Move {self.obj_item}from the {self.target_rec} to the {self.goal_rec}.{robot_history}Your output format should be in pure JSON format as follow:{json_format_info}."""
        question_prompt_align_with_ovmm_no_gp = f"""You are an AI visual assistant that can manage a single robot. You receive the robot's task,one image representing the robot's current view and eight frames of the scene from the robot's tour. You need to output the robot's next action. Actions the robot can perform are "search_scene_frame","nav_to_point","pick" and "place".
These frames are from the robot's tour of the scene:
Image-1:<image>
Image-2:<image>
Image-3:<image>
Image-4:<image>
Image-5:<image>
Image-6:<image>
Image-7:<image>
Image-8:<image>
If you can not find the target you need to identify,you should find the frame that the robot should navigate to complete the task,and output "search_scene_frame" action and the id of frame.
Robot's current view is: Image-9:<image>.If you can find the position that you should navigate to, pick or place,you should output your action information.
Besides,you need to explain why you choose this action in your output and summarize by combining your chosen action with historical information.
Robot's Task: Move {self.obj_item}from the {self.target_rec} to the {self.goal_rec}.{robot_history}Your output format should be in pure JSON format as follow:{json_format_info}."""
        if arm_or_head == "head_rgb":
            return question_prompt_align_with_ovmm_no_gp
        elif arm_or_head == "arm_workspace_rgb":
            return question_prompt_align_with_ovmm
        else:
            return None
        #DO NOT FORGET CHANGE THE ANNOTATION
    def first_search_image_path_answer_old(self,k_frame):
        answer = {
            "reasoning":f"My history indicates that I am just beginning my task.Based on my task,I must first navigate to {self.target_rec} where the {self.obj_item} is located. In my current view, I can not see {self.target_rec}, so I need to search scene frames. In  Image-{k_frame}, I can see {self.target_rec}, so the Image-{k_frame} is what I should choose.",
            "action":"search_scene_frame",
            "action_information":k_frame,
            "summarization":f"The task has started and I am navigating to find the {self.target_rec} where the {self.obj_item} is located."
            }
        self.summary = f"The task has started and I am navigating to find the {self.target_rec} where the {self.obj_item} is located."
        return answer
    def first_search_image_path_answer_generate(self,k_frame):#THIS IS FOR GPT anno image
        answer = {
            "reasoning":f"My history indicates that I am just beginning my task.Based on my task,I must first navigate to {self.target_rec} where the {self.obj_item} is located. In my current view, I can not see {self.target_rec}, so I need to search scene frames.",
            "anno_prompt_for_gpt":f"There are eight images from scene graph images.The robot's current task is to navigate to the {self.target_rec}.As the robot can not find the target from its current view,it need to navigate to one of the images which contain its target.For ground truth,the robot need to navigate to the Image-{k_frame} as there is only Image-{k_frame} contain the {self.target_rec}.Now you are the robot.Your output need to first consider your own tasks,describe each image in DETAIL and deduce which image to select based on the ground truth I provide,but pretend that you do not know the ground truth.Please ensure your output is in the first-person format and does not contain line breaks.",
            "action":"search_scene_frame",
            "action_information":k_frame,
            "summarization":f"The task has started and I am navigating to find the {self.target_rec} where the {self.obj_item} is located."
            }
        self.summary = f"The task has started and I am navigating to find the {self.target_rec} where the {self.obj_item} is located."
        return answer
    def first_search_image_path_answer(self,k_frame):  #DO NOT FORGET CHANGE THE ANNOTATION
        answer = k_frame
        return answer
    
    def first_nav_answer(self,target_rec_info,nav_point_info,action_info):
        answer = {
            "reasoning":f"I need to navigate to {self.target_rec} where the {self.obj_item} is located, pick the {self.obj_item} up,navigate to the {self.goal_rec} and place the {self.obj_item}.My history indicates that I am searching for {self.target_rec} to pick the {self.obj_item}, so I must continue to navigate to {self.target_rec} where the {self.obj_item} is located. In my current view, I can see that {self.target_rec} is located on the {target_rec_info} of the image.As there is not green point on the objects,I need to continue to navigate to the navigable area which is closer to {self.target_rec}.I can determine that the ground in the {nav_point_info} of the image is a navigable area.",
            "action":"nav_to_point",
            "action_information":f"{action_info}",
            "summarization":f"The task has started and I am getting closer to {self.target_rec} where the {self.obj_item} is located."
            }
        self.summary = f"The task has started and I am getting closer to {self.target_rec} where the {self.obj_item} is located."
        return answer
    def first_nav_answer_no_gp(self,target_rec_info,nav_point_info,action_info,object_info=None):
        answer = {
            "reasoning":f"I need to navigate to {self.target_rec} where the {self.obj_item} is located, pick the {self.obj_item} up,navigate to the {self.goal_rec} and place the {self.obj_item}.My history indicates that I am searching for {self.target_rec} to pick the {self.obj_item}, so I must continue to navigate to {self.target_rec} where the {self.obj_item} is located. In my current view, I can see that {self.target_rec} is located on the {target_rec_info} of the image.As I am not close enough to attach the object,I need to continue to navigate to the navigable area which is closer to {self.obj_item}.I can determine that the ground in the {nav_point_info} of the image is a navigable area.",
            "action":"nav_to_point",
            "action_information":f"{action_info}",
            "summarization":f"The task has started and I am getting closer to {self.target_rec} where the {self.obj_item} is located."
            }
        self.summary = f"The task has started and I am getting closer to {self.target_rec} where the {self.obj_item} is located."
        return answer
    def first_nav_answer_add_object_anno(self,target_rec_info,nav_point_info,action_info,object_info = None):
        if object_info:
            object_prompt = f"Additionally, I also see the {self.obj_item} is in the {object_info} of the image. Since there is not green point on the {self.obj_item}, I should continue navigating closer to it. Considering the {self.obj_item} is in the {object_info} of the image,and the navigable area of the ground,I can determine that the ground in the {nav_point_info} of the image is the perfect choice to navigate to."
        else:
            object_prompt = f"However, I didn't find the {self.obj_item} in the image. I need to decide my next navigation direction based on the {self.target_rec}'s position in the image to increase my chances of seeing the {self.obj_item}. Since the {self.target_rec} is mainly on the {target_rec_info} of the image, I should attach to the {target_rec_info} area. Considering the navigable regions, I ultimately chose the the ground in the {nav_point_info} of the image as my next navigation target."
        answer = {
            "reasoning":f"Based on my task,I need to navigate to {self.target_rec} where the {self.obj_item} is located, pick the {self.obj_item} up,navigate to the {self.goal_rec} and place the {self.obj_item}.My history indicates that I am searching for {self.target_rec} to pick the {self.obj_item}, so I must continue to navigate to {self.target_rec} where the {self.obj_item} is located. In my current view, I can see that {self.target_rec} is located on the {target_rec_info} of the image.{object_prompt}",
            "action":"nav_to_point",
            "action_information":f"{action_info}",
            "summarization":f"The task has started and I am getting closer to {self.target_rec} where the {self.obj_item} is located."
            }
        self.summary = f"The task has started and I am getting closer to {self.obj_item} located on the {self.target_rec}."
        return answer
    def pick_answer(self,target_rec_info,obj_info,action_info,object_info = None):
        answer = {
            "reasoning":f"Based on my task,I need to navigate to {self.target_rec} where the {self.obj_item} is located, pick the {self.obj_item} up,navigate to the {self.goal_rec} and place the {self.obj_item}.My history indicates that I am getting closer to {self.target_rec} where the {self.obj_item} is located. In my current view, I can see that {self.target_rec} is located on the {target_rec_info} of the image, and I can find that the {self.obj_item} is located on the {obj_info} of the image.The most important is there are some green points on the {self.obj_item},so it is close enough for my arm to pick it up.Therefore, I need to pick up the {self.obj_item}",
            "action":"pick",
            "action_information":f"{action_info}",
            "summarization":f"The task has started and I have navigated to {self.target_rec} and picked up the {self.obj_item}."
            }
        return answer
    def pick_answer_no_gp(self,target_rec_info,obj_info,action_info,object_info = None):
        answer = {
            "reasoning":f"Based on my task,I need to navigate to {self.target_rec} where the {self.obj_item} is located, pick the {self.obj_item} up,navigate to the {self.goal_rec} and place the {self.obj_item}.My history indicates that I am getting closer to {self.target_rec} where the {self.obj_item} is located. In my current view, I can see that {self.target_rec} is located on the {target_rec_info} of the image, and I can find that the {self.obj_item} is located on the {obj_info} of the image.The most important is it is close enough for my arm to pick {self.obj_item} up.Therefore, I need to pick up the {self.obj_item}",
            "action":"pick",
            "action_information":f"{action_info}",
            "summarization":f"The task has started and I have navigated to {self.target_rec} and picked up the {self.obj_item}."
            }
        return answer
    def second_search_image_path_answer_old(self,k_frame):
        answer = {
            "reasoning":f"My history indicates that I have navigated to {self.target_rec} and picked up the {self.obj_item}.Based on my task,now I need to navigate to the {self.goal_rec}.In my current view, I cannot see {self.goal_rec}, so I need to search scene frames. In scene Image-{k_frame}, I can see {self.goal_rec}, so the Image-{k_frame} is what I should choose.",
            "action":"search_scene_frame",
            "action_information":k_frame,
            "summarization":f"The task has started and I have navigated to {self.target_rec} and picked up the {self.obj_item}.I am navigating to find the {self.goal_rec} where I should place {self.obj_item}."
            }
        self.summary = f"The task has started and I have navigated to {self.target_rec} and picked up the {self.obj_item}, I am navigating to find the {self.goal_rec} where I should place {self.obj_item}."
        return answer
    def second_search_image_path_answer(self,k_frame):  #DO NOT FORGET CHANGE THE ANNOTATION,THIS IS TEST ID ONLY
        answer = k_frame
        return answer
    def second_search_image_path_answer_generate(self,k_frame): #THIS IS FOR GPT anno image
        answer = {
            "reasoning":f"My history indicates that I have navigated to {self.target_rec} and picked up the {self.obj_item}.Based on my task,now I need to navigate to the {self.goal_rec}.In my current view, I cannot see {self.goal_rec}, so I need to search scene frames.",
            "anno_prompt_for_gpt":f"There are eight images from scene graph images.The robot's current task is to navigate to the {self.goal_rec}.As the robot can not find the target from its current view,it need to navigate to one of the images which contain its target.For ground truth,the robot need to navigate to the Image-{k_frame} as there is only Image-{k_frame} contain the {self.goal_rec}.Now you are the robot.Your output need to first consider your own tasks,describe each image in DETAIL and deduce which image to select based on the ground truth I provide,but pretend that you do not know the ground truth.Please ensure your output is in the first-person format and does not contain line breaks.",
            "action":"search_scene_frame",
            "action_information":k_frame,
            "summarization":f"The task has started and I have navigated to {self.target_rec} and picked up the {self.obj_item}.I am navigating to find the {self.goal_rec} where I should place {self.obj_item}."
            }
        self.summary = f"The task has started and I have navigated to {self.target_rec} and picked up the {self.obj_item}, I am navigating to find the {self.goal_rec} where I should place {self.obj_item}."
        return answer

    def second_nav_answer(self,goal_rec_info,nav_point_info,action_info):
        answer = {
            "reasoning":f"Based on my task,I need to navigate to {self.target_rec} where the {self.obj_item} is located, pick the {self.obj_item} up,navigate to the {self.goal_rec} and place the {self.obj_item}.My history indicates that I have picked up {self.obj_item} and am searching for {self.goal_rec}.I must continue to navigate to {self.goal_rec} and place {self.obj_item}. In my current view, I can see that {self.goal_rec} is located on the {goal_rec_info} of the image.As there are not enough green points on the {self.goal_rec},I need to get closer to it.I can determine that the ground in the {nav_point_info} of the image is a navigable area.",
            "action":"nav_to_point",
            "action_information":f"{action_info}",
            "summarization":f"The task has started and I have navigated to {self.target_rec} and picked up the {self.obj_item}.I am getting closer to {self.goal_rec} where I should place {self.obj_item}."
            }
        self.summary = f"The task has started and I have navigated to {self.target_rec} and picked up the {self.obj_item}, I am getting closer to {self.goal_rec} where I should place {self.obj_item}."
        return answer
    def second_nav_answer_no_gp(self,goal_rec_info,nav_point_info,action_info):
        answer = {
            "reasoning":f"Based on my task,I need to navigate to {self.target_rec} where the {self.obj_item} is located, pick the {self.obj_item} up,navigate to the {self.goal_rec} and place the {self.obj_item}.My history indicates that I have picked up {self.obj_item} and am searching for {self.goal_rec}.I must continue to navigate to {self.goal_rec} and place {self.obj_item}. In my current view, I can see that {self.goal_rec} is located on the {goal_rec_info} of the image.As I am not close enough to attach the {self.goal_rec},I need to get closer to it.I can determine that the ground in the {nav_point_info} of the image is a navigable area.",
            "action":"nav_to_point",
            "action_information":f"{action_info}",
            "summarization":f"The task has started and I have navigated to {self.target_rec} and picked up the {self.obj_item}.I am getting closer to {self.goal_rec} where I should place {self.obj_item}."
            }
        self.summary = f"The task has started and I have navigated to {self.target_rec} and picked up the {self.obj_item}, I am getting closer to {self.goal_rec} where I should place {self.obj_item}."
        return answer
    def place_answer(self,goal_rec_info,action_info):
        answer ={
            "reasoning":f"Based on my task,I need to navigate to {self.target_rec} where the {self.obj_item} is located, pick the {self.obj_item} up,navigate to the {self.goal_rec} and place the {self.obj_item}.My history indicates that the I have picked up {self.obj_item} and am getting closer to {self.goal_rec}. In my current view, I can see that {self.goal_rec} is located on the {goal_rec_info} of the image.As there are several green points on {self.goal_rec},it is close enough for me to place {self.obj_item}.Therefore, I need to place the {self.obj_item} on the {self.goal_rec}",
            "action":"place",
            "action_information":f"{action_info}",
            "summarization":f"As I have placed the {self.obj_item} on the {self.goal_rec},I have finished all the task."
            }
        return answer
    def place_answer_no_gp(self,goal_rec_info,action_info):
        answer ={
            "reasoning":f"Based on my task,I need to navigate to {self.target_rec} where the {self.obj_item} is located, pick the {self.obj_item} up,navigate to the {self.goal_rec} and place the {self.obj_item}.My history indicates that the I have picked up {self.obj_item} and am getting closer to {self.goal_rec}. In my current view, I can see that {self.goal_rec} is located on the {goal_rec_info} of the image.Besides,it is close enough to {self.goal_rec} for me to place {self.obj_item}.Therefore, I need to place the {self.obj_item} on the {self.goal_rec}",
            "action":"place",
            "action_information":f"{action_info}",
            "summarization":f"As I have placed the {self.obj_item} on the {self.goal_rec},I have finished all the task."
            }
        return answer
    def get_conversations(self,action_name,bbox_dict,rec_index = None,flag_if_picked = False):
        if action_name == "search_for_goal_rec":
            self.summary = f"The task has started and I have navigated to {self.target_rec} and picked up the {self.obj_item}."
        question = self.process_questions(arm_or_head = self.arm_or_head)
        if action_name == "search_for_object_rec":
            answer = self.first_search_image_path_answer_old(rec_index[0]+1)
        if action_name == "search_for_goal_rec":
            answer = self.second_search_image_path_answer_old(rec_index[1]+1)
        if action_name == "nav_to_point" and not flag_if_picked:
            target_info = self.determine_position(bbox_dict['target_bbox'][0])
            # print("bbox_dict['nav_point_bbox']:",bbox_dict['nav_point_bbox'])
            nav_point_info = self.determine_position(bbox_dict['nav_point_bbox'][0])
            object_info = self.determine_position_for_obj(bbox_dict['obj_bbox'][0])
            answer = self.first_nav_answer(target_info,nav_point_info,bbox_dict['agent_0_pos']) if self.arm_or_head == "arm_workspace_rgb" else self.first_nav_answer_no_gp(target_info,nav_point_info,bbox_dict['agent_0_pos'])
        if action_name == "nav_to_point" and flag_if_picked:
            goal_info = self.determine_position(bbox_dict['goal_bbox'][0])
            nav_point_info = self.determine_position(bbox_dict['nav_point_bbox'][0])
            answer = self.second_nav_answer(goal_info,nav_point_info,bbox_dict['agent_0_pos']) if self.arm_or_head == "arm_workspace_rgb" else self.second_nav_answer_no_gp(goal_info,nav_point_info,bbox_dict['agent_0_pos'])
        if action_name == "pick":
            target_info = self.determine_position(bbox_dict['target_bbox'][0])
            obj_info = self.determine_position(bbox_dict['obj_bbox'][0])
            answer = self.pick_answer(target_info,obj_info,bbox_dict['agent_0_pos']) if self.arm_or_head == "arm_workspace_rgb" else self.pick_answer_no_gp(target_info,obj_info,bbox_dict['agent_0_pos'])
        if action_name == "place":
            goal_info = self.determine_position(bbox_dict['goal_bbox'][0])
            answer = self.place_answer(goal_info,bbox_dict['agent_0_pos']) if self.arm_or_head == "arm_workspace_rgb" else self.place_answer_no_gp(goal_info,bbox_dict['agent_0_pos'])
        conversations = [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": json.dumps(answer)
            }
        ]
        return conversations
# qa = QA_process()
# print(qa.process_questions("bed","desk","chair"))