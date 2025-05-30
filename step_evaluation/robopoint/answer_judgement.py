import json
import numpy as np
from scipy import ndimage
def calculate_center(points):
    return np.mean(points, axis=0)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def process_line(line):
    data = json.loads(line)
    annotation = json.loads(data["conversations"][1]["value"])
    gt_action = annotation["action"]
    gt_action_information = json.loads(str(annotation["action_information"]))
    try:
        prediction = json.loads(data["prediction"])
    except:
        return gt_action,-1,-1,-1,-1,-1,0
    pre_action = prediction["action"]
    if gt_action_information == [-1,-1]:
        return gt_action,-1,-1,-1,-1,-1,0
    if gt_action != pre_action:
        return gt_action, 0, 0, data["prompt_tokens"],data["completion_tokens"],data["time_total"],0
    if gt_action == "search_scene_frame":
        try:
            pre_action_information = json.loads(str(prediction["answer_info"]))
        except:
            return gt_action,-1,-1,-1,-1,-1,0
        return gt_action, str(gt_action_information) == str(pre_action_information), -1, data["prompt_tokens"],data["completion_tokens"],data["time_total"],1
    gt_x, gt_y, gt_w, gt_h = gt_action_information[0]
    ground_truth = [gt_x, gt_y, gt_x + gt_w, gt_y + gt_h]
    pre_action_information = str(prediction["answer_info"])
    prediction_str = pre_action_information.strip(" []")
    prediction_points = [tuple(map(float, point.strip("()").split(", "))) for point in prediction_str.split("), (")]
    prediction_center = calculate_center(prediction_points)
    ground_truth_center = calculate_center([
        [ground_truth[0], ground_truth[1]],
        [ground_truth[2], ground_truth[3]]
    ])
    pre_num_center = [prediction_center[0]*1000,prediction_center[1]*1000]
    l2_distance = np.linalg.norm(ground_truth_center - pre_num_center)
    success_value = 1 - l2_distance / (1000 * np.sqrt(2))
    return gt_action, success_value, 0, data["prompt_tokens"],data["completion_tokens"],data["time_total"],1

def main():
    total_success_values = {}
    total_ious = {}
    total_ans = {}
    count_actions = {}
    total_prompt_tokens = []
    total_completion_tokens = []
    total_time = []
    total_item = 0
    action_choose = []
    with open('robopoint_vanilla_gpt.jsonl', 'r') as file:
        for i,line in enumerate(file):
            action, success_value, iou,prompt_tokens,completion_tokens,time_total,action_bool = process_line(line)
            if action not in total_success_values:
                total_success_values[action] = 0
                total_ans[action] = []
                total_ious[action] = 0
                count_actions[action] = 0
            if success_value == -1:
                total_success_values[action] += 0
                total_ious[action] += 0
                total_ans[action].append(0)
            else:
                total_item+=1
                total_success_values[action] += success_value
                total_ans[action].append(success_value)
                total_ious[action] += iou
                total_prompt_tokens.append(prompt_tokens)
                total_completion_tokens.append(completion_tokens)
                total_time.append(time_total)
            action_choose.append(action_bool)
            count_actions[action] += 1
    for action in total_success_values:
        avg_success_value = total_success_values[action] / count_actions[action]
        avg_iou = total_ious[action] / count_actions[action]
        print(f"Action: {action}, Std:{ndimage.standard_deviation(np.array(total_ans[action]))},Avg Success Value: {avg_success_value:.4f}, Avg IoU: {avg_iou:.4f}")
    avg_prompt_tokens = sum(total_prompt_tokens)/ len(total_prompt_tokens)
    avg_completion_tokens  = sum(total_completion_tokens) / len(total_completion_tokens)
    avg_time_ = sum(total_time)/ len(total_time)
    print(f"avg_prompt_tokens:{avg_prompt_tokens}")
    print(f"avg_completion_tokens:{avg_completion_tokens}")
    print(f"avg_time_:{avg_time_}")
    print(f"action_choose_acc:{sum(action_choose)/ len(action_choose)}")
    print(f"success load:{total_item}")
if __name__ == "__main__":
    main()
