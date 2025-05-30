from ruamel.yaml import YAML
import os
yaml = YAML()
llm_yaml_path = './102344193.yaml'
def get_numbers_from_filenames(directory):
    numbers = []
    for filename in os.listdir(directory):
        if filename.endswith('.scene_instance.json'):
            number = filename.split('.scene_instance')[0]
            numbers.append(number)
    return numbers
# scene_config_directory_path = 'data/scene_datasets/hssd-hab/scenes'
# scene_sample = get_numbers_from_filenames(scene_config_directory_path)
# files_name = ["104862384_172226319","106878960_174887073","108736851_177263586","108736824_177263559","107734254_176000121"]
# files_name = scene_sample
def extract_keys_from_txt(file_path):
    keys = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or not line.startswith('(') or not line.endswith(')'):
                continue
            try:
                content = line[1:-1].split(',', 1)
                key = content[0].strip().strip("'")
                keys.append(key)
            except Exception as e:
                print(f"Error processing line: {line}, Error: {e}")
    return keys

file_path = 'data_1225_scene.txt'
# keys = extract_keys_from_txt(file_path)
keys = ['102343992', '102815835',]
#failure_yaml
files_name = keys
for file_id in files_name:
    with open(llm_yaml_path,'r') as file:
        data = yaml.load(file)
        for scene_set in data['scene_sets']:
            if scene_set['name'] == 'test':
                scene_set['included_substrings'] = [file_id]
        scene_yaml_path = './allycb_dir'
        if not os.path.exists(scene_yaml_path):
            os.mkdir(scene_yaml_path)
        with open(os.path.join(scene_yaml_path,f'{file_id}.yaml'),'w') as file:
            yaml.dump(data,file)