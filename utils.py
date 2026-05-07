import json
import os
import pickle
import random

import numpy as np
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def print_opts(opts):
    """Prints the values of all command-line arguments."""
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if isinstance(data, dict) or isinstance(data, list):
                return data
    except json.JSONDecodeError:
        print("File content is not valid JSON format or file is empty")
    except FileNotFoundError:
        print("Specified file does not exist")
    except Exception as e:
        print(f"Error reading file: {e}")

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print("File not found, please check the path")
    except Exception as e:
        print(f"Error reading file: {e}")

def list_filenames(directory, file_type="json", sort=True, sort_key=None):
    dirs = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path) and item.endswith("." + file_type if "." not in file_type else file_type):
            dirs.append(full_path)
    if sort:
        dirs = [(x, float(x.split(f"{sort_key}_")[-1].split(".json")[0].split("_")[0])) for x in dirs]
        dirs.sort(key=lambda x: x[1])
    return dirs

def read_file(file_path, file_type):
    if file_type == "txt":
        return read_text_file(file_path)
    elif file_type == "json":
        return read_json_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
