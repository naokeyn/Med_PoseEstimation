import json
from argparse import ArgumentParser

def load_json(path):
    with open(path, "r", encoding="utf-8") as r:
        data = json.load(r)
    return data

def get_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="path to config file")
    args = parser.parse_args()
    config = load_json(args.config)
    
    return config


if __name__ == "__main__":
    config = get_args()
    print(config)
