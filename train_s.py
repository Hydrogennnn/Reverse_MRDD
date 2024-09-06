import argparse
from configs.basic_cfg import get_cfg


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # load config
    args = init_args()
    config = get_cfg(args.config_file)

