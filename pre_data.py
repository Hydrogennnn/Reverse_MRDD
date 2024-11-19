from utils.datatool import (get_val_transformations,
                            get_train_dataset,
                            get_val_dataset)
from configs.basic_cfg import get_cfg
import os
from utils.datatool import __dataset_dict
import json
import random

res_dir = "./MaskView"
def generate(data_set, path, config):
    file_dir = os.path.join(res_dir, path)
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    # Write the file

    num_to_select = int(len(data_set) * config.eval.modal_missing_ratio)
    random_indices = random.sample(range(len(data_set)), num_to_select)
    random_views = [random.randint(0, config.views - 1) for _ in range(num_to_select)]
    print('name', config.dataset.name, 'seed:', config.seed, 'len:', num_to_select)
    file_path= os.path.join(file_dir, config.dataset.name + ".json")
    with open(file_path, "w") as file:
        json.dump({'indices': random_indices, 'views': random_views}, file)

def main():
    cfg_file_list = ["./configs/emnist.yaml",
                     "./configs/fmnist.yaml",
                     "./configs/coil-100.yaml"]

    # for key in __dataset_dict.keys():
    #     dataclass = __dataset_dict.get(key, None)
    #     train_set = dataclass(root='MyData', train=False,
    #                         transform=get_val_transformations(), download=True)

    for cfg_path in cfg_file_list:
        config = get_cfg(cfg_path)
        data_class = __dataset_dict.get(config.dataset.name, None)
        if data_class is None:
            raise ValueError("Dataset name error.")
        train_set = data_class(root=config.dataset.root, train=True,
                               transform=get_val_transformations(config), views=config.views)
        val_set = data_class(root=config.dataset.root, train=False,
                             transform=get_val_transformations(config), views=config.views)

        # reproduct
        seed = config.seed
        random.seed(seed)
        generate(train_set, "train", config)
        generate(val_set, "test", config)




if __name__ == "__main__":
    main()



