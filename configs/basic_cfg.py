from yacs.config import CfgNode as CN


_C = CN()

# SYSTEM CONFIG
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 4
_C.SYSTEM.NUM_WORKERS = 4


_C.train = CN()
_C.train.optimizer = "Adam"
#禁用严格模式
_C.set_new_allowed(True)

def get_cfg(cfg_path):
    config = _C.clone()

    config.merge_from_file(cfg_path)
    return config