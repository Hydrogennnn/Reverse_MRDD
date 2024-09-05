from yacs.config import CfgNode as CN


_C = CN()

# SYSTEM CONFIG
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 4
_C.SYSTEM.NUM_WORKERS = 4


_C.train = CN()
_C.train.optimizer = "Adam"


def get_cfg()
