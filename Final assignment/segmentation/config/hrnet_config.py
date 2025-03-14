# segmentation/config/hrnet_config.py
import numpy as np

MODEL_CONFIGS = {
    'hrnet48': {
        'STAGE1': {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 1,
            'NUM_BLOCKS': [4],
            'NUM_CHANNELS': [64],
            'BLOCK': 'BOTTLENECK',
            'FUSE_METHOD': 'SUM'
        },
        'STAGE2': {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 2,
            'NUM_BLOCKS': [4, 4],
            'NUM_CHANNELS': [48, 96],
            'BLOCK': 'BASIC',
            'FUSE_METHOD': 'SUM'
        },
        'STAGE3': {
            'NUM_MODULES': 4,
            'NUM_BRANCHES': 3,
            'NUM_BLOCKS': [4, 4, 4],
            'NUM_CHANNELS': [48, 96, 192],
            'BLOCK': 'BASIC',
            'FUSE_METHOD': 'SUM'
        },
        'STAGE4': {
            'NUM_MODULES': 3,
            'NUM_BRANCHES': 4,
            'NUM_BLOCKS': [4, 4, 4, 4],
            'NUM_CHANNELS': [48, 96, 192, 384],
            'BLOCK': 'BASIC',
            'FUSE_METHOD': 'SUM'
        }
    }
}
