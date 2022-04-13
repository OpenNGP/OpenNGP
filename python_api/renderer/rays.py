from collections import namedtuple

Rays = namedtuple(
    'Rays',
    (
        'origins',
        'directions',
        'viewdirs',
        'radii',
        'lossmult',
        'near',
        'far'
    )
)


RaysWithDepth = namedtuple(
    'RaysWithDepth',
    (
        'idx',
        'ray_idx',
        'origins',
        'directions',
        'viewdirs',
        'radii',
        'lossmult',
        'near',
        'far',
        'depth',
        'mask'
    )
)
