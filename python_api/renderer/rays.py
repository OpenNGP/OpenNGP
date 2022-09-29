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

RaysWithDepthCos = namedtuple(
    'RaysWithDepthCos',
    (
        'origins',
        'directions',
        'viewdirs',
        'radii',
        'lossmult',
        'near',
        'far',
        'depth_cos'
    )
)


RaysWithDepthCos2 = namedtuple(
    'RaysWithDepthCos2',
    (
        'ray_idx',
        'origins',
        'directions',
        'viewdirs',
        'radii',
        'lossmult',
        'near',
        'far',
        'depth_cos'
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
        'depth_cos',
        'depth',
        'mask'
    )
)
