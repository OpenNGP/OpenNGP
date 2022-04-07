from collections import namedtuple

Rays = namedtuple('Rays', ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
