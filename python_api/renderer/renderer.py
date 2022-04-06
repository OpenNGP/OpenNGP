from typing import List, Tuple
from python_api.primitive.primitive import Primitive
from .rays import Rays
from .renderpass import RenderPass


class Renderer:
    def __init__(self, pass_config: List[Tuple[str, str]]) -> None:
        self.passes = [RenderPass(p[0], p[1]) for p in pass_config]

    def render(self, rays: Rays, primitive: Primitive):
        outputs, ctx = [], {}
        for renderpass in self.passes:
            rets = renderpass.render_pixel(rays, primitive, **ctx)
            ctx = rets._asdict()
            outputs.append(ctx)
        return outputs