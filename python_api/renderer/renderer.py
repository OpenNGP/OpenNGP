from typing import List, Dict, Tuple
from python_api.primitive.primitive import Primitive
from .rays import Rays
from .renderpass import RenderPass, RenderPassResult


class Renderer:
    def __init__(self, pass_config: List[Tuple[str, Dict]]) -> None:
        self.passes = [RenderPass(*p) for p in pass_config]

    def __repr__(self):
        return '\n'.join([str(p) for p in self.passes])

    def render(self, rays: Rays, primitive: Primitive, context: Dict = {}) -> List[RenderPassResult]:
        outputs, ctx = [], context
        for renderpass in self.passes:
            rets = renderpass.render_pixel(rays, primitive, ctx)
            ctx = rets._asdict()
            outputs.append(rets)
        return outputs