from amworkflow.src.interface.api import amWorkflow as aw
@aw.engine.amworkflow()
def geometry_spawn(pm):
    box = aw.geom.create_box(length=pm.length,
                        width= pm.width,
                        height=pm.height,
                        radius=pm.radius)
    return box