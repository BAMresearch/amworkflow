import pygmsh
import pyvista as pv
import math as m

def wall_mesher(radius: float,
                width: float,
                height: float,
                length: float,
                alpha: float = None):
    with pygmsh.geo.Geometry() as geom:
        if alpha == None:
            alpha = (length / radius)% (m.pi * 2)
        R = radius + (width / 2)
        r = radius - (width / 2)
        p1 = geom.add_point([0,0,0])
        p0 = geom.add_point([-radius,0,0])
        p2 = geom.add_point([R - r * m.cos(alpha),r * m.sin(alpha),0])
        p11 = geom.add_point([width,0,0])
        p22 = geom.add_point([(1 - m.cos(alpha)) * R, R * m.sin(alpha),0])
        line1 = geom.add_line(p1, p11)
        line2 = geom.add_line(p22, p2)
        curve1 = geom.add_circle_arc(p2, p0, p1 )
        curve2 = geom.add_circle_arc(p11, p0, p22 )
        curve_loop = geom.add_curve_loop([line1, curve2, line2, curve1])
        surface = geom.add_plane_surface(curve_loop)
        extrude = geom.extrude(surface, (0,0,height), num_layers=10)
        # Generate the mesh
        mesh = geom.generate_mesh()
        return mesh
    
    