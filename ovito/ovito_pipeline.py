"""
OVITO Python script (optional).
Usage from shell:
ovitos ovito/ovito_pipeline.py sim/out/traj.gsd
It sets size/coloring and defines a viewport layout to highlight IN/MID/OUT.
"""
import sys
from ovito.io import import_file
from ovito.vis import Viewport
from ovito.data import Particles


if len(sys.argv) < 2:
print('Usage: ovitos ovito/ovito_pipeline.py sim/out/traj.gsd')
sys.exit(1)


pipe = import_file(sys.argv[1])
# Display tweaks
from ovito.vis import ParticlesVis
vis = pipe.scene.objects[0].vis
if isinstance(vis, ParticlesVis):
vis.radius = 0.6
vis.coloring = ParticlesVis.ColoringScheme.Uniform


vp = Viewport(type=Viewport.Type.Perspective)
vp.zoom_all()
vp.render_image(filename='sim/out/preview.png', size=(1600, 900))
print('Rendered sim/out/preview.png')