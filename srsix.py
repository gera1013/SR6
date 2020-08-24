from gl import Render, color
from obj import ObjFile, Texture
from shaders import *

r = Render(768, 432)

r.texture = Texture('model.bmp')
r.active_shader = phong

posModel = [0, 0, -3]

# HIGH ANGLE
print('Dibujando high angle...')
r.lookAt(posModel, [0, 1.5, -0.7])
r.glLoadObj('model.obj', posModel, [1, 1, 1], [0, 0, 0])
r.glFinish('high.bmp')
print('Finalizado')

# MEDIUM ANGLE
r.glClear()
print('\nDibujando medium angle...')
r.lookAt(posModel, [0, 0, -0.5])
r.glLoadObj('model.obj', posModel, [1, 1, 1], [0, 0, 0])
r.glFinish('medium.bmp')
print('Finalizado')

# LOW ANGLE
r.glClear()
print('\nDibujando low angle...')
r.lookAt(posModel, [0, -1.2, -1])
r.glLoadObj('model.obj', posModel, [1, 1, 1], [0, 0, 0])
r.glFinish('low.bmp')
print('Finalizado')

# DUTCH ANGLE
r.glClear()
print('\nDibujando dutch angle...')
r.lookAt(posModel, [0, 0, -0.3])
r.glLoadObj('model.obj', posModel, [1, 1, 1], [0, 0, 10])
r.glFinish('dutch.bmp')
print('Finalizado')