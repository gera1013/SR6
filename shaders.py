from gl import *
import numpy as np

def gourad(render, **kwargs):
    u, v, w = kwargs['barycentricCoordenates']
    ta, tb, tc = kwargs['textureCoordenates']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['_color_']

    b /= 255
    g /= 255
    r /= 255

    if render.texture:
        tx = ta[0] * u + tb[0] * v + tc[0] * w
        ty = ta[1] * u + tb[1] * v + tc[1] * w
        
        texColor = render.texture.getColor(tx, ty)

        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = [nx, ny, nz]

    intensity = dotProduct(normal, render.light)

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return r, g, b
    else:
        return 0, 0, 0

def toonShader(render, **kwargs):
    u, v, w = kwargs['barycentricCoordenates']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['_color_']

    b /= 255
    g /= 255
    r /= 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = [nx, ny, nz]

    intensity = dotProduct(normal, render.light)

    if intensity < 0.2:
        intensity = 0.2
    elif intensity < 0.4:
        intensity = 0.4
    elif intensity < 0.6:
        intensity = 0.6
    elif intensity < 0.8:
        intensity = 0.8
    elif intensity < 1:
        intensity = 1

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return r, g, b
    else:
        return 0, 0, 0

def thermalVision(render, **kwargs):
    u, v, w = kwargs['barycentricCoordenates']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['_color_']

    b /= 255
    g /= 255
    r /= 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = [nx, ny, nz]

    intensity = dotProduct(normal, render.light)

    if intensity > 0:
        if intensity > 0.97:
            r = 240 / 255
            g = 126 / 255
            b = 125 / 255
        elif intensity > 0.90:
            r = 239 / 255
            g = 86 / 255
            b = 107 / 255
        elif intensity > 0.85:
            r = 236 / 255
            g = 36 / 255
            b = 72 / 255
        elif intensity > 0.81: 
            r = 237 / 255
            g = 41 / 255
            b = 68 / 255
        elif intensity > 0.75:
            r = 239 / 255
            g = 55 / 255
            b = 59 / 255
        elif intensity > 0.68:
            r = 243 / 255
            g = 85 / 255
            b = 54 / 255
        elif intensity > 0.61:
            r = 248 / 255
            g = 128 / 255
            b = 40 / 255
        elif intensity > 0.56:
            r = 252 / 255
            g = 160 / 255
            b = 32 / 255
        elif intensity > 0.5:
            r = 252 / 255
            g = 187 / 255
            b = 22 / 255
        elif intensity > 0.42:
            r = 232 / 255
            g = 211 / 255
            b = 30 / 255
        elif intensity > 0.33:
            r = 158 / 255
            g = 203 / 255
            b = 58 / 255
        elif intensity > 0.26:
            r = 72 / 255
            g = 173 / 255
            b = 73 / 255
        elif intensity > 0.20:
            r = 20 / 255
            g = 155 / 255
            b = 123 / 255
        elif intensity > 0.13:
            r = 27 / 255
            g = 129 / 255
            b = 191 / 255
        elif intensity > 0.08:
            r = 38 / 255
            g = 87 / 255
            b = 171 / 255
        elif intensity > 0.04:
            r = 35 / 255
            g = 60 / 255
            b = 134 / 255
        elif intensity > 0.001:
            r = 36 / 255
            g = 42 / 255
            b = 105 / 255

    if intensity > 0:
        return r, g, b

    else:
        return (20 / 255), (20 / 255), (70 / 255)

def phong(render, **kwargs):
    u, v, w = kwargs['barycentricCoordenates']
    ta, tb, tc = kwargs['textureCoordenates']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['_color_']

    b /= 255
    g /= 255
    r /= 255

    if render.texture:
        tx = ta[0] * u + tb[0] * v + tc[0] * w
        ty = ta[1] * u + tb[1] * v + tc[1] * w

        texColor = render.texture.getColor(tx, ty)

        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = [nx, ny, nz]

    intensity = np.dot(normal, render.light)

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return r, g, b
    else:
        return 0,0,0