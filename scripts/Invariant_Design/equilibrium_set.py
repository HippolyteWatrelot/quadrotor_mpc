import numpy as np
import yaml
import codac as co
from vibes import vibes
from sympy import *


_mass = 1.478
mass_offset = 0.05

I = np.array([[0.0115202,         0,         0],
              [        0, 0.0115457,         0],
              [        0,         0, 0.0218256]])
path = '/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/'


X0 = co.IntervalVector([

    [-2, 2],  # d_xq
    [-2, 2],  # d_y1
    [-2, 2],  # d_z
    [-np.pi/4, np.pi/4],  # roll
    [-np.pi, np.pi],  # rolld
    [-np.pi / 4, np.pi / 4],  # pitch
    [-np.pi, np.pi],  # pitchd
    [-2, 2],  # yawd
    [-2, 2],  # prev_uwb_x
    [-2, 2],  # prev_uwb_y

    [-2, 2],  # ux
    [-2, 2],  # uy
    [-2, 2],  # uz
    [-2, 2],  # uwz

    [I[0, 0] - I[0, 0]*mass_offset/_mass, I[0, 0] + I[0, 0]*mass_offset/_mass],  #I1
    [I[1, 1] - I[1, 1]*mass_offset/_mass, I[1, 1] + I[1, 1]*mass_offset/_mass],  #I2
    [I[2, 2] - I[2, 2]*mass_offset/_mass, I[2, 2] + I[2, 2]*mass_offset/_mass],  #I3
    [_mass-mass_offset, _mass+mass_offset]   #mass

])

with open(path + "scripts/yaml/Linear_Coefficients/INTERVAL_reduced_partial_derivatives_values.yaml", "r") as f1:
    d_config = yaml.safe_load(f1)

print(d_config)

variables = ['d_xq', 'd_yq', 'd_z', 'roll', 'rolld', 'pitch', 'pitchd', 'yawd', 'prev_uwx_b', 'prev_uwy_b']
vec = Matrix([Symbol('d_xq'), Symbol('d_yq'), Symbol('d_z'), Symbol('roll'), Symbol('rolld'), Symbol('pitch'), Symbol('pitchd'), Symbol('yawd'), Symbol('prev_uwx_b'), Symbol('prev_uwy_b')])


def row_forward(v):
    row = Matrix([d_config[v][elt] for elt in variables])
    form = row.T * vec
    print(form[0])
    return str(form[0])


vars_ = "d_xq,d_yq,d_z,roll,rolld,pitch,pitchd,yawd,prev_uwx_b,prev_uwy_b,ux,uy,uz,uwz,I1,I2,I3,mass"

expr1  = row_forward("d2_xq").replace("**", "^")
f1 = co.Function(vars_, expr1)
expr2  = row_forward("d2_yq").replace("**", "^")
f2 = co.Function(vars_, expr2)
expr3  = row_forward("d2_z").replace("**", "^")
f3 = co.Function(vars_, expr3)
expr4  = row_forward("rolld").replace("**", "^")
f4 = co.Function(vars_, expr4)
expr5  = row_forward("rolldd").replace("**", "^")
f5 = co.Function(vars_, expr5)
expr6  = row_forward("pitchd").replace("**", "^")
f6 = co.Function(vars_, expr6)
expr7  = row_forward("pitchdd").replace("**", "^")
f7 = co.Function(vars_, expr7)
expr8  = row_forward("yawdd").replace("**", "^")
f8 = co.Function(vars_, expr8)
expr9  = row_forward("d_prev_uwx_b").replace("**", "^")
f9 = co.Function(vars_, expr9)
expr10  = row_forward("d_prev_uwy_b").replace("**", "^")
f10 = co.Function(vars_, expr10)

S1 = co.SepFunction(f1, [0, oo])
S2 = co.SepFunction(f2, [0, oo])
S3 = co.SepFunction(f3, [0, oo])
S4 = co.SepFunction(f4, [0, oo])
S5 = co.SepFunction(f5, [0, oo])
S6 = co.SepFunction(f6, [0, oo])
S7 = co.SepFunction(f7, [0, oo])
S8 = co.SepFunction(f8, [0, oo])
S9 = co.SepFunction(f9, [0, oo])
S10 = co.SepFunction(f10, [0, oo])

S = S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10


eps = 0.1  # précision

# Projection pour affichage : (x,y) = dimensions 0 et 1
all_boxes = co.SIVIA(X0, S, eps)

colors = ["orange", "red", "green"]
dump_boxes = {"orange": [], "red": [], "green": []}
for i, boxes in enumerate(all_boxes.values()):
    for box in boxes:
        b = [it for it in box]
        dump_boxes[colors[i]].append(b)

with open(path + "quadrotor_equilibrium_states/IT1.npz", 'wb') as f:
    np.savez_compressed(f, array1=np.array(dump_boxes['orange']), array2=np.array(dump_boxes['green']),
                        array3=np.array(dump_boxes['red']))


"""
ib.pySIVIA(X0, S, eps,
        proj=[0,1],
        **{
            "color in": "black[red]",
            "color out": "blue[cyan]",
            "color maybe": "yellow[white]"
        })
"""
