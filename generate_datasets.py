#!/usr/bin/python3
import argparse
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.integrate import odeint
from typing import Dict, Callable


def write_csv(states, file_name: str):
    if os.path.exists(file_name) and not force:
        print(f'[ERROR] {file_name} already exists. Please remove/rename it or provide -f/--force to overwrite it.')
        exit(1)
    elif os.path.exists(file_name) and force:
        print(f'[INFO], overwriting dataset in {file_name} with generated data.')
    columns = []
    for i, _ in enumerate(np.transpose(states)):
        columns.append('x_{}'.format(i))

    df = pd.DataFrame(states, columns=columns)
    df.to_csv(file_name, index=False)


def hyperroessler(state: Tuple[float, float, float, float], t, a: float = 0.25, b: float = 3, c: float = 0.5, d: float = 0.05):
    x, y, z, w = state
    x_dot = -(y + z)
    y_dot = x + a * y + w
    z_dot = b + x * z
    w_dot = -c * z + d * w
    return x_dot, y_dot, z_dot, w_dot


def do_hyperroessler(dt: float, lle: float, limit: float):
    global force
    initial_state = (-10, -14, 0.3, 29)
    t = np.arange(0.0, limit, dt)
    states = odeint(hyperroessler, initial_state, t)

    file_name = f'data/hyperroessler_{dt}_{lle}.csv'
    write_csv(states, file_name)

    return states[:, 0], states[:, 1], states[:, 2], states[:, 3]


def roessler(state: Tuple[float, float, float], t, a: float = 0.2, b: float = 0.2, c: float = 5.7):
    x, y, z = state
    x_dot = -(y + z)
    y_dot = x + a * y
    z_dot = b + z * (x - c)
    return x_dot, y_dot, z_dot


def do_roessler(dt: float, lle: float, limit: float):
    global force
    # Set initial values
    initial_state = (1., 1., 1.)
    t = np.arange(0.0, limit, dt)
    states = odeint(roessler, initial_state, t)

    file_name = f'data/roessler_{dt}_{lle}.csv'
    write_csv(states, file_name)

    return states[:, 0], states[:, 1], states[:, 2]


def lorenz(state, t, r: float = 28.0, s: float = 10.0, b: float = 8.0 / 3.0) -> (float, float, float):
    x, y, z = state
    x_dot = s * (y - x)
    y_dot = x * (r - z) - y
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


def do_lorenz(dt: float, lle: float, limit: float):
    global force
    # Set initial values
    initial_state = (1., 1., 0.0)
    t = np.arange(0., limit, dt)
    states = odeint(lorenz, initial_state, t)

    file_name = f'data/lorenz_{dt}_{lle}.csv'
    write_csv(states, file_name)

    return states[:, 0], states[:, 1], states[:, 2]


def lorenz96(x, t, F: int = 8, N: int = 40):
    states = np.zeros(N)
    states[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
    states[1] = (x[2] - x[N-1]) * x[0] - x[1]
    states[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
    for i in range(2, N-1):
        states[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
    states = states + F

    return states


def do_lorenz96(dt: float, lle: float, limit: float, F: int = 8, N: int = 40):
    global force
    x0 = np.array(
        [4.576779242071500331e-01, 8.057981139137008197e-01, 5.936302860531461612e-01, 7.022680033563672986e-01,
         2.364087735667341761e-02, 9.711322082657990462e-01, 8.099764387303913793e-01, 2.071705633191478491e-03,
         1.788836026311699801e-01, 1.721220138213540585e-01, 2.873043948142445236e-01, 1.302805486633077381e-01,
         4.671808838038193912e-04, 5.656743312129933754e-01, 1.204007647876925713e-01, 3.125920859533409812e-01,
         2.581453561037583277e-01, 4.266410574655790100e-01, 1.299498965154927133e-01, 4.738537931648419965e-01,
         6.823623947807900825e-01, 9.225964064693840117e-01, 3.701157288876431029e-01, 9.971315730804696242e-01,
         4.246168425746921216e-01, 7.283140665770646560e-01, 7.401168277866753131e-01, 1.763626475987889464e-02,
         5.268669108539577595e-01, 4.462878606016699168e-01, 4.058271303692850829e-02, 9.034491973809515297e-01,
         3.052061756269204285e-01, 3.377472694377342544e-01, 2.308801963144646585e-03, 1.149766519797130737e-01,
         4.601118144981631852e-01, 1.115270826568280915e-01, 8.045623642991480695e-01, 8.969576958707504710e-01])
    t = np.arange(0.0, limit, dt)
    states = odeint(lorenz96, x0, t)

    file_name = f'data/lorenz96_{dt}_{lle}.csv'
    write_csv(states, file_name)

    return states[:, 0], states[:, 1], states[:, 2]


def thomas(state, t, b: float = 0.1):
    x, y, z = state
    x_dot = np.sin(y) - b * x
    y_dot = np.sin(z) - b * y
    z_dot = np.sin(x) - b * z

    return x_dot, y_dot, z_dot


def do_thomas(dt: float, lle: float, limit: int):
    global force
    initial_state = (0.0, 1.0, 0.0)
    t = np.arange(0., limit, dt)
    states = odeint(thomas, initial_state, t)[1000:]

    file_name = f'data/thomas_{dt}_{lle}.csv'
    write_csv(states, file_name)

    return states[:, 0], states[:, 1], states[:, 2]


def do_mackeyglass(dt: float, lle: float, limit: float):
    global force
    beta = 0.2
    gamma = 0.1
    tau = 17
    n = 10
    np.random.seed(0)
    initialization = np.random.rand(tau)
    data = np.zeros(int(limit))
    for i in range(tau):
        data[i] = initialization[i]

    for i in range(tau + 1, int(limit)):
        data[i] = data[i - 1] + dt * (beta * (data[i - 1 - tau] / (1 + np.power(data[i - 1 - tau], n))) - gamma * data[i - 1])

    np.savetxt(f'data/mackeyglass_{dt}_{lle}.csv', data[100:])


def plot(xs, ys, zs, caption, save_to=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(xs, ys, zs, lw=0.5)
    ## for thomas b=0.32899
    #ax.set_xticks(np.arange(2.20, 2.36, 0.05))
    #ax.set_xlim(2.225, 2.35)
    #ax.set_yticks(np.arange(2.20, 2.36, 0.05))
    #ax.set_ylim(2.225, 2.35)
    #ax.set_zticks(np.arange(2.20, 2.36, 0.05))
    #ax.set_zlim(2.225, 2.35)
    ## for thomas b=0.1
    ax.set_xticks(np.arange(-4, 5, 2))
    ax.set_xlim(-5.5, 5.5)
    ax.set_yticks(np.arange(-4, 5, 2))
    ax.set_ylim(-5.5, 5.5)
    ax.set_zticks(np.arange(-4, 5, 2))
    ax.set_zlim(-5.5, 5.5)
    #plt.axis('off')
    #ax.set_xlabel("X Axis")
    #ax.set_ylabel("Y Axis")
    #ax.set_zlabel("Z Axis")
    #ax.set_title(caption)

    #plt.show()
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to generate different chaotic time series data sets.')
    parser.add_argument('-f', '--force', dest='force', action='store_true', default=False,
                        help='use the force to overwrite already existing datasets with the same name as the generated ones.')
    args = parser.parse_args()
    force = args.force

    systems: Dict[str, Tuple[float, float, Callable]] = {'roessler': (0.12, 0.069, do_roessler),
                                          'lorenz': (0.01, 0.905, do_lorenz),
                                          'lorenz96': (0.05, 1.67, do_lorenz96),
                                          'thomas': (0.1, 0.055, do_thomas),
                                          'mackeyglass': (1.0, 0.006, do_mackeyglass),
                                          'hyperroessler': (0.1, 0.14, do_hyperroessler)}

    for system, (dt, lle, fn) in systems.items():
        input_steps = 150.0
        samples = 10000
        output_steps = int(np.ceil(1.0 / lle / dt))
        safety_factor = 5.0
        max_limit = int(samples * (input_steps + output_steps) * safety_factor * dt)

        print(f'Generating data from {system} with an LLE of {lle} and a dt of {dt}:')
        print(f'> ... {int((input_steps + output_steps) * samples * safety_factor)} states')
        print(f'> ... {samples} samples with a safety_factor of {safety_factor}')
        print(f'> ... {input_steps} input_steps')
        print(f'> ... {output_steps} output_steps')
        _ = fn(dt=dt, lle=lle, limit=max_limit)
        print(50 * '-')
