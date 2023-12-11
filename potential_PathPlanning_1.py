from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
import copy

# Parameters
KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
AREA_WIDTH = 30.0  # potential area width [m]
# the number of previous positions used to check oscillations
OSCILLATIONS_DETECTION_LENGTH = 3

show_animation = True


def calc_potential_field(goal_x, goal_y, obstacle_x, obstacle_y, reso, rr, start_x, start_y):
    minx = min(min(obstacle_x), start_x, goal_x) - AREA_WIDTH / 2.0
    miny = min(min(obstacle_y), start_y, goal_y) - AREA_WIDTH / 2.0
    maxx = max(max(obstacle_x), start_x, goal_x) + AREA_WIDTH / 2.0
    maxy = max(max(obstacle_y), start_y, goal_y) + AREA_WIDTH / 2.0
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        x = ix * reso + minx

        for iy in range(yw):
            y = iy * reso + miny
            ug = calc_attractive_potential(x, y, goal_x, goal_y)
            uo = calc_repulsive_potential(x, y, obstacle_x, obstacle_y, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny


def calc_attractive_potential(x, y, goal_x, goal_y):
    return 0.5 * KP * np.hypot(x - goal_x, y - goal_y)


def calc_repulsive_potential(x, y, obstacle_x, obstacle_y, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(obstacle_x):
        d = np.hypot(x - obstacle_x[i], y - obstacle_y[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - obstacle_x[minid], y - obstacle_y[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0


def get_motion_model():
    # dx, dy
    # all the 8 neighbouring cells to be checked
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion

def get_wind_model(seed):
    np.random.seed(seed)
    #wind = np.random.normal(0, 0.5, 2)
    wind = np.random.normal(0, 0.75, 2)
    return wind

def oscillations_detection(previous_ids, ix, iy):
    previous_ids.append((ix, iy))

    if (len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH):
        previous_ids.popleft()

    # check if contains any duplicates by copying into a set
    previous_ids_set = set()
    for index in previous_ids:
        if index in previous_ids_set:
            return False
        else:
            previous_ids_set.add(index)
    return False


class ExternalForcePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(5,32))
        self.fc2 = spectral_norm(nn.Linear(32,32))
        self.fc3 = spectral_norm(nn.Linear(32,2))
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


def potential_field_planning(start_x, start_y, goal_x, goal_y, obstacle_x, obstacle_y, reso, rr, model,
                             loss_func, optimizer, isTraining=True):
    # calc potential field
    pmap, minx, miny = calc_potential_field(goal_x, goal_y, obstacle_x, obstacle_y, reso, rr, start_x, start_y)

    # search path
    d = np.hypot(start_x - goal_x, start_y - goal_y)
    ix = round((start_x - minx) / reso)
    iy = round((start_y - miny) / reso)
    gix = round((goal_x - minx) / reso)
    giy = round((goal_y - miny) / reso)

    obs_x = (np.array(copy.deepcopy(obstacle_x)) - minx) / reso + 0.3
    obs_y = (np.array(copy.deepcopy(obstacle_y)) - miny) / reso + 0.3

    if show_animation:
        draw_heatmap(pmap)
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(ix, iy, marker="o", markerfacecolor="orange", markersize=12, markeredgewidth=0)
        plt.plot(gix, giy, "*r", markersize=20)
        plt.scatter(obs_x, obs_y, marker='o', s=200, c='black')
        plt.annotate("GOAL", xy=(gix + 2, giy + 2), color='red')
        plt.annotate("START", xy=(25, 22), color='orange')
        plt.axis(False)

    path_x, path_y = [start_x], [start_y]
    motion = get_motion_model()
    collision = False
    outside = False
    # seed = 26236
    seed = 498411
    # pre_ix = ix
    # pre_iy = iy
    while d >= reso:
        minp = float("inf")
        minix, miniy = -1, -1
        wind = np.clip(get_wind_model(seed), -1, 1)
        seed += 1
        for i, _ in enumerate(motion):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
                p = float("inf")  # outside area
                print("outside potential!")
                outside = True
            else:
                p = pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
        ix = minix
        iy = miniy
        state = np.array([path_x[-1], path_y[-1], minix, miniy, minp])
        np.random.seed(int(seed/2))

        # OMAC
        # state = torch.from_numpy(state).float()

        # OoD-Control
        state = torch.from_numpy(state + 0.05 * np.random.normal(0, 1, state.size)).float()

        predict = model(state)
        # predict = torch.clamp(predict, min=-1.5, max=1.5)
        ix = minix + wind[0]
        iy = miniy + wind[1]
        origin = torch.from_numpy(np.array([minix, miniy])).float()
        now = predict + torch.from_numpy(np.array([ix, iy])).float()
        loss = loss_func(origin, now)
        # print(loss)
        if isTraining == True:
            loss.backward()
            optimizer.step()
        
        # origin
        ix = ix - wind[0]
        iy = iy - wind[1]

        # predict
        predict = predict.detach().numpy()
        # ix -= 0.2 * predict[0]
        # iy -= 0.2 * predict[1]

        xp = ix * reso + minx
        yp = iy * reso + miny
        d = np.hypot(goal_x - xp, goal_y - yp)
        path_x.append(xp)
        path_y.append(yp)

        for idx in range(len(obstacle_x)):
            if np.hypot(obstacle_x[idx] - xp, obstacle_y[idx] - yp) < 2:
                print("Collision at ({},{})!".format(ix, iy))
                plt.plot(ix, iy, marker="x", markersize=7, markerfacecolor="red", markeredgewidth=2,
                         markeredgecolor="red")
                plt.pause(0.10)
                collision = True
                break

        if collision == True or outside == True:
            break

        if show_animation:
            plt.plot(ix, iy, marker=".", markersize=7, markerfacecolor="orange", markeredgewidth=0)
            plt.pause(0.10)

    print("Finished")

    return path_x, path_y


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def main():
    print("potential_field_planning start")

    start_x = 0.0  # start x position [m]
    start_y = 0.0  # start y position [m]
    goal_x = 25.0  # goal x position [m]
    goal_y = 30.0  # goal y position [m]
    grid_size = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]

    obstacle_x = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
          15.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 24.0, 24.0, 24.0, 24.0, 26.0, 27.0, 28.0,
          16.0, 17.0, 18.0, 19.0, 11.0, 24.0, 18.0, 18.0, 18.0, 18.0]  # obstacle x position list [m]
    obstacle_y = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
          22.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 19.0, 18.0, 17.0, 16.0, 25.0, 25.0, 25.0,
          9.5, 9.5, 9.5, 9.5, 20.0, 10.0, 4.0, 3.0, 2.0, 1.0]  # obstacle y position list [m]

    if show_animation:
        plt.grid(False)
        plt.axis("auto")

    # path generation
    
    torch.manual_seed(1234)
    model = ExternalForcePredictor()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_func = nn.MSELoss()
    
    model.train()
    for _ in range(10):
        _, _ = potential_field_planning(
            start_x, start_y, goal_x, goal_y, obstacle_x, obstacle_y, grid_size, robot_radius, model,
            loss_func, optimizer)
        plt.cla()
    
    model.eval()
    _, _ = potential_field_planning(
            start_x, start_y, goal_x, goal_y, obstacle_x, obstacle_y, grid_size, robot_radius, model,
            loss_func, optimizer, isTraining=False)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
