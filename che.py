# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

import matplotlib.pyplot as plt

show_animation = True  # 启动动画显示的选项


class AStarPlanner:

    def __init__(self, grid, resolution, rr):
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = grid.shape[1], grid.shape[0]
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(grid)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        start_node = self.Node(self.calc_xy_index(sx, self.min_x), self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x), self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0], current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        for x in range(-int(self.rr / self.resolution), int(self.rr / self.resolution) + 1):
            for y in range(-int(self.rr / self.resolution), int(self.rr / self.resolution) + 1):
                if node.x + x < 0 or node.x + x >= self.x_width or node.y + y < 0 or node.y + y >= self.y_width:
                    return False
                if self.obstacle_map[node.x + x][node.y + y]:
                    return False

        return True

    def calc_obstacle_map(self, grid):
        self.y_width = grid.shape[0]
        self.x_width = grid.shape[1]
        self.obstacle_map = np.zeros((self.x_width, self.y_width), dtype=bool)
        for ix in range(self.x_width):
            for iy in range(self.y_width):
                if grid[iy, ix] != 0:
                    self.obstacle_map[ix][iy] = True

    @staticmethod
    def get_motion_model():
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    # 地图生成
    grid = np.zeros((61, 61))
    for i in range(61):
        for j in range(61):
            if i == 0 or i == 60 or j == 0 or j == 60:
                grid[i, j] = 1

    # 起点终点设定
    sx, sy = 10.0, 10.0
    gx, gy = 50.0, 50.0

    resolution = 1.0
    rr = 1.0

    a_star = AStarPlanner(grid, resolution, rr)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    plt.plot(rx, ry, "-g")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
