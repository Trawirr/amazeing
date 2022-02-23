import numpy as np
import matplotlib.pyplot as plt

class Maze:
    def __init__(self, rows, cols, prob_rows, prob_cols):
        self.rows = rows
        self.cols = cols
        self.prob_rows = prob_rows
        self.prob_cols = prob_cols
        self.walls_row = None
        self.walls_col = None
        self.start = None

    # loading maze with given walls
    def load_maze(self, walls_row, walls_col, start):
        self.walls_row = walls_row
        self.walls_col = walls_col
        self.start = start

    # generating random maze
    # if make_better is True generator tries to avoid tiles surrounded by 4 walls
    def generate_maze(self, make_better=True):
        self.walls_row = [[1 if np.random.rand() < self.prob_rows else 0 for x in range(self.cols)] for y in
                          range(self.rows + 1)]
        self.walls_col = [[1 if np.random.rand() < self.prob_cols else 0 for x in range(self.rows)] for y in
                          range(self.cols + 1)]
        self.start = [0, self.cols // 2]
        self.walls_row[self.start[0]][self.start[1]] = 0
        self.walls_row[self.rows] = [0 for x in range(self.cols)]
        self.make_maze_better()

    # printing lists of walls
    def print(self):
        print(self.walls_row, '\n', self.walls_col, '\n', self.start)

    # checking if there is a wall between two tiles
    def is_wall_between(self, start, dest):
        delta = [dest[i] - start[i] for i in range(2)]

        # horizontal
        if delta == [0, -1]:
            return self.walls_col[start[1]][start[0]]
        elif delta == [0, 1]:
            return self.walls_col[start[1] + 1][start[0]]
        # vertical
        elif delta == [-1, 0]:
            return self.walls_row[start[0]][start[1]]
        elif delta == [1, 0]:
            return self.walls_row[start[0] + 1][start[1]]
        else:
            return None

    # returning all possible adjacent tiles
    def get_adjacent_tiles(self, start):
        delta = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        neighbour_tiles = [[start[0] + d[0], start[1] + d[1]] for d in delta]
        neighbour_tiles = [t for t in neighbour_tiles if 0 <= t[0] < self.rows and 0 <= t[1] < self.cols]
        neighbour_tiles = [t for t in neighbour_tiles if not self.is_wall_between(start, t)]
        return neighbour_tiles

    # returning all unvisited tiles from given list
    def get_unvisited_tiles(self, tiles, maze):
        unvisited_tiles = [t for t in tiles if maze[t[0]][t[1]] == 0]
        return unvisited_tiles

    # solving the maze using BFS
    def bfs(self):
        queue = [self.start+[1]]
        maze = [[0 for c in range(self.cols)] for r in range(self.rows)]
        while len(queue) > 0:
            # get the first tile in the queue list
            tile = queue.pop(0)
            # if the tile has been already visited - skip it
            if maze[tile[0]][tile[1]] > 0:
                continue
            else:
                maze[tile[0]][tile[1]] = tile[2]
            # check if the maze is solved
            if tile[0] == self.rows - 1:
                return self.get_path(tile, maze), maze
            adjacent_tiles = self.get_unvisited_tiles(self.get_adjacent_tiles(tile), maze)
            # add available adjacent tiles to the end of the queue list
            for t in adjacent_tiles:
                queue.append(t+[tile[2]+1])
        return [], maze

    # fix tiles surrounded by 4 walls
    def make_maze_better(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if not self.get_adjacent_tiles([i, j]):
                    delta = np.random.randint(2)
                    # 50% for deleting horizontal wall
                    if np.random.rand()<.5:
                        self.walls_row[i+delta][j] = 0
                    # 50% for deleting vertical wall
                    else:
                        self.walls_col[j+delta][i] = 0

    def check_if_better(self):
        print()
        for i in range(self.rows):
            for j in range(self.cols):
                if not self.get_adjacent_tiles([i, j]):
                    print(i,j,'no adjacent tiles')

    # checking if maze is solvable
    def is_solvable(self):
        last_tile = self.bfs()[0]
        if not last_tile:
            return False
        return True

    # returning path of solved maze
    def get_path(self, end, maze):
        end = [end[0], end[1]]
        path = [end]
        while maze[path[0][0]][path[0][1]] != 1:
            for tile in self.get_adjacent_tiles(path[0]):
                if maze[tile[0]][tile[1]] == maze[path[0][0]][path[0][1]] - 1:
                    path.insert(0, tile)
                    break
        return path

    # drawing maze
    def draw_maze(self, path):
        image = np.zeros((4*self.rows+1, 4*self.cols+1, 3), np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = (255, 255, 255)

        # drawing horizontal walls
        for i in range(self.rows+1):
            for j in range(self.cols):
                if self.walls_row[i][j] == 1:
                    index = [4*i, 4*j]
                    for x in range(5):
                        image[index[0]][index[1]+x] = (0, 0, 0)

        # drawing vertical walls
        for i in range(self.cols+1):
            for j in range(self.rows):
                if self.walls_col[i][j] == 1:
                    index = [4*j, 4*i]
                    for x in range(5):
                        image[index[0]+x][index[1]] = (0, 0, 0)
        path = [[2+tile[0]*4, 2+tile[1]*4] for tile in path]
        path.insert(0, [0, 2+self.start[1]*4])
        path.append([self.rows*4, path[len(path)-1][1]])

        for x in range(len(path)-1):
            for i in range(min(path[x][0], path[x+1][0]), max(path[x][0], path[x+1][0]) + 1, 1):
                for j in range(min(path[x][1], path[x+1][1]), max(path[x][1], path[x+1][1]) + 1, 1):
                    image[i][j] = (0, 255, 0)

        plt.imshow(image)
        plt.show()