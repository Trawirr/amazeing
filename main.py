import amazeing

# my_maze = amazeing.Maze(3 , 3, .3, .5)
# my_maze.load_maze([[1,0,1], [0,1,1], [1,0,1], [0,0,0]], [[1,1,1], [0,0,0], [0,0,0], [1,1,1]], [0,1])
# my_maze.print()
# path, maze = my_maze.bfs()
# print("Solution: ", path, maze)
# my_maze.draw_maze(path, maze)

my_maze = amazeing.Maze(31, 31, .45, .6)
my_maze.generate_maze()
while not my_maze.is_solvable():
    my_maze.generate_maze()
path = my_maze.bfs()
print(path)
my_maze.draw_maze(path)