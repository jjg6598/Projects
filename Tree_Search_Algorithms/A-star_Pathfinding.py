#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pygame
import math
from queue import PriorityQueue


# In[2]:


WIDTH = 600
WIN = pygame.display.set_mode((WIDTH, WIDTH))

pygame.display.set_caption('A* Path Finding Algorithm')


# In[3]:


#Colors
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)
white = (255, 255, 255)
black = (0, 0, 0)
purple = (128, 0, 128)
orange = (255, 165, 0)
grey = (128, 128, 128)
turquoise = (64, 224, 208)


# In[4]:


#NODES

class Node():
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row*width
        self.y = col*width
        self.color = white
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
        
    def get_pos(self):
        return self.row, self.col
    
    def is_closed(self):
        return self.color == red
    
    def is_open(self):
        return self.color == green
        
    def is_barrier(self):
        return self.color == black
        
    def is_start(self):
        return self.color == orange
        
    def is_end(self):
        return self.color == turquoise
        
    def reset(self):
        self.color = white
        
    def make_closed(self):
        self.color = red
        
    def make_open(self):
        self.color = green
        
    def make_barrier(self):
        self.color = black
        
    def make_start(self):
        self.color = orange
        
    def make_end(self):
        self.color = turquoise
        
    def make_path(self):
        self.color = purple
        
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
        
    def update_neighbors(self, grid):
        self.neighbors = []
        
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  #Down
            self.neighbors.append(grid[self.row + 1][self.col])
            
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  #Up
            self.neighbors.append(grid[self.row - 1][self.col])
            
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  #Right
            self.neighbors.append(grid[self.row][self.col + 1])
            
        if self.row > 0 and not grid[self.row][self.col - 1].is_barrier():  #Left
            self.neighbors.append(grid[self.row][self.col - 1])
    
    def __lt__(self, other):
        return False


# In[5]:


#HEURISTIC FUNCTION

def h(p1, p2):
    
    x1, y1 = p1
    x2, y2 = p2
    
    return abs(x1 - x2) + abs(y1 - y2)


# In[6]:


def reconstruct_path(came_from, current, draw):
    
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()
    


# In[7]:


#ALGORITHM

def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    
    g_score = {node: float('inf') for row in grid for node in row}
    g_score[start] = 0 
    
    f_score = {node: float('inf') for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())
    
    open_set_hash = {start}
    
    while not open_set.empty():
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
    
        current = open_set.get()[2]
        open_set_hash.remove(current)
        
        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
                
        draw()
        
        if current != start:
            current.make_closed()
                
    return False


# In[8]:


#MAKE GRID OF NODES

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
            
    return grid


# In[9]:


#DRAW LINES OF GRID

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, grey, (0, i*gap), (width, i*gap))
        for j in range(rows):
            pygame.draw.line(win, grey, (j*gap, 0), (j*gap, width))
            


# In[10]:


#DRAW NODES AND LINES

def draw(win, grid, rows, width):
    win.fill(white)
    
    for row in grid:
        for node in row:
            node.draw(win)
            
    draw_grid(win, rows, width)
    pygame.display.update()


# In[11]:


#RETURN MOUSE CLICK POSITION

def get_click_position(pos, rows, width):
    gap = width // rows
    
    y, x = pos
    
    row = y // gap
    col = x // gap
    
    return row, col


# In[12]:


def main(win, width, rows):
    ROWS = rows
    grid = make_grid(ROWS, width)
    
    start = None
    end = None
    
    run = True
    
    while run:
        draw(win, grid, ROWS, width)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_click_position(pos, ROWS, width)
                node = grid[row][col]

                if not start and node != end: #Create start point
                    start = node
                    start.make_start()

                elif not end and node != start: #Create end point
                    end = node
                    end.make_end()

                elif node != start and node != end: #Make barriers
                    node.make_barrier()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_click_position(pos, ROWS, width)
                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                if node == end: 
                    end = None

            if event.type == pygame.KEYDOWN:
                
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                            
                    algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
            
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
            
    pygame.quit()


# In[13]:


main(WIN, WIDTH, 50)

