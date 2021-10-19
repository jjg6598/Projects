#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


board_test = [
    [0, 0, 0, 0, 8, 6, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 3, 0],
    [0, 0, 8, 1, 0, 3, 0, 0, 6],
    [3, 0, 0, 0, 0, 4, 0, 9, 7],
    [4, 0, 0, 0, 5, 0, 0, 0, 1],
    [2, 9, 0, 6, 0, 0, 0, 0, 4],
    [9, 0, 0, 2, 0, 1, 6, 0, 0],
    [0, 5, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 5, 7, 0, 0, 0, 0]
]


# In[3]:


np_board_test = np.array(board_test)


# In[4]:


def print_board(board):
    
    for i in range(len(board)):
    
        block_size = int(len(board)**(1/2))
    
        if i % block_size == 0 and i != 0:
            print('- - - - - - - - - - - -')
        
        for j in range(len(board[0])):
            if j % block_size == 0 and j != 0:
                print(' | ', end='')
                
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + ' ', end='')
        


# In[5]:


#print_board(board_test)


# In[6]:


def find_blank(board):
    
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                return [i, j] #row, col
    
    return None
    


# In[7]:


def valid(board, number, position):
    
    #Check Row
    row_valid = number not in board[position[0]]
    
    if row_valid is False:
        return False
    
    #Check Column 
    col_valid = number not in board[:,position[1]]
    
    if col_valid is False:
        return False
    
    #Check Box
    box_co = [i*3 for i in [position[0] // 3, position[1] // 3]]
    box = board[box_co[0]:box_co[0]+3, box_co[1]:box_co[1]+3]
    
    box_valid = number not in box
    
    if box_valid is False:
        return False
    
    #All Valid
    return True
    


# In[8]:


def solve(board):
    
    blank = find_blank(board)
    if not blank:
        return True
    
    else:
        row, col = blank
    
    for i in range(1,10):
        if valid(board, i, [row, col]):
            board[row][col] = i
            
            if solve(board):
                return True
            
            board[row][col] = 0
    
    return False


# In[9]:


print_board(np_board_test)
solve(np_board_test)
print('                ')
print_board(np_board_test)

