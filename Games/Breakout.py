#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pygame 
from pygame.locals import *

pygame.init()


# In[2]:


#Screen Dimensions
screen_width = 600
screen_height = 600

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Breakout')

#Define Font
font = pygame.font.SysFont('Constantia', 30)

#define colors
background = (234, 218, 184)

#Blocks
brick_red = (242, 85, 96)
brick_green = (86, 174, 87)
brick_blue = (69, 177, 232)

#Paddle Colors
paddle_col = (142, 135, 123)
paddle_outline = (100, 100, 100)

text_color = (78, 81, 139)

#Define game variables
cols = 6
rows = 6
clock = pygame.time.Clock()
fps = 60
live_ball = False
game_over = 0


# In[3]:


#Print Text 

def draw_text(text, font, text_color, x, y):
    img = font.render(text, True, text_color)
    screen.blit(img, (x, y))


# In[4]:


#Brick Class

class Brick():
    
    def __init__(self):
        self.width = screen_width // cols
        self.height = 50
        
    def create_brick(self):
        self.bricks = []
        
        #Create emply list
        brick_individual = []
        for row in range(rows):
            brick_row = []
            
            for col in range(cols):
                brick_x = col * self.width
                brick_y = row * self.height
                rect = pygame.Rect(brick_x, brick_y, self.width, self.height)
                
                if row < 2:
                    strength = 3
                elif row < 4:
                    strength = 2
                elif row < 6:
                    strength = 1
                    
                brick_individual = [rect, strength]
                brick_row.append(brick_individual)
            
            self.bricks.append(brick_row)
            
    def draw_wall(self):
        
        for row in self.bricks:
            
            for brick in row:
                if brick[1] == 3:
                    brick_color = brick_blue
                elif brick[1] == 2:
                    brick_color = brick_green
                elif brick[1] == 1:
                    brick_color = brick_red
                
                pygame.draw.rect(screen, brick_color, brick[0])
                pygame.draw.rect(screen, background, (brick[0]), 2)
                


# In[5]:


#Paddle Class

class Paddle():
    def __init__(self):
        self.reset()
        
    def move(self):
        self.direction = 0
        
        key = pygame.key.get_pressed()
        if key[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
            self.direction = -1
            
        if key[pygame.K_RIGHT] and self.rect.right < screen_width:
            self.rect.x += self.speed
            self.direction = 1
            
    def draw(self):
        pygame.draw.rect(screen, paddle_col, self.rect)
        pygame.draw.rect(screen, paddle_outline, self.rect, 3)
        
    def reset(self):
        self.height = 20 
        self.width = int(screen_width / cols)
        self.x = int((screen_width / 2) - (self.width / 2))
        self.y = screen_height - (self.height * 2)
        self.speed = 10 
        self.rect = Rect(self.x, self.y, self.width, self.height)
        self.direction = 0


# In[6]:


#Ball Class

class game_ball():
    def __init__(self, x, y):
        self.reset(x, y)
        
        
    def draw(self):
        pygame.draw.circle(screen, paddle_col,
                           (self.rect.x + self.ball_radius,
                            self.rect.y + self.ball_radius),
                            self.ball_radius)
        pygame.draw.circle(screen, paddle_outline,
                           (self.rect.x + self.ball_radius,
                            self.rect.y + self.ball_radius),
                            self.ball_radius, 3)
        
    def move(self):
        
        collision_threshold = 5
        
        wall_destroyed = 1
        
        row_count = 0
        for row in wall.bricks:
            
            item_count = 0
            for item in row:
                if self.rect.colliderect(item[0]):
                    if abs(self.rect.bottom - item[0].top) < collision_threshold and self.speed_y > 0:
                        self.speed_y *= -1
                    if abs(self.rect.top - item[0].bottom) < collision_threshold and self.speed_y < 0:
                        self.speed_y *= -1
                    if abs(self.rect.right- item[0].left) < collision_threshold and self.speed_x > 0:
                        self.speed_x *= -1
                    if abs(self.rect.left - item[0].right) < collision_threshold and self.speed_x < 0:
                        self.speed_x *= -1
                    
                    if wall.bricks[row_count][item_count][1] > 1:
                        wall.bricks[row_count][item_count][1] -= 1
                    else:
                        wall.bricks[row_count][item_count][0] = (0, 0, 0, 0)
                     
                if wall.bricks[row_count][item_count][0] != (0, 0, 0, 0):
                    wall_destroyed = 0
                
                item_count += 1
            row_count += 1
        
        if wall_destroyed == 1:
            self.game_over = 1
        
        if self.rect.left < 0 or self.rect.right > screen_width:
            self.speed_x *= -1
            
        if self.rect.top < 0:
            self.speed_y *= -1
        if self.rect.bottom > screen_height:
            self.game_over = -1
        
        if self.rect.colliderect(player_paddle):
            if abs(self.rect.bottom - player_paddle.rect.top) < collision_threshold and self.speed_y > 0:
                self.speed_y *= -1
                self.speed_x += player_paddle.direction
                if self.speed_x > self.speed_max:
                    self.speed_x = self.speed_max
                elif self.speed_x < 0 and self.speed_x < -self.speed_max:
                    self.speed_x = -self.speed_max
            else:
                self.speed_x *= -1
        
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        
        return self.game_over
    
    def reset(self, x, y):
        self.ball_radius = 10
        self.x = x - self.ball_radius
        self.y = y
        self.rect = Rect(self.x, self.y, self.ball_radius*2, self.ball_radius*2)
        self.speed_x = 4
        self.speed_y = -4
        self.speed_max = 5
        self.game_over = 0
         


# In[7]:


#Create Wall
wall = Brick()
wall.create_brick()
    
#Create Paddle   
player_paddle = Paddle()

#Create Ball
ball = game_ball(player_paddle.x + (player_paddle.width // 2),
                 player_paddle.y - player_paddle.height)


# In[8]:


run = True

while run:
    
    clock.tick(fps)
    
    screen.fill(background)
    
    wall.draw_wall()
    player_paddle.draw()
    ball.draw()
    
    if live_ball:
        player_paddle.move()
        game_over = ball.move()
        if game_over != 0:
            live_ball = False
    
    
    if not live_ball:
        if game_over == 0:
            draw_text('CLICK ANYWHERE TO START', font, text_color, 100, screen_height // 2 + 100)
        if game_over == 1:
            draw_text('YOU WON!', font, text_color, 240, screen_height // 2 + 50)
            draw_text('CLICK ANYWHERE TO START', font, text_color, 100, screen_height // 2 + 100)
        if game_over == -1:
            draw_text('YOU LOST!', font, text_color, 240, screen_height // 2 + 50)
            draw_text('CLICK ANYWHERE TO START', font, text_color, 100, screen_height // 2 + 100)
        
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.MOUSEBUTTONDOWN and live_ball == False:
            live_ball = True
            ball.reset(player_paddle.x + (player_paddle.width // 2),
                       player_paddle.y - player_paddle.height)
            player_paddle.reset()
            wall.create_brick()
            
    pygame.display.update()    
    
pygame.quit()

