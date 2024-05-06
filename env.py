import pygame
import sys
import random
import math
import numpy as np

pygame.init()

# Consts
WIDTH = 800
HEIGHT = 300
RADIUS = 30 
CIRCLE_X = 3*WIDTH // 4 
BORDER_WIDTH = 5 

# Colors
ORANGE = (255, 165, 0)
GREEN = (0, 255, 0)
BACKGROUND_COLOR = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pygame Environment with Circles and Squares")


orange_circle_y = HEIGHT // 4
green_circle_y = 3 * HEIGHT // 4  

# flow field parameters
Nx = WIDTH
Ny = HEIGHT
rho0 = 100
tau = 0.6
NL = 9
idxs = np.arange(NL)
cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])


F = np.ones((Ny, Nx, NL))  # Distribution function
F += 0.01 * np.random.randn(Ny, Nx, NL)
X, Y = np.meshgrid(range(Nx), range(Ny))
F[:, :, 3] = 2.3  # Adjust initial distribution
rho = np.sum(F, 2)
for i in idxs:
    F[:, :, i] *= rho0 / rho

# Cylinder (bluff body)
cylinder = (X - Nx/4)**2 + (Y - Ny/2)**2 < (Ny/4)**2

# Function to generate a random point inside the green circle
def random_point_in_green_circle():
    angle = random.uniform(0, 2 * math.pi)  
    distance = random.uniform(0, RADIUS) 
    offset_x = distance * math.cos(angle)
    offset_y = distance * math.sin(angle)
    x = CIRCLE_X + offset_x
    y = green_circle_y + offset_y

    return x, y


def update_flow_field():
    global F, rho, ux, uy
    
    # Drift (streaming step)
    for i, cx, cy in zip(idxs, cxs, cys):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
        F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)
    
    # Reflective boundary conditions at the cylinder (bluff body)
    bndryF = F[cylinder, :]
    bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]  # Reverse directions for reflection
    
    # Calculate fluid variables
    rho = np.sum(F, axis=2)
    ux = np.sum(F * cxs, axis=2) / rho
    uy = np.sum(F * cys, axis=2) / rho
    
    # Compute equilibrium distribution function
    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy)**2 / 2 - 3 * (ux**2 + uy**2) / 2)
    
    # Collision step
    F += -(1.0 / tau) * (F - Feq)
    
    # Apply boundary conditions
    F[cylinder, :] = bndryF
    
    # Calculate vorticity for visualization
    vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
    vorticity[cylinder] = np.nan
    
    # Return the updated flow field variables
    return ux, uy, vorticity

def draw_quiver(screen, ux, uy):

    grid_size = 10  
    Ny, Nx = ux.shape
    
    rows = Ny // grid_size
    cols = Nx // grid_size
    
    for row in range(rows):
        for col in range(cols):
            
            start_x = col * grid_size
            start_y = row * grid_size
            
            center_x = start_x + grid_size // 2
            center_y = start_y + grid_size // 2
            
            avg_ux = np.mean(ux[row * grid_size:(row + 1) * grid_size, col * grid_size:(col + 1) * grid_size])
            avg_uy = np.mean(uy[row * grid_size:(row + 1) * grid_size, col * grid_size:(col + 1) * grid_size])
            
            if avg_ux != 0 or avg_uy != 0:
                vector_length = math.sqrt(avg_ux ** 2 + avg_uy ** 2)
                if vector_length != 0:

                    scale_factor = 10
                    scaled_ux = avg_ux / vector_length * scale_factor
                    scaled_uy = avg_uy / vector_length * scale_factor
                    
                    end_x = center_x + scaled_ux
                    end_y = center_y + scaled_uy
                    
                    pygame.draw.line(screen, BLUE, (center_x, center_y), (end_x, end_y), 2)
                    arrow_angle = math.atan2(scaled_uy, scaled_ux)
                    arrow_length = 5
  
                    left_arrow_x = end_x - arrow_length * math.cos(arrow_angle - math.pi / 6)
                    left_arrow_y = end_y - arrow_length * math.sin(arrow_angle - math.pi / 6)
                    right_arrow_x = end_x - arrow_length * math.cos(arrow_angle + math.pi / 6)
                    right_arrow_y = end_y - arrow_length * math.sin(arrow_angle + math.pi / 6)

                    pygame.draw.line(screen, BLUE, (end_x, end_y), (left_arrow_x, left_arrow_y), 2)
                    pygame.draw.line(screen, BLUE, (end_x, end_y), (right_arrow_x, right_arrow_y), 2)

def main():

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Von Kármán Vortex Street Simulation")
    x, y = random_point_in_green_circle()

    running = True  
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        screen.fill(BACKGROUND_COLOR)

        ux, uy, vorticity = update_flow_field() # Update the flow field
        draw_quiver(screen, ux, uy)  # velocity field

        pygame.draw.circle(screen, (0, 0, 0), (Nx // 4, Ny // 2), Ny // 4 + 10)
        pygame.draw.circle(screen, GREEN, (CIRCLE_X, green_circle_y), RADIUS, BORDER_WIDTH) # start circle
        pygame.draw.circle(screen, ORANGE, (CIRCLE_X, orange_circle_y), RADIUS, BORDER_WIDTH) # goal circle
        pygame.draw.circle(screen, BLACK, (int(x), int(y)), 5) # boat point
        pygame.display.update()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()