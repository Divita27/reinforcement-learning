import pygame
import sys
import random
import math
import numpy as np
from gym import spaces
import gym
# import gymnasium as gym
# from gymnasium import spaces

class VortexENV(gym.Env):
    def __init__(self, width=400, height=200, radius=40, agent = 'velocity'):
        
        super(VortexENV, self).__init__()
        pygame.init()
        
        # Constants
        self.WIDTH = width
        self.HEIGHT = height
        self.RADIUS = radius
        self.CIRCLE_X = 3 * self.WIDTH // 4
        self.BORDER_WIDTH = 5
        self.agent_type = agent

        # gym init
        # self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32) # FIXME: UserWarning coming
        # self.observation_space = spaces.Box(low=np.array([-self.WIDTH, -self.HEIGHT, -5, -5]), high=np.array([self.WIDTH, self.HEIGHT, 5, 5]), shape=(4,), dtype=np.float32)

        # Colors
        self.ORANGE = (255, 165, 0)
        self.GREEN = (0, 255, 0)
        self.BACKGROUND_COLOR = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)

        # screen setup
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        self.screen.fill(self.BACKGROUND_COLOR)
        pygame.display.set_caption("Von Kármán Vortex Street Simulation")

        # Flow field parameters
        self.Nx = self.WIDTH
        self.Ny = self.HEIGHT
        self.rho0 = 100
        self.tau = 0.6
        self.NL = 9
        self.idxs = np.arange(self.NL)
        self.cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
        self.cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
        self.weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

        # Defining the elements of the simulation
        X, Y = np.meshgrid(range(self.Nx), range(self.Ny))
        self.cylinder = (X - self.Nx // 4) ** 2 + (Y - self.Ny // 2) ** 2 < (self.Ny // 4) ** 2
        self.orange_circle_y = self.HEIGHT // 4
        self.green_circle_y = 3 * self.HEIGHT // 4

        self.reset()

    def reset(self, start_time="random"):
        self.init_simulation(start_time)
        self.n_steps = 0
        self.v = 0.1
        self.x, self.y = self.spawn_boat()
        self.target_x, self.target_y = self.spawn_target()
        self.angle_rad = random.uniform(-math.pi, math.pi)
        self.v_x = np.sin(self.angle_rad) * self.v
        self.v_y = np.cos(self.angle_rad) * self.v
        self.update_flow_field()
        
        rel_goal_x = self.target_x - self.x
        rel_goal_y = self.target_y - self.y

        bg_vel_x = self.ux[int(self.y), int(self.x)]
        bg_vel_y = self.uy[int(self.y), int(self.x)]
        
        vorticity_matrix = (np.roll(self.ux, -1, axis=0) - np.roll(self.ux, 1, axis=0)) - (np.roll(self.uy, -1, axis=1) - np.roll(self.uy, 1, axis=1))
        vorticity = vorticity_matrix[int(self.y), int(self.x)]

        if self.agent_type == 'velocity':
            observation = (rel_goal_x/self.WIDTH, rel_goal_y/self.HEIGHT, bg_vel_x, bg_vel_y)
        elif self.agent_type == 'vorticity':
            observation = (rel_goal_x/self.WIDTH, rel_goal_y/self.HEIGHT, vorticity, vorticity)
        elif self.agent_type == 'flow_blind':
            observation = (rel_goal_x/self.WIDTH, rel_goal_y/self.HEIGHT)
        else:
            print('invalid agent type')

        return observation
    
    def init_simulation(self, start_time):
        if start_time == "random":
            timesteps = np.loadtxt('random_timesteps.txt', dtype=int)
            random_timestep = np.random.choice(timesteps)
            # print(f"Randomly selected timestep: {random_timestep}")

            # Loading the presaved flow states
            flow_states = np.load('saved_states.npy')

            index = np.where(timesteps == random_timestep)[0][0]
            self.F = np.array(flow_states[index])
            # print(self.F)

        else:
            # Initialize flow field
            self.F = np.ones((self.Ny, self.Nx, self.NL))  # Distribution function
            # self.F += 0.01 * np.random.randn(self.Ny, self.Nx, self.NL)
            self.F[:, :, 3] = 2.3  # Adjust initial distribution
            self.rho = np.sum(self.F, 2)
            for i in self.idxs:
                self.F[:, :, i] *= self.rho0 / self.rho
    
    def step(self, action):
        # print(action)
        # angle_radians = math.radians(action)
        self.angle_rad = action * math.pi
        new_v_x = self.v * math.cos(self.angle_rad)
        new_v_y = self.v * math.sin(self.angle_rad)
        
        vel_x = self.ux[int(self.y), int(self.x)] + new_v_x
        vel_y = self.uy[int(self.y), int(self.x)] + new_v_y
        
        # Getting the nackground voticity for previous timestep
        previous_vorticity_matrix = (np.roll(self.ux, -1, axis=0) - np.roll(self.ux, 1, axis=0)) - (np.roll(self.uy, -1, axis=1) - np.roll(self.uy, 1, axis=1))
        previous_vorticity = previous_vorticity_matrix[int(self.y), int(self.x)]
        
        # Store previous position of point
        previous_position = np.array([self.x, self.y])
        self.x += vel_x
        self.y += vel_y

        # Update the flow
        self.update_flow_field()  
        current_position = np.array([self.x, self.y])
        target_position = np.array([self.target_x, self.target_y])
        
        # Relative distance for reward
        distance_previous = np.linalg.norm(previous_position - target_position)
        distance_current = np.linalg.norm(current_position - target_position)
        
        rel_goal_x = self.target_x - self.x
        rel_goal_y = self.target_y - self.y
        
        # getting the background velocity at the new position
        bg_vel_x = self.ux[min(int(self.y), 199), min(int(self.x), 399)]
        bg_vel_y = self.uy[min(int(self.y), 199), min(int(self.x), 399)]
        
        # Getting the nackground voticity for current timestep
        current_vorticity_matrix = (np.roll(self.ux, -1, axis=0) - np.roll(self.ux, 1, axis=0)) - (np.roll(self.uy, -1, axis=1) - np.roll(self.uy, 1, axis=1))
        current_vorticity = current_vorticity_matrix[min(int(self.y), 199), min(int(self.x), 399)]
        
        # returing observation space based on agent type
        if self.agent_type == "velocity":
            observation = (rel_goal_x/self.WIDTH, rel_goal_y/self.HEIGHT, bg_vel_x, bg_vel_y)
        elif self.agent_type == "vorticity":
            observation = (rel_goal_x/self.WIDTH, rel_goal_y/self.HEIGHT, current_vorticity, previous_vorticity)
        elif self.agent_type == "flow_blind":
            observation = (rel_goal_x/self.WIDTH, rel_goal_y/self.HEIGHT)
        else:
            print("invalid agent type")
            exit()
            
        # Termination check
        done, target_reached = self.check_terminal_state()
        
        # Reward
        reward = -1 + 10 * ((distance_previous - distance_current) / self.v) + \
            (10000 if target_reached else 0)
        
        self.n_steps += 1
        
        # self.render()

        return observation, reward, done, {}

    def check_terminal_state(self):
        # Check if the point is out of bounds
        if self.x <= 0 or self.x >= self.WIDTH or self.y <= 0 or self.y >= self.HEIGHT:
            return True, False
        # Check collision with the cylinder
        if (self.x - self.Nx / 4)**2 + (self.y - self.Ny / 2)**2 <= (self.Ny / 4)**2:
            return True, False
        # Check if the point reaches the traget region
        if (self.x - self.target_x)**2 + (self.y - self.target_y)**2 <= (self.RADIUS / 3)**2:
            return True, True
    
        return False, False

    def spawn_boat(self):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, self.RADIUS)
        offset_x = distance * math.cos(angle)
        offset_y = distance * math.sin(angle)
        x = self.CIRCLE_X + offset_x
        y = self.green_circle_y + offset_y
        return x, y
    
    def spawn_target(self):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, 2*self.RADIUS/3)
        offset_x = distance * math.cos(angle)
        offset_y = distance * math.sin(angle)
        x = self.CIRCLE_X + offset_x
        y = self.orange_circle_y + offset_y
        return x, y

    def update_flow_field(self):
        for i, cx, cy in zip(self.idxs, self.cxs, self.cys):
            self.F[:, :, i] = np.roll(self.F[:, :, i], cx, axis=1)
            self.F[:, :, i] = np.roll(self.F[:, :, i], cy, axis=0)
        bndryF = self.F[self.cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        rho = np.sum(self.F, 2)
        self.ux = np.sum(self.F * self.cxs, 2) / rho
        self.uy = np.sum(self.F * self.cys, 2) / rho
        Feq = np.zeros(self.F.shape)
        for i, cx, cy, w in zip(self.idxs, self.cxs, self.cys, self.weights):
            Feq[:, :, i] = rho * w * (1 + 3 * (cx * self.ux + cy * self.uy) + 9 * (cx * self.ux + cy * self.uy)**2 / 2 - 3 * (self.ux**2 + self.uy**2) / 2)
        self.F += -(1.0 / self.tau) * (self.F - Feq)
        self.F[self.cylinder, :] = bndryF

    def draw_quiver(self):
        grid_size = 10
        for row in range(0, self.Ny, grid_size):
            for col in range(0, self.Nx, grid_size):
                center_x, center_y = col + grid_size // 2, row + grid_size // 2
                avg_ux = np.mean(self.ux[row:row + grid_size, col:col + grid_size])
                avg_uy = np.mean(self.uy[row:row + grid_size, col:col + grid_size])
                if avg_ux != 0 or avg_uy != 0:
                    vector_length = math.sqrt(avg_ux**2 + avg_uy**2)
                    scaled_ux = avg_ux / vector_length * 10
                    scaled_uy = avg_uy / vector_length * 10
                    end_x = center_x + scaled_ux
                    end_y = center_y + scaled_uy
                    pygame.draw.line(self.screen, self.BLUE, (center_x, center_y), (end_x, end_y), 2)

                    # Calculate arrowhead points for each line
                    arrow_angle = math.atan2(scaled_uy, scaled_ux)
                    arrow_length = 5
                    left_arrow_x = end_x - arrow_length * math.cos(arrow_angle - math.pi / 6)
                    left_arrow_y = end_y - arrow_length * math.sin(arrow_angle - math.pi / 6)
                    right_arrow_x = end_x - arrow_length * math.cos(arrow_angle + math.pi / 6)
                    right_arrow_y = end_y - arrow_length * math.sin(arrow_angle + math.pi / 6)

                    # Draw arrowheads
                    pygame.draw.line(self.screen, self.BLUE, (end_x, end_y), (left_arrow_x, left_arrow_y), 2)
                    pygame.draw.line(self.screen, self.BLUE, (end_x, end_y), (right_arrow_x, right_arrow_y), 2)

    def render(self, angle_rad=None):
        self.screen.fill(self.BACKGROUND_COLOR)
        self.draw_quiver()
        pygame.draw.circle(self.screen, self.BLACK, (self.Nx // 4, self.Ny // 2), self.Ny // 4 + 10)
        pygame.draw.circle(self.screen, self.GREEN, (self.CIRCLE_X, self.green_circle_y), self.RADIUS, self.BORDER_WIDTH) # start-spawncircle
        pygame.draw.circle(self.screen, self.ORANGE, (self.CIRCLE_X, self.orange_circle_y), self.RADIUS, self.BORDER_WIDTH) # target-spawn circle
        pygame.draw.circle(self.screen, self.BLACK, (int(self.x), int(self.y)), 5) # boat
        pygame.draw.circle(self.screen, self.RED, (int(self.target_x), int(self.target_y)), self.RADIUS/3) # targe-spawn circle
        # draw the heading of the boat
        if angle_rad is not None:
            heading_x = self.x + 10 * math.cos(angle_rad)
            heading_y = self.y + 10 * math.sin(angle_rad)
            pygame.draw.line(self.screen, self.RED, (int(self.x), int(self.y)), (int(heading_x), int(heading_y)), 3)
        else:
            heading_x = self.x + 10 * math.cos(self.angle_rad)
            heading_y = self.y + 10 * math.sin(self.angle_rad)
            pygame.draw.line(self.screen, self.RED, (int(self.x), int(self.y)), (int(heading_x), int(heading_y)), 3)
        pygame.display.update()

    def baseline_test(self):
        running = True
        while running:
            
            self.screen.fill(self.BACKGROUND_COLOR)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update_flow_field()
            self.draw_quiver()
            int_x, int_y = int(self.x), int(self.y)
            if 0 <= int_x < self.WIDTH and 0 <= int_y < self.HEIGHT:
                velocity_x = self.ux[int_y, int_x] + self.v_x
                velocity_y = self.uy[int_y, int_y] + self.v_y
                self.x += velocity_x
                self.y += velocity_y

            pygame.draw.circle(self.screen, self.BLACK, (self.Nx // 4, self.Ny // 2), self.Ny // 4 + 10)
            pygame.draw.circle(self.screen, self.GREEN, (self.CIRCLE_X, self.green_circle_y), self.RADIUS, self.BORDER_WIDTH) # start-spawncircle
            pygame.draw.circle(self.screen, self.ORANGE, (self.CIRCLE_X, self.orange_circle_y), self.RADIUS, self.BORDER_WIDTH) # target-spawn circle
            pygame.draw.circle(self.screen, self.BLACK, (int(self.x), int(self.y)), 5) # boat
            pygame.draw.circle(self.screen, self.RED, (int(self.target_x), int(self.target_y)), self.RADIUS/3) # targe-spawn circle
            pygame.display.update()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    env = VortexENV()
    # env.baseline_test()
    for i in range(2):
        # observation = env.reset(start_time="zero")
        observation = env.reset()
        done = False
        while not done:
            step = env.n_steps
            action = env.action_space.sample()[0]
            # action = 270
            print(f"Action: {action}")
            observation, reward, done, info = env.step(action)
            if done:
                print(f"Episode finished after {env.n_steps} steps")
                env.reset()
            if step % 10 == 0:
                env.render() # FIXME: Load it once in a while
    env.close()