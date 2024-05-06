import pygame
import sys
import random
import math
import numpy as np

class VortexStreetSimulation:
    def __init__(self, width=400, height=200, radius=30):
        pygame.init()
        
        # Constants
        self.WIDTH = width
        self.HEIGHT = height
        self.RADIUS = radius
        self.CIRCLE_X = 3 * self.WIDTH // 4
        self.BORDER_WIDTH = 5

        # Colors
        self.ORANGE = (255, 165, 0)
        self.GREEN = (0, 255, 0)
        self.BACKGROUND_COLOR = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)

        # Screen setup
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Von Kármán Vortex Street Simulation")

        # Flow field parameters
        self.Nx = self.WIDTH
        self.Ny = self.HEIGHT
        self.rho0 = 500
        self.tau = 0.6
        self.NL = 9
        self.idxs = np.arange(self.NL)
        self.cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
        self.cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
        self.weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

        self.init_simulation()

    def init_simulation(self):
        self.F = np.ones((self.Ny, self.Nx, self.NL))  # Distribution function
        self.F += 0.01 * np.random.randn(self.Ny, self.Nx, self.NL)
        self.F[:, :, 3] = 2.3  # Adjust initial distribution
        self.rho = np.sum(self.F, 2)
        for i in self.idxs:
            self.F[:, :, i] *= self.rho0 / self.rho
        X, Y = np.meshgrid(range(self.Nx), range(self.Ny))
        self.cylinder = (X - self.Nx // 4) ** 2 + (Y - self.Ny // 2) ** 2 < (self.Ny // 4) ** 2
        self.orange_circle_y = self.HEIGHT // 4
        self.green_circle_y = 3 * self.HEIGHT // 4
        self.x, self.y = self.random_point_in_green_circle()
        self.v_x = -0.05
        self.v_y = -0.1

    def random_point_in_green_circle(self):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, self.RADIUS)
        offset_x = distance * math.cos(angle)
        offset_y = distance * math.sin(angle)
        x = self.CIRCLE_X + offset_x
        y = self.green_circle_y + offset_y
        return x, y

    def update_flow_field(self):
        for i, cx, cy in zip(self.idxs, self.cxs, self.cys):
            self.F[:, :, i] = np.roll(self.F[:, :, i], cx, axis=1)
            self.F[:, :, i] = np.roll(self.F[:, :, i], cy, axis=0)
        bndryF = self.F[self.cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        self.F[self.cylinder, :] = bndryF
        rho = np.sum(self.F, 2)
        ux = np.sum(self.F * self.cxs, 2) / rho
        uy = np.sum(self.F * self.cys, 2) / rho
        Feq = np.zeros(self.F.shape)
        for i, cx, cy, w in zip(self.idxs, self.cxs, self.cys, self.weights):
            Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy)**2 / 2 - 3 * (ux**2 + uy**2) / 2)
        self.F += -(1.0 / self.tau) * (self.F - Feq)
        return ux, uy

    def draw_quiver(self, ux, uy):
        grid_size = 10
        for row in range(0, self.Ny, grid_size):
            for col in range(0, self.Nx, grid_size):
                center_x, center_y = col + grid_size // 2, row + grid_size // 2
                avg_ux = np.mean(ux[row:row + grid_size, col:col + grid_size])
                avg_uy = np.mean(uy[row:row + grid_size, col:col + grid_size])
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


    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.screen.fill(self.BACKGROUND_COLOR)
            ux, uy = self.update_flow_field()
            self.draw_quiver(ux, uy)
            int_x, int_y = int(self.x), int(self.y)
            if 0 <= int_x < self.WIDTH and 0 <= int_y < self.HEIGHT:
                velocity_x = ux[int_y, int_x] + self.v_x
                velocity_y = uy[int_y, int_y] + self.v_y
                self.x += velocity_x
                self.y += velocity_y
            pygame.draw.circle(self.screen, self.BLACK, (self.Nx // 4, self.Ny // 2), self.Ny // 4 + 10)
            pygame.draw.circle(self.screen, self.GREEN, (self.CIRCLE_X, self.green_circle_y), self.RADIUS, self.BORDER_WIDTH)
            pygame.draw.circle(self.screen, self.ORANGE, (self.CIRCLE_X, self.orange_circle_y), self.RADIUS, self.BORDER_WIDTH)
            pygame.draw.circle(self.screen, self.BLACK, (int(self.x), int(self.y)), 5)
            pygame.display.update()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    sim = VortexStreetSimulation()
    sim.run()
