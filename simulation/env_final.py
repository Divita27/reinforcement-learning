import matplotlib.pyplot as plt
import numpy as np
import random

class LatticeBoltzmannSimulation:
    def __init__(self, Nx=400, Ny=100, rho0=100, tau=0.6, Nt=4000):
        self.Nx = Nx
        self.Ny = Ny
        self.rho0 = rho0
        self.tau = tau
        self.Nt = Nt
        self.NL = 9
        self.idxs = np.arange(self.NL)
        self.cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
        self.cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
        self.weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
        self.fig = plt.figure(figsize=(4, 2), dpi=80)
        self.radius = Ny / 16
        self.reset()

    def reset(self):
        self.F = np.ones((self.Ny, self.Nx, self.NL))
        self.F[:, :, 3] = 2.3  # Modified speed direction
        self.rho = np.sum(self.F, axis=2)
        for i in self.idxs:
            self.F[:, :, i] *= self.rho0 / self.rho
        self.cylinder = self.init_cylinder()
        self.target_area = self.init_target_area()
        self.point_pos = np.array(self.generate_point_position(), dtype=float)
        self.velocity = 0.09
        self.point_vel = np.array([0, 0], dtype=float)
        self.point_trail = []
        self.timestep = random.randint(900, 3000)
        self.play_simulation(self.timestep - 1)
        return self.get_state()

    def init_cylinder(self):
        X, Y = np.meshgrid(range(self.Nx), range(self.Ny))
        return (X - self.Nx // 4) ** 2 + (Y - self.Ny // 2) ** 2 < (self.radius) ** 2

    def init_target_area(self):
        target_center_x, target_center_y = self.generate_target_position()
        X, Y = np.meshgrid(range(self.Nx), range(self.Ny))
        target_radius = self.radius / 3
        return (X - int(target_center_x)) ** 2 + (Y - int(target_center_y)) ** 2 < (int(target_radius)) ** 2

    def generate_target_position(self):
        angle = random.uniform(0, 2 * np.pi)
        length = random.uniform(0, 4 * self.radius)
        center_x = (5 * (2 * self.radius)) + (self.Nx / 4) + (length * np.cos(angle))
        center_y = (2.05 * (2 * self.radius) + (self.Ny / 2)) + (length * np.sin(angle))
        return center_x, center_y

    def generate_point_position(self):
        angle = random.uniform(0, 2 * np.pi)
        length = random.uniform(0, 4 * self.radius)
        center_x = (5 * (2 * self.radius)) + (self.Nx / 4) + (length * np.cos(angle))
        center_y = (-2.05 * (2 * self.radius) + (self.Ny / 2)) + (length * np.sin(angle))
        return center_x, center_y
    
    def play_simulation(self, num_steps):
        for _ in range(num_steps):
            self.update_velocities()
            self.apply_collision()
            self.apply_boundaries()

    def step(self, action):
        # Update velocities based on action (if applicable)
        self.update_velocities()
        self.apply_collision()
        self.apply_boundaries()
        self.update_position(action)
        self.point_trail.append(self.point_pos.copy())
        return self.get_state(), self.compute_reward(), self.is_done(), {}

    def update_velocities(self):
        for i, cx, cy in zip(self.idxs, self.cxs, self.cys):
            self.F[:, :, i] = np.roll(self.F[:, :, i], cx, axis=1)
            self.F[:, :, i] = np.roll(self.F[:, :, i], cy, axis=0)

    def apply_collision(self):
        rho = np.sum(self.F, 2)
        ux = np.sum(self.F * self.cxs, 2) / rho
        uy = np.sum(self.F * self.cys, 2) / rho
        Feq = np.zeros(self.F.shape)
        for i, cx, cy, w in zip(self.idxs, self.cxs, self.cys, self.weights):
            Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy) ** 2 / 2 - 3 * (ux ** 2 + uy ** 2) / 2)
        self.F += -(1.0 / self.tau) * (self.F - Feq)

    def apply_boundaries(self):
        bndryF = self.F[self.cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        self.F[self.cylinder, :] = bndryF

    def update_position(self):
        sample_pos_x = int(np.clip(self.point_pos[0], 0, self.Nx - 1))
        sample_pos_y = int(np.clip(self.point_pos[1], 0, self.Ny - 1))
        ux = np.sum(self.F * self.cxs, 2) / self.rho
        uy = np.sum(self.F * self.cys, 2) / self.rho
        local_fluid_vel = np.array([ux[sample_pos_y, sample_pos_x], uy[sample_pos_y, sample_pos_x]])
        self.point_pos += self.point_vel + local_fluid_vel

    def get_state(self):
        # Return the current state, potentially the grid of velocities or densities
        return self.F

    def compute_reward(self):
        # Define how reward is computed based on state
        return 0

    def is_done(self):
        # Check if the point has reached the target area
        
        return False

    def render(self):
        plt.cla()
        ux = np.sum(self.F * self.cxs, 2) / self.rho
        uy = np.sum(self.F * self.cys, 2) / self.rho
        ux[self.cylinder] = 0
        uy[self.cylinder] = 0
        vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        vorticity[self.cylinder] = np.nan
        vorticity = np.ma.array(vorticity, mask=self.cylinder)
        plt.imshow(vorticity, cmap='bwr')
        plt.imshow(~self.cylinder, cmap='gray', alpha=0.3)
        plt.contour(self.cylinder, levels=[0.5], colors='black', linewidths=1)
        plt.contour(self.target_area, levels=[0], colors='black', linewidths=1)
        for trail_pos in self.point_trail:
            plt.plot(trail_pos[0], trail_pos[1], 'o', color='black', markersize=1)
        plt.plot(self.point_pos[0], self.point_pos[1], 'o', color='black', markersize=3)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        # plt.pause(0.001)

if __name__ == "__main__":
    simulation = LatticeBoltzmannSimulation()
    for _ in range(100):
        simulation.step(None)
        simulation.render()