import matplotlib.pyplot as plt
import numpy as np
import random

def main():
	
	# Simulation parameters
	Nx                     = 400    # resolution x-dir
	Ny                     = 200    # resolution y-dir
	rho0                   = 100    # average density
	tau                    = 0.6    # collision timescale
	Nt                     = 40000   # number of timesteps
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Lattice speeds / weights
	NL = 9
	idxs = np.arange(NL)
	cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
	cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
	weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
	
	# Initial Conditions
	F = np.ones((Ny,Nx,NL)) #* rho0 / NL
	np.random.seed(42)
	# F += 0.01*np.random.randn(Ny,Nx,NL)
	X, Y = np.meshgrid(range(Nx), range(Ny))
	# F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
	F[:,:,3] = 2.3
	rho = np.sum(F,2)
	for i in idxs:
		F[:,:,i] *= rho0 / rho
	
	# Cylinder boundary
	radius = Ny / 4
	X, Y = np.meshgrid(range(Nx), range(Ny))
	cylinder = (X - Nx/4)**2 + (Y - Ny/2)**2 < (radius)**2

	# # Save random timesteps
	# random_timesteps = np.random.choice(range(1500, 3000), 500, replace=False)
	# random_timesteps.sort()
	# np.savetxt('random_timesteps.txt', random_timesteps, fmt='%d')

	# Randomly placed target circle
	def generate_target_position(rad):
		angle = random.uniform(0, 2 * np.pi)
		length = random.uniform(0, 4 * rad)
		center_x = (5 * (2 * rad)) + (Nx / 4) + (length * np.cos(angle))
		center_y = (2.05 * (2 * rad) + (Ny / 2)) + (length * np.sin(angle))
		return center_x, center_y
    
	target_center_x, target_center_y = generate_target_position(radius)
	target_radius = radius / 3
	target_area = (X - int(target_center_x))**2 + (Y - int(target_center_y))**2 < (int(target_radius))**2

	random_area = (X - ((5 * (2 * radius)) + (Nx / 4)))**2 + (Y - ((2.05 * (2 * radius)) + (Ny / 2)))**2 < (4 * radius)**2
	random_area_2 = (X - ((5 * (2 * radius)) + (Nx / 4)))**2 + (Y - ((-2.05 * (2 * radius) + (Ny / 2))))**2 < (4 * radius)**2

	saved_states = []
	# Prep figure
	fig = plt.figure(figsize=(4,2), dpi=80)

	def generate_swimmer_position(rad):
		angle = random.uniform(0, 2 * np.pi)
		length = random.uniform(0, 4 * rad)
		center_x = (5 * (2 * rad)) + (Nx / 4) + (length * np.cos(angle))
		center_y = (-2.05 * (2 * rad) + (Ny / 2)) + (length * np.sin(angle))
		return 300, 50
	
    # Point object initialization
	point_trail = []
	# point_pos = np.array([1.25*Nx/4, Ny/4], dtype=float)  # Use dtype=float for continuous position updates
	point_pos = np.array(generate_swimmer_position(radius), dtype=float)
	point_vel = np.array([0,0], dtype=float)  # Initial velocity
	arrow_length = 100
	
	# Simulation Main Loop
	for it in range(Nt):
		print(it)

		# F[:, -1, [6,7,8]] = F[:, -2, [6,7,8]] # right boundary
		# F[:, 0, [2,3,4]] = F[:, 1, [2,3,4]]
		
		# Drift
		for i, cx, cy in zip(idxs, cxs, cys):
			F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
			F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
		
		# Set reflective boundaries
		bndryF = F[cylinder,:]
		bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]
	
		# Calculate fluid variables
		rho = np.sum(F,2)
		ux  = np.sum(F*cxs,2) / rho
		uy  = np.sum(F*cys,2) / rho
		
		# Apply Collision
		Feq = np.zeros(F.shape)
		for i, cx, cy, w in zip(idxs, cxs, cys, weights):
			Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
		
		F += -(1.0/tau) * (F - Feq)
		
		# Apply boundary 
		F[cylinder,:] = bndryF
		
		# sample_pos_x = int(np.clip(point_pos[0], 0, Nx-1))
		# sample_pos_y = int(np.clip(point_pos[1], 0, Ny-1))
		# local_fluid_vel = np.array([ux[sample_pos_y, sample_pos_x], uy[sample_pos_y, sample_pos_x]])
		# print(ux[0,1], uy[0,1])
		# exit()
		# overall_vel = point_vel + local_fluid_vel
		# point_pos += overall_vel
		# point_trail.append(point_pos.copy())

		# if it in random_timesteps:
		# 	saved_states.append(F.copy())
		
		# plot in real time - color 1/2 particles blue, other half red
		if (plotRealTime and (it % 100) == 0) or (it == Nt-1):
			plt.cla()
			ux[cylinder] = 0
			uy[cylinder] = 0
			# print(ux.shape, uy.shape)
			vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
			vorticity[cylinder] = np.nan
			vorticity = np.ma.array(vorticity, mask=cylinder)
			# plot velocity
			plt.imshow(np.sqrt(ux**2 + uy**2), cmap='viridis')
			# plt.quiver(X[::10,::10], Y[::10,::10], ux[::10,::10], uy[::10,::10], color='black')
			# plt.imshow(vorticity, cmap='bwr')
			plt.imshow(~cylinder, cmap='gray', alpha=0.3)
			plt.contour(cylinder, levels=[0.5], colors='black', linewidths=1)
			# plt.contour(target_area, levels=[0], colors='black', linewidths=1)
			# plt.contour(random_area, levels=[0], colors='red', linewidths=1)
			# plt.contour(random_area_2, levels=[0], colors='green', linewidths=1)
			plt.clim(-.1, .1)
			# for trail_pos in point_trail:
			# 	plt.plot(trail_pos[0], trail_pos[1], 'o', color='black', markersize=1)
			# plt.plot(point_pos[0], point_pos[1], 'o', color='black', markersize=3)
			# plt.arrow(point_pos[0], point_pos[1], arrow_length * point_vel[0], arrow_length * point_vel[1], color='black', head_width=1)
			ax = plt.gca()
			ax.invert_yaxis()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.set_aspect('equal')	
			plt.pause(0.00001)
			
	# Save figure
	plt.savefig('latticeboltzmann.png',dpi=240)
	plt.show()

	# np.save('saved_states.npy', saved_states)
	    
	return 0

if __name__== "__main__":
  main()