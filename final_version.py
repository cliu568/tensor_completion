import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

#size of tensor
n = 200
#CP rank of tensor
r = 4
#number of observations
num_samples = 50000
p = min(float(num_samples/ (n*n*n)), 1)

#Whether initialization is random or computed using initialization step
randominit = True

#Whether the underlying tensor will have correlated components
correlated = False

#Whether observations are exact or noisy
noisy = False
noise_size = 0.1




#which algorithm to run, can be "Matrix Alt Min, Tensor Powering, Subspace Powering, all"
which_alg = "Tensor Powering"

save_file = "tensorpowering+noisy200_4_50000.csv"

#Number of different tensors to run on
num_runs = 100


#Number of iterations per run
num_iter = 400

#min error threshold
threshold = 10**(-13)



#generate random uncorrelated tensor
def gen(n,r):
	coeffs = np.ones(r)
	x_vecs = np.random.normal(0,1,(r,n))
	y_vecs = np.random.normal(0,1,(r,n))
	z_vecs = np.random.normal(0,1,(r,n))
	return (coeffs, x_vecs,y_vecs,z_vecs)




#generate random correlated tensor
def gen_biased(n,r):
	coeffs = np.zeros(r)
	x_vecs = np.zeros((r,n))
	y_vecs = np.zeros((r,n))
	z_vecs = np.zeros((r,n))
	for i in range(r):
		coeffs[i] = 0.5**i
		if(i==0):
			x_vecs[i] = np.sqrt(n) * normalize(np.random.normal(0,1,n))
			y_vecs[i] = np.sqrt(n) * normalize(np.random.normal(0,1,n))
			z_vecs[i] = np.sqrt(n) * normalize(np.random.normal(0,1,n))
		else:
			x_vecs[i] = np.sqrt(n) * normalize(np.random.normal(0,0.5,n) + x_vecs[0])
			y_vecs[i] = np.sqrt(n) * normalize(np.random.normal(0,0.5,n) + y_vecs[0])
			z_vecs[i] = np.sqrt(n) * normalize(np.random.normal(0,0.5,n) + z_vecs[0])
	return (coeffs, x_vecs,y_vecs,z_vecs)







#evaluate tensor given coordinates
def T(i,j,k, coeffs, x_vecs, y_vecs, z_vecs):
	ans = 0
	for a in range(r):
		ans += coeffs[a] * x_vecs[a][i] * y_vecs[a][j] * z_vecs[a][k]
	return ans





#sample observations, a is num_samples
#returns 3 lists of coordinates
def sample(a):
	samples = np.random.choice(n**3, a, replace=False)
	x_coords = samples%n
	y_coords = (((samples - x_coords)/n)%n).astype(int)
	z_coords = (((samples - n*y_coords - x_coords)/(n*n))%n).astype(int)
	return (x_coords, y_coords, z_coords)




#Given samples and tensor T, construct dictionary x_dict that stores the observations
def fill(x_coords, y_coords, z_coords, coeffs, x_vecs, y_vecs, z_vecs, x_dict):
	num_samples = x_coords.size
	for i in range(num_samples):
		#For x_dict coordinates are in order x,y,z
		if(x_coords[i] in x_dict.keys()):
			if(y_coords[i] in x_dict[x_coords[i]].keys()):
				if(z_coords[i] in x_dict[x_coords[i]][y_coords[i]].keys()):
					pass
				else:
					x_dict[x_coords[i]][y_coords[i]][z_coords[i]] = T(x_coords[i] , y_coords[i] , z_coords[i], coeffs, x_vecs, y_vecs, z_vecs)
			else:
				x_dict[x_coords[i]][y_coords[i]] = {}
				x_dict[x_coords[i]][y_coords[i]][z_coords[i]] = T(x_coords[i] , y_coords[i] , z_coords[i], coeffs, x_vecs, y_vecs, z_vecs)
		else:
			x_dict[x_coords[i]] = {}
			x_dict[x_coords[i]][y_coords[i]] = {}
			x_dict[x_coords[i]][y_coords[i]][z_coords[i]] = T(x_coords[i] , y_coords[i] , z_coords[i], coeffs, x_vecs, y_vecs, z_vecs)





#normalize vector
def normalize(v):
	u = v/np.linalg.norm(v)
	return u


#given rxn array, output orthonormal basis
def orthonormalize(V):
	a = len(V)
	b = len(V[0])
	for i in range(a):
		for j in range(i):
			V[i] = V[i] - np.dot(V[i],V[j])*V[j]
		V[i] = normalize(V[i])
	return V

#implicit sparse matrix multiplication where M is stored as a dictionary
def mult(M,v):
	u = np.zeros(n)
	for coord1 in M.keys():
		for coord2 in M[coord1].keys():
			u[coord1] += M[coord1][coord2] * v[coord2]
	return u


#Compute initial subspace estimates
def initialization(x_dict):
	M_x = np.zeros((n,n))
	for x in x_dict.keys():
		for y in x_dict[x].keys(): 
			for z1 in x_dict[x][y].keys():
				for z2 in x_dict[x][y].keys():
					val = x_dict[x][y][z1] * x_dict[x][y][z2]
					if(z1 == z2):
						val = val/p
					else:
						val = val/(p*p)
					M_x[z1][z2] += val
	svd = TruncatedSVD(n_components=r)
	svd.fit(M_x)
	return(svd.components_)

	

#Unfold and perform matrix completion via altmin
def matrix_altmin(V_x, V_yz):
	#Solve for next iteration of x
	lsq_solution = []
	for i in range(n):
		features = []
		target = []
		for y_coord in x_dict[i].keys():
			for z_coord in x_dict[i][y_coord].keys():
				features.append(V_yz[n*y_coord + z_coord])
				target.append(x_dict[i][y_coord][z_coord])


		features = np.array(features)
		target = np.array(target)

		reg = LinearRegression(fit_intercept = False).fit(features, target)
		lsq_solution.append(reg.coef_)

	x_solution = np.array(lsq_solution)

	#Solve for next iteration of yz
	lsq_solution2 = []
	for i in range(n):
		for j in range(n):
			features = []
			target = []
			if i in y_dict.keys() and j in y_dict[i].keys():
				for x_coord in y_dict[i][j].keys():
					features.append(x_solution[x_coord])
					target.append(y_dict[i][j][x_coord])
				features = np.array(features)
				target = np.array(target)

				reg = LinearRegression(fit_intercept = False).fit(features, target)
				lsq_solution2.append(reg.coef_)
			else:
				lsq_solution2.append(np.zeros(r))

	newV_x = x_solution
	newV_yz =np.array(lsq_solution2)
	return(newV_x, newV_yz)


#Normalized MSE for unfolded matrix completion
def eval_error_matrix(V_x,V_yz):
	#take random sample of entries to speed up evaluation
	num_trials = 1000
	total_error = 0
	total_norm = 0
	for i in range(num_trials):
		x = np.random.randint(n)
		y = np.random.randint(n)
		z = np.random.randint(n)

		
		prediction = 0
		for j in range(r):
			prediction += V_x[x][j] * V_yz[n * y + z][j]

		true_val = T(x,y,z, coeffs, x_vecs,y_vecs, z_vecs)

		total_norm += np.square(true_val)
		total_error += np.square(prediction - true_val)
	return np.sqrt(total_error/total_norm)

			
					


#altmin for naive tensor powering
def power_altmin(V_x, V_y, V_z , x_dict):

	lsq_solution = []
	for i in range(n):
		features = []
		target = []
		for y_coord in x_dict[i].keys():
			for z_coord in x_dict[i][y_coord].keys():

				#subsample to speed up and get "unstuck"
				check = np.random.randint(2)
				if(check == 0):	
					features.append(np.multiply(V_y[y_coord], V_z[z_coord]))
					target.append(x_dict[i][y_coord][z_coord])

		features = np.array(features)
		target = np.array(target)

		reg = LinearRegression(fit_intercept = False).fit(features, target)
		lsq_solution.append(reg.coef_)


	lsq_solution = np.array(lsq_solution)
	return(lsq_solution)


#Normalized MSE for naive tensor powering
def eval_error_direct(V_x,V_y,V_z, x_dict):

	num_trials = 1000
	total_error = 0
	total_norm = 0
	for i in range(num_trials):
		x = np.random.randint(n)
		y = np.random.randint(n)
		z = np.random.randint(n)

		
		prediction = 0
		for j in range(r):
			prediction += V_x[x][j] * V_y[y][j] * V_z[z][j]

		true_val = T(x,y,z, coeffs, x_vecs,y_vecs, z_vecs)

		total_norm += np.square(true_val)
		total_error += np.square(prediction - true_val)
	return np.sqrt(total_error/total_norm)


#altmin for our algorithm
def subspace_altmin(V_x, V_y, V_z , x_dict):

	lsq_solution = []
	for i in range(n):
		features = []
		target = []
		for y_coord in x_dict[i].keys():
			for z_coord in x_dict[i][y_coord].keys():

				#subsample to speed up and get "unstuck"
				check = np.random.randint(2)
				if(check == 0):	
					features.append(np.tensordot(V_y[y_coord], V_z[z_coord] , axes = 0).flatten())
					target.append(x_dict[i][y_coord][z_coord])

		features = np.array(features)
		target = np.array(target)

		reg = LinearRegression(fit_intercept = False).fit(features, target)
		lsq_solution.append(reg.coef_)


	lsq_solution = np.transpose(np.array(lsq_solution))
	svd = TruncatedSVD(n_components=r)
	svd.fit(lsq_solution)

	return(np.transpose(svd.components_))





#Normalized MSE for our algorithm
def eval_error_subspace(V_x,V_y,V_z, x_dict):
	features = []
	target = []
	#Find coefficients in V_x x V_y x V_z basis
	for x_coord in x_dict.keys():
		for y_coord in x_dict[x_coord].keys():
			for z_coord in x_dict[x_coord][y_coord].keys():

				#speed up by using less entries
				check = np.random.randint(10)
				if(check == 0):
					target.append(x_dict[x_coord][y_coord][z_coord])
					part = np.tensordot(V_x[x_coord], V_y[y_coord], axes = 0).flatten()
					full = np.tensordot(part, V_z[z_coord], axes = 0).flatten()
					features.append(full)

	features = np.array(features)
	target = np.array(target)
	reg = LinearRegression(fit_intercept = False).fit(features, target)
	solution_coeffs = reg.coef_
	#print(reg.score(features, target))
	#print(solution_coeffs)

	#Evaluate RMS error
	num_trials = 1000
	total_error = 0
	total_norm = 0
	for i in range(num_trials):
		x = np.random.randint(n)
		y = np.random.randint(n)
		z = np.random.randint(n)

		part = np.tensordot(V_x[x], V_y[y], axes = 0).flatten()
		feature = np.tensordot(part, V_z[z], axes = 0).flatten()
		prediction = np.dot(feature, solution_coeffs)

		true_val = T(x,y,z, coeffs, x_vecs,y_vecs, z_vecs)

		total_norm += np.square(true_val)
		total_error += np.square(prediction - true_val)
	return np.sqrt(total_error/total_norm)









#Keep track of errors for all runs
all_errors = []




for run in range(num_runs):
	#store error over time for this run
	error = []
	curr_error = 1.0

	#Construct random tensor
	if(correlated):
		coeffs, x_vecs,y_vecs,z_vecs = gen_biased(n,r)
	else:
		coeffs, x_vecs,y_vecs,z_vecs = gen(n,r)
	x_coords,y_coords,z_coords = sample(num_samples)


	#x_dict,y_dict, z_dict each stores all observed entries 
	#x_dict has coordinates in order x,y,z
	#y_dict has coordinates in order y,z,x
	#z_dict has coordinates in order z,x,y

	x_dict = {}
	y_dict = {}
	z_dict = {}
	fill(x_coords, y_coords, z_coords, coeffs, x_vecs, y_vecs, z_vecs, x_dict)
	fill(y_coords, z_coords, x_coords, coeffs, y_vecs, z_vecs, x_vecs, y_dict)
	fill(z_coords, x_coords, y_coords, coeffs, z_vecs, x_vecs, y_vecs, z_dict)


	#Add Noise
	if(noisy):
		for x_coord in x_dict.keys():
			for y_coord in x_dict[x_coord].keys():
				for z_coord in x_dict[x_coord][y_coord].keys():
					x_dict[x_coord][y_coord][z_coord] += np.random.normal(0,noise_size)
					y_dict[y_coord][z_coord][x_coord] += np.random.normal(0,noise_size)
					z_dict[z_coord][x_coord][y_coord] += np.random.normal(0,noise_size)


	#Initialization
	if(randominit):
		V_x = np.random.normal(0,1,(r,n))
		V_y = np.random.normal(0,1,(r,n))
		V_z = np.random.normal(0,1,(r,n))
		V_x = orthonormalize(V_x)
		V_y = orthonormalize(V_y)
		V_z = orthonormalize(V_z)
		V_x = np.transpose(V_x)
		V_y = np.transpose(V_y)
		V_z = np.transpose(V_z)

	else:
		V_x = np.transpose(initialization(y_dict))
		V_y = np.transpose(initialization(z_dict))
		V_z = np.transpose(initialization(x_dict))


	#For unfolding and matrix completion
	V_xmat = np.random.normal(0,1, (r,n))
	V_yzmat = np.random.normal(0,1, (r, n*n))
	V_xmat = orthonormalize(V_xmat)
	V_yzmat = orthonormalize(V_yzmat)
	V_xmat = np.transpose(V_xmat)
	V_yzmat = np.transpose(V_yzmat)


	V_x2 = np.copy(V_x)
	V_y2 = np.copy(V_y)
	V_z2 = np.copy(V_z)
	

	print(n)
	print(r)
	print(num_samples)


	


	#AltMin Steps
	for i in range(num_iter):
		print(i)
		if(which_alg == "Matrix Alt Min" or which_alg == "all"):
			print("Matrix Alt Min")
			V_xmat, V_yzmat = matrix_altmin(V_xmat, V_yzmat)
			curr_error = eval_error_matrix(V_xmat, V_yzmat)
			print(curr_error)
			error.append(curr_error)
			
		if(which_alg == "Tensor Powering" or which_alg == "all"):
			print("Tensor Powering")
			if(curr_error > threshold):
				V_x = power_altmin(V_x,V_y,V_z, x_dict)
				V_y = power_altmin(V_y,V_z,V_x, y_dict)
				V_z = power_altmin(V_z,V_x,V_y, z_dict)
				curr_error = eval_error_direct(V_x,V_y,V_z, x_dict)
			print(curr_error)
			error.append(curr_error)


		if(which_alg == "Subspace Powering" or which_alg == "all"):
			print("Subspace Powering")
			if(curr_error > threshold):
				V_x2 = subspace_altmin(V_x2,V_y2,V_z2, x_dict)
				V_y2 = subspace_altmin(V_y2,V_z2,V_x2, y_dict)
				V_z2 = subspace_altmin(V_z2,V_x2,V_y2, z_dict)
				curr_error = eval_error_subspace(V_x2,V_y2,V_z2, x_dict)
			print(curr_error)
			error.append(curr_error)


	all_errors.append(error)
	to_save = np.transpose(np.array(all_errors))
	avg_errors = np.mean(to_save, axis = 0)
	np.savetxt(save_file, to_save, delimiter=",")






	



