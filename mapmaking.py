import numpy as np

def sim_sky(shape=(500,500), res=1, alpha=2, src_dens=1e-4, src_size=1, src_amp=50):
	"""Simulate a toy model of the sky, consisting of a power law "cmb" and a few
	point sources."""
	# Find the frequency corresponding to each position in
	# 2d fourier space.
	ly = np.fft.fftfreq(shape[-2], res)
	lx = np.fft.fftfreq(shape[-1], res)
	l  = (ly[:,None]**2 + lx[None,:]**2)**0.5
	# Generate a random realization with a power law spectrum
	rand = np.fft.fft2(np.random.standard_normal(shape))
	spec = np.maximum(l,l[0,1])**-alpha
	fcmb = rand * spec**0.5
	sky  = np.fft.ifft2(fcmb).real
	# Add some point sources too. This way of doing it is slow but simple
	pos  = np.mgrid[:shape[-2],:shape[-1]]
	nsrc = int(np.product(shape)*res**2 * src_dens)
	for i in range(nsrc):
		psrc = np.random.uniform(0,1,2)*shape*res
		r2   = np.sum((pos-psrc[:,None,None])**2,0)
		amp  = np.random.exponential()*src_amp
		sky += np.exp(-0.5*np.sum((pos-psrc[:,None,None])**2,0)/src_size**2)*src_amp
	return sky

def sim_dataset(map, num_data=2, dt=1, fknee=0.1, alpha=3, sigma=2):
	"""Simulate a dataset consisting of num_data scans across the sky.
	Returns a list of Data objects, each of which contains the tod,
	the pointing and the noise spectrum."""
	res = []
	for i in range(num_data):
		point = sim_pointing(map, i % 2)
		noise_spec = sim_noise_spec(point.shape[-1], dt=dt, fknee=fknee, alpha=alpha, sigma=sigma)
		tod = sim_tod(map, point, noise_spec)
		res.append(Data(tod, point, noise_spec))
	return res

class Data:
	def __init__(self, tod, point, noise_spec):
		self.tod   = tod
		self.point = point
		self.noise_spec = noise_spec

def solve_plain(dataset, shape):
	"""Solve the simplified mapmaking equation Ax=b,
	where A = P'P and b = P'd, e.g. ignoring noise
	properties such as correlations."""
	rhs  = np.zeros(shape)
	hits = np.zeros(shape)
	for data in dataset:
		rhs  += PT(data.tod,     data.point, shape)
		hits += PT(data.tod*0+1, data.point, shape)
	return rhs/hits

def solve_full(dataset, shape, niter=100, verbose=True):
	"""Solve the full map-making equation
	Ax=b, where A = P'N"P and b = P'N"d."""
	# Set up our A matrix. We don't compute
	# explicitly because it's too big. Instead,
	# we define it as a function that can be applied
	# to a vector x. We will then use Conjugate Gradients
	# to invert it.
	def A(x):
		# x is 1d because the conjugate gradient solver works
		# on 1d arrays. So start by expanding it to 2d.
		x   = x.reshape(shape)
		res = x*0
		for data in dataset:
			tod  = P(x, data.point)
			tod  = mul_inv_noise(tod, data.noise_spec)
			res += PT(tod, data.point, shape)
		return res.reshape(-1)
	# Build our right-hand side b
	b = np.zeros(shape)
	for data in dataset:
		tod = mul_inv_noise(data.tod, data.noise_spec)
		b  += PT(tod, data.point, shape)
	# And solve
	cg = CG(A, b.reshape(-1))
	while cg.i < niter:
		cg.step()
		if verbose: print "%4d %15.7e" % (cg.i, cg.err)
	return cg.x.reshape(shape)

def sim_noise_spec(nsamp, dt=1, fknee=0.1, alpha=3, sigma=2):
	"""Build a simple atmosphere + white noise model, and return it
	as a power spectrum."""
	freq   = np.abs(np.fft.fftfreq(nsamp, dt))
	return (1+(np.maximum(freq,freq[1])/fknee)**-alpha)*sigma**2

def sim_pointing(map, dir=0):
	"""Simulate a telescope scanning across the given map. The scanning pattern is
	as simple as possible: The samples hit the center of each pixel, and we
	scan rowwise (dir=0) or columnwise (dir=1)."""
	# The pointing is an [{y,x},nsamp] array of pixel positions
	# The einsum stuff is just to swap the second and third axis
	# of pixmap, which contains the pixel coordinates of each pixel.
	pixmap = np.mgrid[:map.shape[-2],:map.shape[-1]]
	if dir == 0: point = pixmap.reshape(2,-1)
	else:        point = np.einsum("iyx->ixy",pixmap).reshape(2,-1)
	return point

def sim_tod(map, point, noise_spec):
	"""Simulate a noisy TOD using the model d = Pm + n"""
	tod    = P(map, point)
	rand   = np.fft.fft(np.random.standard_normal(tod.shape[-1]))
	fnoise = rand * noise_spec**0.5
	tod   += np.fft.ifft(fnoise).real
	return tod

def P(map, point):
	"""Pointing matrix: Project map to tod"""
	point = np.round(point).astype(int)
	return map[point[0],point[1]]

def PT(tod, point, shape):
	"""Transpose pointing matrix."""
	point = np.round(point).astype(int)
	point_flat = np.ravel_multi_index(point, shape[-2:])
	map = np.bincount(point_flat, tod, minlength=shape[-2]*shape[-1])
	map = map.reshape(shape[-2:])
	return map

def mul_inv_noise(tod, noise_spec):
	"""Multiply by the inverse noise matrix. We assume that the noise
	is stationary, which means that it can be represented by a simple
	power spectrum noise_spec. This function is used to apply inverse
	variance weighting to the data."""
	ftod  = np.fft.fft(tod)
	ftod /= noise_spec
	return np.fft.ifft(ftod).real

def default_M(x):     return np.copy(x)
def default_dot(a,b): return a.dot(np.conj(b))
class CG:
	"""A simple Preconditioned Conjugate gradients solver. Solves
	the equation system Ax=b."""
	def __init__(self, A, b, x0=None, M=default_M, dot=default_dot):
		"""Initialize a solver for the system Ax=b, with a starting guess of x0 (0
		if not provided). Vectors b and x0 must provide addition and multiplication,
		as well as the .copy() method, such as provided by numpy arrays. The
		preconditioner is given by M. A and M must be functors acting on vectors
		and returning vectors. The dot product may be manually specified using the
		dot argument. This is useful for MPI-parallelization, for example."""
		# Init parameters
		self.A   = A
		self.b   = b
		self.M   = M
		self.dot = dot
		if x0 is None:
			self.x = b*0
			self.r = b
		else:
			self.x   = x0.copy()
			self.r   = b-self.A(self.x)
		# Internal work variables
		n = b.size
		self.z   = self.M(self.r)
		self.rz  = self.dot(self.r, self.z)
		self.rz0 = float(self.rz)
		self.p   = self.z
		self.err = np.inf
		self.d   = 4
		self.arz = []
		self.i   = 0
	def step(self):
		"""Take a single step in the iteration. Results in .x, .i
		and .err being updated. To solve the system, call step() in
		a loop until you are satisfied with the accuracy. The result
		can then be read off from .x."""
		Ap = self.A(self.p)
		alpha = self.rz/self.dot(self.p, Ap)
		self.x += alpha*self.p
		self.r -= alpha*Ap
		self.z = self.M(self.r)
		next_rz = self.dot(self.r, self.z)
		self.err = next_rz/self.rz0
		beta = next_rz/self.rz
		self.rz = next_rz
		self.p = self.z + beta*self.p
		self.arz.append(self.rz*alpha)
		self.i += 1
