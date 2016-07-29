import numpy as np, mapmaking
from matplotlib import pyplot
print "Generating input sky"
sky      = mapmaking.sim_sky()
pyplot.matshow(sky,vmin=-20,vmax=20)
pyplot.title("Input sky")
pyplot.show()
print "Generating noisy TOD simulations"
dataset  = mapmaking.sim_dataset(sky)
for data in dataset:
	pyplot.plot(data.tod)
pyplot.show()
print "Solving for map while ignoring noise correlations"
map_plain = mapmaking.solve_plain(dataset, sky.shape)
pyplot.matshow(map_plain)
pyplot.title("Solution assuming white noise")
pyplot.show()
print "Solving for map while taking noise correlations into account"
map_full  = mapmaking.solve_full(dataset, sky.shape)
pyplot.matshow(map_full,vmin=-20,vmax=20)
pyplot.title("Full solution")
pyplot.show()
print "Solving for solution without crosslinking"
for i in range(2):
	map_single = mapmaking.solve_full(dataset[i:i+1], sky.shape)
	pyplot.matshow(map_single, vmin=-20,vmax=20)
	if i == 0: pyplot.title("Horizontal scans only")
	else:      pyplot.title("Vertical scans only")
pyplot.show()

