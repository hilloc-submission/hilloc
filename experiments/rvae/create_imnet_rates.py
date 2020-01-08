import re
import pickle
import numpy as np
import ast
from matplotlib import pyplot as plt

with open('image_locations.pickle', 'rb') as f:
    all_im_locations = pickle.load(f)

all_imbpds = {}
for seed in range(1, 31):
    if seed in [9, 15, 16, 17, 18, 19, 20]:
        continue
    with open('seed{}'.format(seed), 'r') as f:
        lines = f.readlines()

    im_locations = None
    if seed > 20:
        # need to read the image_locations from file
        im_locations = re.findall(r'locations:\n(.*)\nUsing', ''.join(lines))[0]
        im_locations = ast.literal_eval(im_locations)
        im_locations = [int(im_loc) for im_loc in im_locations]

    data = []
    previous_iter = None
    for line in lines:
        bpd = re.findall('bpd: (.{4})', line)
        iter = re.findall('Encoded (.*)\/', line)
        if len(bpd):
            if len(iter):
                iter_ = int(iter[0])
            else:
                iter_ = previous_iter
            data.append((iter_, float(bpd[0])))
        if len(iter):
            previous_iter = int(iter[0])


    def create_im_bpds(iter_data, im_locations):
        im_bpds = [iter_data[0][1]] * 5
        im_locations = im_locations[5:]
        iters = [i[0] for i in iter_data]
        for i, im_location in enumerate(im_locations):
            # find nearest iter
            nearest_iter_index = np.searchsorted(iters, im_location)
            if nearest_iter_index >= len(iter_data):
                break
            im_bpds.append(iter_data[nearest_iter_index][1])
        return im_bpds

    if im_locations is None:
        im_locations = all_im_locations[seed]
    all_imbpds[seed] = create_im_bpds(data, im_locations)

all_data = np.zeros((len(all_imbpds.keys()), 400))
for i, values in enumerate(all_imbpds.values()):
    all_data[i, :] = np.pad(values, pad_width=(0, 400-len(values)), mode='edge')

means = np.mean(all_data, axis=0)
print(means)
plt.plot(means)
plt.show()


# def smooth(scalars, weight): # Weight between 0 and 1
#     last = scalars[0]  # First value in the plot (first timestep)
#     smoothed = list()
#     for point in scalars:
#         smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
#         smoothed.append(smoothed_val)                        # Save it
#         last = smoothed_val                                  # Anchor the last smoothed value
#
#     return smoothed
#
# smoothed = smooth(means, 0.999)
#
# plt.plot(smoothed)
# plt.show()

