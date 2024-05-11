import torch
from preprocessing import prepare
from utilits import calculate_pixels, calculate_weights

data_dir='/home/iot/Desktop/RAKIB_MI/1_multi_parametric_MRI/3D_prostate_cancer_seg/Dataset/Dataset/'

data_in = prepare(data_dir, cache=False)

# Assuming you have already defined your prepare function and obtained train_loader and test_loader
train_loader, test_loader = prepare(data_dir, cache=False)


device = torch.device("cuda:0")

# Calculate pixel counts for training and testing data
train_val = calculate_pixels(train_loader)
test_val = calculate_pixels(test_loader)

# Use the pixel counts to calculate weights for your loss function
weights_train = calculate_weights(train_val[0], train_val[1]).to(device)
weights_test = calculate_weights(test_val[0], test_val[1]).to(device)

print(f"Train weights: {weights_train}")
