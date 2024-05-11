from preprocessing import prepare

# Define your input directory where the data is located.
data_dir = '/home/iot/Desktop/RAKIB_MI/1_multi_parametric_MRI/3D_prostate_cancer_seg/Dataset/Dataset/'  # Replace with the actual path to your dataset.

# Call the prepare function with the input directory.
train_loader, test_loader = prepare(data_dir)

# Now you can use the train_loader and test_loader in your training loop.
# For example, iterate through the data batches:
for batch in train_loader:
    data = batch["img"]
    label = batch["seg"]

    # Convert it to size [1, 1, 256, 256, 16]
    # label = label.unsqueeze(1)  #this statement is needed in training section
    # label = label[0:1, 0:1]


    # Perform your training using the data and label
    # Make sure your training loop corresponds to your specific model and loss function.

    print(data.shape)
    print(label.shape)

    break
