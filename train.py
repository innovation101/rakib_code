from monai.networks.nets import UNet,AttentionUnet,RegUNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss,DiceFocalLoss, FocalLoss
# from CE_loss import DiceLoss, DiceCELoss, 
from monai.metrics import DiceMetric
import tqdm
import segmentation_models_pytorch_3d as smp
import torch

from preprocessing import prepare
from utilits import calculate_pixels, calculate_weights, train

data_in = '/home/iot/Desktop/RAKIB_MI/1_multi_parametric_MRI/3D_prostate_cancer_seg/Dataset/Dataset/'
model_dir = '/home/iot/Desktop/RAKIB_MI/1_multi_parametric_MRI/3D_prostate_cancer_seg/pca_trained_model/Random/with_weight_DiceFocalLoss/'



data_in = prepare(data_in, cache=False)

# Assuming you have already defined your prepare function and obtained train_loader and test_loader
train_loader, test_loader = prepare(data_in, cache=False)


device = torch.device("cuda:0")

# # Calculate pixel counts for training and testing data
# train_val = calculate_pixels(train_loader)
# test_val = calculate_pixels(test_loader)

# # Use the pixel counts to calculate weights for your loss function
# weights_train = calculate_weights(train_val[0], train_val[1]).to(device)
# weights_test = calculate_weights(test_val[0], test_val[1]).to(device)

# print(f"Train weights: {weights_train}")


model = smp.Unet(
    encoder_name="resnet50",        
    in_channels=3,                  
    strides=((2, 2, 2), (4, 2, 1), (2, 2, 2), (2, 2, 1), (1, 2, 3)),
    classes=2, 
).to(device)


# model=U_Net3D(img_ch=3, output_ch=2).to(device)


# Pass the calculated weights to your loss function
# loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=weights_train)

# loss_function = FocalLoss(include_background= True, to_onehot_y=True,gamma=2,weight=weights_train)
# loss_function = FocalLoss(include_background= True, to_onehot_y=True,gamma=2)

# loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(84372121,3118439).to(device))
loss_function = DiceFocalLoss(include_background= True, to_onehot_y=True, sigmoid=True, squared_pred=True,gamma=2)


optimizer = torch.optim.SGD(model.parameters(), 0.1, 0.9, 1e-5, weight_decay=1e-5)



if __name__ == '__main__':

    train(model, data_in, loss_function, optimizer, 300, model_dir)
    
    
    