#utilits

from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
# from CE_loss import DiceCELoss
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import csv
import tqdm

from monai.metrics import DiceMetric

# dice_metric = DiceMetric(include_background=False, reduction="mean",get_not_nans=False)

def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value


def iou_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True, jaccard=True)
    value = 1 - dice_value(predicted, target).item()
    return value


def calculate_weights(val1, val2):
    '''
    In this function we take the number of the background and the forgroud pixels to return the `weights`
    for the cross entropy loss values.
    '''
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count/summ
    weights = 1/weights
    summ = weights.sum()
    weights = weights/summ
    return torch.tensor(weights, dtype=torch.float32)

def calculate_pixels(data):
    val = np.zeros((1, 2))

    for batch in tqdm.tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    print('The last values:', val)
    return val

def calculate_pixels(data):
    val = np.zeros(2)  # Initialize val as a 1-dimensional array
    for batch in tqdm.tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)
        if len(count) == 1:
            count = np.append(count, 0)
        val += count
    print('The last values:', val)
    return val


def train(model, data_in, loss, optim, max_epochs, model_dir, test_interval=1 , device=torch.device("cuda:0")):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []

    train_loader, test_loader = data_in

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        for batch_data in train_loader:

            train_step += 1

            volume = batch_data["img"]
            label = batch_data["seg"]
            label = label != 0
            volume, label = (volume.to(device), label.to(device))

            optim.zero_grad()

            
            outputs = model(volume)

            train_loss = loss(outputs, label)

            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")

            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        print('-'*20)

        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)

        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')

        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        if (epoch + 1) % test_interval == 0:
            with open(os.path.join(model_dir, f'dice_metrics_epoch_{epoch+1}.csv'), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch number', 'Patient ID', 'Dice Metric'])

                model.eval()
                with torch.no_grad():
                    test_epoch_loss = 0
                    test_metric = 0
                    epoch_metric_test = 0
                    test_step = 0

                    for test_data in test_loader:

                        test_step += 1

                        test_volume = test_data["img"]
                        test_label = test_data["seg"]
                        test_label = test_label != 0
                        test_volume, test_label = (test_volume.to(device), test_label.to(device),)

                        test_outputs = model(test_volume)

                        test_loss = loss(test_outputs, test_label)
                        test_epoch_loss += test_loss.item()

                        test_metric = dice_metric(test_outputs, test_label)
                        epoch_metric_test += test_metric

                        # Save dice metric for each patient in CSV
                        patient_id = test_data["patient_id"]
                        writer.writerow([epoch + 1, patient_id, test_metric])

                    test_epoch_loss /= test_step
                    print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                    save_loss_test.append(test_epoch_loss)
                    np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

                    epoch_metric_test /= test_step
                    print(f'test_dice_epoch: {epoch_metric_test:.4f}')
                    save_metric_test.append(epoch_metric_test)
                    np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

                    if epoch_metric_test > best_metric:
                        best_metric = epoch_metric_test
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))

                    print(
                        f"current epoch: {epoch + 1} current mean dice: {test_metric:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")
    
     # Load data from saved numpy files
    epoch = np.arange(1, max_epochs + 1)
    train_loss = np.load(os.path.join(model_dir, 'loss_train.npy'))
    train_metric = np.load(os.path.join(model_dir, 'metric_train.npy'))
    valid_metric = np.load(os.path.join(model_dir, 'metric_test.npy'))
    # Create a DataFrame
    data = {
        'Epoch number': epoch,
        'Train Loss': train_loss,
        'train dice loss': train_metric,
        'valid dice metric': valid_metric,
        # 'valid iou metric': valid_metric
    }
    df = pd.DataFrame(data)
    # Save the DataFrame to an Excel file
    excel_file_path = os.path.join(model_dir, 'training_metrics.xlsx')
    df.to_excel(excel_file_path, index=False)
    print(f"Excel file saved to: {excel_file_path}")  
    
    # Merge all epoch CSV files into one
    merge_epoch_csv(model_dir, max_epochs)

def merge_epoch_csv(model_dir, max_epochs):
    all_data = pd.DataFrame()
    for epoch in range(1, max_epochs + 1):
        csv_file = os.path.join(model_dir, f'dice_metrics_epoch_{epoch}.csv')
        if os.path.exists(csv_file):
            epoch_data = pd.read_csv(csv_file)
            all_data = pd.concat([all_data, epoch_data])

    merged_csv_file = os.path.join(model_dir, 'merged_epoch_metrics.csv')
    all_data.to_csv(merged_csv_file, index=False)
    print(f"All epoch CSV files merged and saved to: {merged_csv_file}")


