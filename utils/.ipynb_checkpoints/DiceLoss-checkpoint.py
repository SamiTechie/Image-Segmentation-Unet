import torch

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten the inputs and targets tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()

        # calculate the Dice coefficient
        dice_coeff = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # calculate the Dice loss
        dice_loss = 1 - dice_coeff

        return dice_loss
