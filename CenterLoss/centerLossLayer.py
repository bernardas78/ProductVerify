from CenterLoss.centerLossLayer_Eucl import CenterLossLayer as CenterLossLayer_Eucl, center_loss as center_loss_Eucl
from CenterLoss.centerLossLayer_Manhattan import CenterLossLayer as CenterLossLayer_Manhattan, center_loss as center_loss_Manhattan

def CenterLossLayer(distName):
    assert (distName=="Manhattan" or distName=="Eucl")
    if distName=="Manhattan":
        return CenterLossLayer_Manhattan
    elif distName=="Eucl":
        return CenterLossLayer_Eucl

def center_loss(distName):
    assert (distName=="Manhattan" or distName=="Eucl")
    if distName=="Manhattan":
        return center_loss_Manhattan
    elif distName=="Eucl":
        return center_loss_Eucl
