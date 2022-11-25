from CenterLoss.centerLossLayer_Eucl import CenterLossLayer as CenterLossLayer_Eucl, center_loss as center_loss_Eucl
from CenterLoss.centerLossLayer_Manhattan import CenterLossLayer as CenterLossLayer_Manhattan, center_loss as center_loss_Manhattan
from CenterLoss.centerLossLayer_Minkowski import CenterLossLayer as CenterLossLayer_Minkowski, center_loss as center_loss_Minkowski

def CenterLossLayer(distName):
    assert (distName=="Manhattan" or distName=="Eucl" or distName=="Minkowski")
    if distName=="Manhattan":
        return CenterLossLayer_Manhattan
    elif distName=="Eucl":
        return CenterLossLayer_Eucl
    elif distName=="Minkowski":
        return CenterLossLayer_Minkowski

def center_loss(distName):
    assert (distName=="Manhattan" or distName=="Eucl" or distName=="Minkowski")
    if distName=="Manhattan":
        return center_loss_Manhattan
    elif distName=="Eucl":
        return center_loss_Eucl
    elif distName=="Minkowski":
        return center_loss_Minkowski
