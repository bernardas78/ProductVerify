from CenterLoss.centerLossLayer_Eucl_InterC import CenterLossLayer as CenterLossLayer_Eucl_InterC, center_loss as center_loss_Eucl_InterC
from CenterLoss.centerLossLayer_Eucl import CenterLossLayer as CenterLossLayer_Eucl, center_loss as center_loss_Eucl
from CenterLoss.centerLossLayer_Manhattan import CenterLossLayer as CenterLossLayer_Manhattan, center_loss as center_loss_Manhattan
from CenterLoss.centerLossLayer_Minkowski import CenterLossLayer as CenterLossLayer_Minkowski, center_loss as center_loss_Minkowski

def CenterLossLayer(distName, inclInterCenter):
    assert (distName=="Manhattan" or distName=="Eucl" or distName=="Minkowski")
    assert (distName=="Eucl" or inclInterCenter==False)
    if distName=="Manhattan":
        return CenterLossLayer_Manhattan
    elif distName=="Eucl" and inclInterCenter==True:
        return CenterLossLayer_Eucl_InterC
    elif distName=="Eucl" and inclInterCenter==False:
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
