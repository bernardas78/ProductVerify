from Layers.triplet_distanceLayer_Manh import DistanceLayer as DistanceLayer_Manh
from Layers.triplet_distanceLayer_Eucl import DistanceLayer as DistanceLayer_Eucl
from Layers.triplet_distanceLayer_Mink3 import DistanceLayer as DistanceLayer_Mink3
from Layers.triplet_distanceLayer_Mink4 import DistanceLayer as DistanceLayer_Mink4
from Layers.triplet_distanceLayer_Cosine import DistanceLayer as DistanceLayer_Cosine

def DistanceLayer(distName):

    assert (distName == "Manh" or distName == "Eucl" or distName == "Mink3" or distName == "Mink4" or distName == "Cosine")

    if distName=="Eucl":
        return DistanceLayer_Eucl
    elif distName=="Manh":
        return DistanceLayer_Manh
    elif distName == "Mink3":
        return DistanceLayer_Mink3
    elif distName=="Mink4":
        return DistanceLayer_Mink4
    elif distName=="Cosine":
        return DistanceLayer_Cosine

