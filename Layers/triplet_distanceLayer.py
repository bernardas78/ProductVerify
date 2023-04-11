from Layers.triplet_distanceLayer_Manh import DistanceLayer as DistanceLayer_Manh, dist_func as dist_func_Manh
from Layers.triplet_distanceLayer_Eucl import DistanceLayer as DistanceLayer_Eucl, dist_func as dist_func_Eucl
from Layers.triplet_distanceLayer_Mink3 import DistanceLayer as DistanceLayer_Mink3, dist_func as dist_func_Mink3
from Layers.triplet_distanceLayer_Mink4 import DistanceLayer as DistanceLayer_Mink4, dist_func as dist_func_Mink4
from Layers.triplet_distanceLayer_Cosine import DistanceLayer as DistanceLayer_Cosine, dist_func as dist_func_Cosine

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

def dist_func(distName):
    assert (distName == "Manh" or distName == "Eucl" or distName == "Mink3" or distName == "Mink4" or distName == "Cosine")

    if distName=="Eucl":
        #print ("Dist func: Eucl")
        return dist_func_Eucl
    elif distName=="Manh":
        #print ("Dist func: Manh")
        return dist_func_Manh
    elif distName == "Mink3":
        #print ("Dist func: Mink, p=3")
        return dist_func_Mink3
    elif distName=="Mink4":
        #print ("Dist func: Mink, p=4")
        return dist_func_Mink4
    elif distName=="Cosine":
        #print ("Dist func: Cosine")
        return dist_func_Cosine