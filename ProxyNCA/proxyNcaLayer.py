from ProxyNCA.proxyNcaLayer_Mink import ProxyNcaLayer as ProxyNcaLayer_Mink, proxynca_loss as proxynca_loss_Mink
from ProxyNCA.proxyNcaLayer_Cosine import ProxyNcaLayer as ProxyNcaLayer_Cosine, proxynca_loss as proxynca_loss_Cosine

def ProxyNcaLayer(distName):
    print ("Choosing ProxyNcaLayer: {}".format(distName))
    assert (distName=="Minkowski" or distName=="Cosine")
    if distName=="Minkowski":
        return ProxyNcaLayer_Mink
    elif distName=="Cosine":
        return ProxyNcaLayer_Cosine

def proxynca_loss(distName):
    assert (distName=="Minkowski" or distName=="Cosine")
    if distName=="Minkowski":
        return proxynca_loss_Mink
    elif distName=="Cosine":
        return proxynca_loss_Cosine
