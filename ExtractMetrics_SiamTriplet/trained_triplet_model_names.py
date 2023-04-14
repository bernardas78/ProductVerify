
x_names = {
    0: "_triplet_20230404_70633408.",     #        1epoch                       AUC = 0.952
    #0: "_triplet_20230406_03015603."     # untrained (1epoch Train10 lr=1e-10;)    AUC = 0.883
    #0: "_triplet_20230406_01001201."     # Train10 val_loss: 0.1215 daug epochs;   AUC = 0.990
    1: "_triplet_20230407_37542831.",     # 100 trainable layers
    2: "_triplet_20230407_87902633.",     # 8 trainable layers
    3: "_triplet_20230407_83535860.",     # 4 trainable layers

    4: "_triplet_20230408_22157196.",  # Manhattan (8 trainable)  val_loss: 0.0840
    5: "_triplet_20230408_87285641.",  # Eucl      (8 trainable)  val_loss: 0.0676
    6: "_triplet_20230408_11875697.",  # Mink3     (8 trainable)  val_loss: 0.0676
    7: "_triplet_20230408_17346348.",  # Mink4     (8 trainable)  val_loss: 0.0661
    8: "_triplet_20230408_45128494."   # Cosine     (8 trainable) val_loss: 0.2417

}



def model_name ( exper_index ):
    return "model"+x_names[exper_index]+"h5"

def lc_name ( exper_index ):
    return "lc"+x_names[exper_index]+"csv"

def suffix_name ( exper_index ):
    return x_names[exper_index]
