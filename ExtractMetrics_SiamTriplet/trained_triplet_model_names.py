
x_names_Eucl = {
    0: "_triplet_20230404_70633408."      #        1epoch                       AUC = 0.952
    #0: "_triplet_20230406_03015603." # untrained (1epoch Train10 lr=1e-10;)    AUC = 0.883

    #0: "_triplet_20230406_01001201." # Train10 val_loss: 0.1215 daug epochs;   AUC = 0.990
}



def model_name ( exper_index ):
    return "model"+x_names_Eucl[exper_index]+"h5"

def lc_name ( exper_index ):
    return "lc"+x_names_Eucl[exper_index]+"csv"

def suffix_name ( exper_index ):
    return x_names_Eucl[exper_index]
