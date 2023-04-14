
x_names_Eucl = {
    #0: "_siam_20230403_91621592.",     #57.8% acc 10 epochs (best epoch - last) AUC = 0.734
    #0: "_siam_20230404_55518080."      #77.1% acc after 14 epochs,              AUC = 0.882
    0: "_siam_20230406_41108880.",       #80.4% acc after 43 epochs,            AUC = 0.903
    #0: "_siam_20230411_04747448.",       # 100 trainable
    1: "_siam_20230411_19162702.",       # 8 trainable
    2: "_siam_20230411_46365986."        # 4 trainable
}



def model_name ( exper_index ):
    return "model"+x_names_Eucl[exper_index]+"h5"

def lc_name ( exper_index ):
    return "lc"+x_names_Eucl[exper_index]+"csv"

def suffix_name ( exper_index ):
    return x_names_Eucl[exper_index]
