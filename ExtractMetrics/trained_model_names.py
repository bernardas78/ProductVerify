
x_names_Eucl = {
    2: "_centerloss_20221108_dense_2.",
    4: "_centerloss_20221108_dense_4.",
    8: "_centerloss_20221108_dense_8.",
    16: "_centerloss_20221108_dense_16.",
    32: "_centerloss_20221108_dense_32.",
    64: "_centerloss_20221108_dense_64.",
    128: "_centerloss_20221108_dense_128.",
    256: "_centerloss_20221108_dense_256.",
    512: "_centerloss_20221108_dense_512.",
    768: "_centerloss_20221108_dense_768.",
    1024: "_centerloss_20221108_dense_1024.",
    1536: "_centerloss_20221108_dense_1536.",
    2048: "_centerloss_20221108_dense_2048."
}

x_names_Eucl_inclInterCenter = {
    512: "_centerloss_20230209_dense_512_Eucl_True_1.000."
}

x_names_Manhattan = {
    2: "_centerloss_20221123_dense_2_Manhattan.",
    4: "_centerloss_20221123_dense_4_Manhattan.",
    8: "_centerloss_20221123_dense_8_Manhattan.",
    16: "_centerloss_20221123_dense_16_Manhattan.",
    32: "_centerloss_20221123_dense_32_Manhattan.",
    64: "_centerloss_20221123_dense_64_Manhattan.",
    128: "_centerloss_20221123_dense_128_Manhattan.",
    256: "_centerloss_20221118_dense_256_Manhattan.",
    512: "_centerloss_20221118_dense_512_Manhattan.",
    768: "_centerloss_20221118_dense_768_Manhattan.",
    1024: "_centerloss_20221118_dense_1024_Manhattan.",
    1536: "_centerloss_20221118_dense_1536_Manhattan.",
    2048: "_centerloss_20221118_dense_2048_Manhattan."
}

x_names_Minkowski_3 = {
    2: "_centerloss_20221125_dense_2_Minkowski_3.",
    4: "_centerloss_20221125_dense_4_Minkowski_3.",
    8: "_centerloss_20221125_dense_8_Minkowski_3.",
    16: "_centerloss_20221125_dense_16_Minkowski_3.",
    32: "_centerloss_20221125_dense_32_Minkowski_3.",
    64: "_centerloss_20221125_dense_64_Minkowski_3.",
    128: "_centerloss_20221125_dense_128_Minkowski_3.",
    256: "_centerloss_20221125_dense_256_Minkowski_3.",
    512: "_centerloss_20221125_dense_512_Minkowski_3.",
    768: "_centerloss_20221125_dense_768_Minkowski_3.",
    1024: "_centerloss_20221125_dense_1024_Minkowski_3.",
    1536: "_centerloss_20221125_dense_1536_Minkowski_3.",
    2048: "_centerloss_20221125_dense_2048_Minkowski_3.",
}

x_names_Minkowski_4 = {
    2: "_centerloss_20221128_dense_2_Minkowski_4.",
    4: "_centerloss_20221127_dense_4_Minkowski_4.",
    8: "_centerloss_20221127_dense_8_Minkowski_4.",
    16: "_centerloss_20221127_dense_16_Minkowski_4.",
    32: "_centerloss_20221127_dense_32_Minkowski_4.",
    64: "_centerloss_20221127_dense_64_Minkowski_4.",
    128: "_centerloss_20221127_dense_128_Minkowski_4.",
    256: "_centerloss_20221127_dense_256_Minkowski_4.",
    512: "_centerloss_20221127_dense_512_Minkowski_4.",
    768: "_centerloss_20221127_dense_768_Minkowski_4.",
    1024: "_centerloss_20221127_dense_1024_Minkowski_4.",
    1536: "_centerloss_20221127_dense_1536_Minkowski_4.",
    2048: "_centerloss_20221127_dense_2048_Minkowski_4.",
}

x_names = {
    "Eucl": x_names_Eucl,
    "Eucl_inclInterCenter": x_names_Eucl_inclInterCenter,
    "Manhattan": x_names_Manhattan,
    "Minkowski_3": x_names_Minkowski_3,
    "Minkowski_4": x_names_Minkowski_4
}

def model_names (dist_name, prelast_size, p_minkowski, inclInterCenter):
    if dist_name=="Minkowski":
        dist_name = "{}_{}".format(dist_name, p_minkowski)
    if inclInterCenter==True:
        dist_name = "{}_{}".format(dist_name,"inclInterCenter")
    return "model"+x_names[dist_name][prelast_size]+"h5"

def lc_names (dist_name, prelast_size, p_minkowski, inclInterCenter):
    if dist_name=="Minkowski":
        dist_name = "{}_{}".format(dist_name, p_minkowski)
    if inclInterCenter==True:
        dist_name = "{}_{}".format(dist_name,"inclInterCenter")
    return "lc"+x_names[dist_name][prelast_size]+"csv"