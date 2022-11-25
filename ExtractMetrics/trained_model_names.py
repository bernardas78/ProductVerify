
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

x_names_Minkowski = {
    2: "_centerloss_20221108_dense_2.",
    #2: "_centerloss_20221123_dense_2_Manhattan.",
}
x_names = {
    "Eucl": x_names_Eucl,
    "Manhattan": x_names_Manhattan,
    "Minkowski": x_names_Minkowski
}

def model_names (dist_name, prelast_size):
    return "model"+x_names[dist_name][prelast_size]+"h5"

def lc_names (dist_name, prelast_size):
    return "lc"+x_names[dist_name][prelast_size]+"csv"