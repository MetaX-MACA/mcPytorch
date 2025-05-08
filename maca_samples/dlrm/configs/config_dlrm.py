# rand seed config
rand_seed = 727

# error accepted eps
eps = 1e-5

# module sparse config
embedding_num = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000]
embedding_dim = 64
embedding_mode = "sum"  # "mean", "sum", "max"
embedding_offset = 0

# module bot config
layer_bot = [512, 512]

# module top config
layer_top = [1024, 1024, 1024, 1]

# module interaction config
interaction_mode = "dot"  # "dot", "cat"

# module loss config
loss_threshold = 0.0
loss_function = "mse"  # "mse", "bce"
