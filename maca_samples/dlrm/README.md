Description of dlrm:
--------------------
An implementation of a deep learning recommendation model (DLRM)
The model input consists of dense and sparse features. The former is a vector
of floating point values. The latter is a list of sparse indices into
embedding tables, which consist of vectors of floating point values.
The selected vectors are passed to mlp networks denoted by triangles,
in some cases the vectors are interacted through operators (Ops).
```
output:
                    probability of a click
model:                        |
                             /\
                            /__\
                              |
      _____________________> Op  <___________________
    /                         |                      \
   /\                        /\                      /\
  /__\                      /__\           ...      /__\
   |                          |                       |
   |                         Op                      Op
   |                    ____/__\_____           ____/__\____
   |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
input:
[ dense features ]     [sparse indices] , ..., [sparse indices]
```
 More precise definition of model layers:
 1) fully connected layers of an mlp

    z = f(y)

    y = Wx + b

 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])

    z = Op(e1,...,ek)

    obtain vectors e1=E[:,p1], ..., ek=E[:,pk]

 3) Operator Op can be one of the following

    Sum(e1,...,ek) = e1 + ... + ek

    Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]

    Cat(e1,...,ek) = [e1', ..., ek']'

    where ' denotes transpose operation


Code reference of dlrm:
-----------------------
```
https://github.com/facebookresearch/dlrm
```


Configuration of dlrm
----------------------
# rand seed config
   rand_seed = 123
# error accepted eps
   eps = 1e-3
# create max num of each input sparse tensor
   max_num_sparse = 10
# module sparse config
   embedding_num = [4, 3, 2]
   embedding_dim = 2
   embedding_mode = "sum"  # "mean", "sum", "max"
   embedding_offset = 0
# module bot config
   layer_bot = [4, 3]
# module top config
   layer_top = [4, 2, 1]
# module interaction config
   interaction_mode = "dot"  # "dot", "cat"
# module loss config
   loss_threshold = 0.0
   loss_function = "mse"  # "mse", "bce"


How to run dlrm code
---------------------
Please refer to MACAMACA PyTorch's User Guide for instructions.


How to judge DLRM running correctly or not
-------------------------------------------
Please refer to MACAMACA PyTorch's User Guide for instructions.


Additional illustrtion
-----------------------
Please refer to MACAMACA PyTorch's User Guide for instructions.
