rem to run: d:\ProductVerify\run_training_local.bat

cd d:\ProductVerify
d:

rem 1: full
rem 2: lst_dense_size
rem 3: epochs
rem 4: patience
rem 5: distName
rem 6: p_minkowski
rem 7: lambda_centerloss
rem 8: pre_cl_layer_ind
rem 9: inclInterCenter
rem 10: lambda2
venv\Scripts\python.exe run_training_denseArgs.py False 512 25 25 Eucl 2 0.1 -2 False 0.000