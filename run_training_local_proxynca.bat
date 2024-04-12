rem to run: d:\ProductVerify\run_training_local_proxynca.bat

cd d:\ProductVerify
d:

rem 1: full
rem 2: lst_dense_size
rem 3: epochs
rem 4: patience
rem 5: distName
rem 6: p_minkowski
rem 7: pre_cl_layer_ind

rem venv\Scripts\python.exe run_training_denseArgs_proxynca.py True 512 15 5 Minkowski 2 0
venv\Scripts\python.exe run_training_denseArgs_proxynca.py True 512 25 10 Minkowski 1 0
