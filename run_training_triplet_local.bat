rem to run: d:\ProductVerify\run_training_triplet_local.bat

cd d:\ProductVerify
d:

rem $1 - Fulls Ds?
rem $2 - epochs
rem $3 - patience
rem $4 - trainable layers (4,8,100)
rem $5 - dist type

rem venv\Scripts\python.exe run_training_triplet.py False 5 2
venv\Scripts\python.exe run_training_triplet.py True 10 10 8 Eucl