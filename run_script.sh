

# observation ratio: 50%

# uber
python BayOTIDE.py --dataset=uber --num_fold=5  --machine=$USER --r=0.4 

# guangzhou
python BayOTIDE.py --dataset=guangzhou --num_fold=5 --machine=$USER --r=0.4

# solar
python BayOTIDE.py --dataset=solar --num_fold=5 --machine=$USER --r=0.4


# observation ratio: 70%

# uber
python BayOTIDE.py --dataset=uber --num_fold=5  --machine=$USER --r=0.2 

# guangzhou
python BayOTIDE.py --dataset=guangzhou --num_fold=5 --machine=$USER --r=0.2

# solar
python BayOTIDE.py --dataset=solar --num_fold=5 --machine=$USER --r=0.2