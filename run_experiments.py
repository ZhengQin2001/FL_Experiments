import subprocess
cmds = ['python main.py -d fashion_mnist --federated_type dpmcf -ds SSG -bm mlp ',
'python main.py -d fashion_mnist --federated_type dpsgd -ds PSG -bm mlp',
'python main.py -d fashion_mnist --federated_type dpsgd -ds SSG -bm mlp']

for cmd in cmds:
    subprocess.run(cmd, shell=True)
