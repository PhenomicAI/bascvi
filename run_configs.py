
import os


files = os.listdir("./config")
files = [f for f in files if '.json' in f]

print("Config files being run:")

for i,f in enumerate(files):
    print("File: ", "F number: ", i)


for i,f in enumerate(files):
    print("File: ", "F number: ", i)
    os.system("python train_embedding_model.py -c ./config/" + f)
