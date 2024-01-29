
import pickle
with open("./heer.txt",'w') as f:
    f.write("Hello wold")
    f.close()

with open("./base_data.pkl",'rb') as f:
    data = pickle.load(f)
    print(data.keys())
    f.close()