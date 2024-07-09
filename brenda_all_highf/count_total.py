import json
with open("./train.json") as f:
    train = json.load(f)

with open("./valid.json") as f:
    valid = json.load(f)


with open("./test.json") as f:
    test = json.load(f)

print(len(train) + len(valid) + len(test))