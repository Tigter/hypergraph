import json

with open('./test_reaction.json', 'r') as f:
    data = json.load(f)

pair = []

for left,right,e in data:
    if len(left) == 1 :
        
        pair.append((left[0],e))
    else:
        for l in left:
            pair.append((l,e))


with open('./test_pair.json', 'w') as f:
    json.dump(pair, f)
    