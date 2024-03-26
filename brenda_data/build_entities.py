import json


with open("/home/skl/yl/ce_project/relation_cl/brenda_data/train_reaction.json") as f:
        train_data = json.load(f)


c_set = set()
e_set = set()
new_train_data = []
for left, right, e in train_data:
    new_left = []
    for c in left:
        if c != "":
            new_left.append(c)
        else:
            print("error")
            
    new_right = []
    for c in right:
         if c != "":
            new_right.append(c)
         else:
            print("error")
    
    if len(new_left) == 0 or len(new_right) == 0:
         continue
    else:
        for c in new_right:
             c_set.add(c)
        for c in new_left:
             c_set.add(c)
        
        e_set.add(e) 
        new_train_data.append((new_left,new_right,e))
c_set = sorted(list(c_set))
e_set = sorted(list(e_set))

entities_dict = {
     c_set[i]: i for i in range(len(c_set))
}

relation_dict = {
     e_set[i]: i for i in range(len(e_set))
}

with open("./reaction_entity.dict","w") as f:
     for k,v in entities_dict.items():
          f.write("%s\t%s\n"% (k,v))

with open("./reaction_relation.dict","w") as f:
     for k,v in relation_dict.items():
          f.write("%s\t%s\n"% (k,v))
