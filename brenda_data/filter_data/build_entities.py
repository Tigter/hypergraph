import json


def buil_dict():
     with open("./train_reaction.json") as f:
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



def build_ec_number_data():
    with open("./p2ec.json") as f:
        datas = json.load(f)

    p_set = set()
    ec_set = set()
    for key,value in datas.items():
        p_set.add(key)
        ec_set.add(value)

    p_list = []
    p2id = {}
    with open("./reaction_relation.dict") as f:
        lines = f.readlines()
        for line in lines:
            p,id_1 = line.strip().split("\t")
            p_list.append(p)
            p2id[p] = int(id_1)

    def handle_ec_data(ec_list):
        classSet = set()
        for ec in ec_list:
            data = ec.split(".")
            classSet.add(data[0])
            classSet.add(".".join(data[0:2]))
            classSet.add(".".join(data[0:3]))
          #   classSet.add(".".join(data[0:4]))
        classSet = list(classSet)
        class2id = {
            classSet[i]:i for i in range(len(classSet))
        }
        return class2id



    def handle_subClassOf(ec_list,class2id):
        triple_set = set()
        for ec in ec_list:
            data = ec.split(".")
            top1 = class2id[data[0]]
            top2 = class2id[".".join(data[0:2])]
            top3 = class2id[".".join(data[0:3])]
          #   top4 = class2id[".".join(data[0:4])]
            triple_set.add((top2, 0,top1))
            triple_set.add((top3, 0,top1))
          #   triple_set.add((top4, 0,top1))
            triple_set.add((top3, 0,top2))
          #   triple_set.add((top4, 0,top2))
          #   triple_set.add((top4, 0,top3))
        
        return list(triple_set)

    def handle_typeOf(datas, p_list,class2id,p2id):
        triple_set = set()

        for p in p_list:
            data = datas[p].split(".")
            top1 = class2id[data[0]]
            top2 = class2id[".".join(data[0:2])]
            top3 = class2id[".".join(data[0:3])]
          #   top4 = class2id[".".join(data[0:4])]
            p = p2id[p]
            triple_set.add((p,0,top1))
            triple_set.add((p,0,top2))
            triple_set.add((p,0,top3))
          #   triple_set.add((p,0,top4))
        return list(triple_set)

    class2id = handle_ec_data(ec_set)
    subclassData = handle_subClassOf(ec_set,class2id)
    typeOfdata = handle_typeOf(datas,p_list, class2id,p2id)

    with open("./ec_data/subclass.json","w") as f:
        json.dump(subclassData,f)

    with open("./ec_data/typeof.json","w") as f:
        json.dump(typeOfdata,f)

    print(len(p_list))
    print(len(class2id))


def build_id2smiles():
     with open("./name2smiles.json","r") as f:
          name2smiles = json.load(f)
     
     with open("./reaction_entity.dict","r") as f:
          name2id = {}
          for line in f:
               name,ids = line.strip().split("\t")
               name2id[name]=int(ids)
     id2smiles = {}
     for key,value in name2id.items():
          if key not in name2smiles:
               print("error")
          else:
               id2smiles[value] = name2smiles[key]

     with open("./id2smiles.json",'w') as f:
          json.dump(id2smiles,f)

# build_id2smiles()    
          
build_ec_number_data()