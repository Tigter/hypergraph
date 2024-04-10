
import json
import pubchempy as pcp
import requests


# with open("./train_reaction.json", "r") as f:
#     datas = json.load(f)

# with open("./name2smiles.json", "r") as f:
#     name2smiles = json.load(f)
# c_set = set()
# e_set = set()
# for left, right, e in datas:
#     for c in left: c_set.add(c)
#     for c in right: c_set.add(c)
#     e_set.add(e)

def get_compound_name(name):
    # Search for the compound in PubChem
    try:
        
        compound = pcp.get_compounds(name, 'smiles')
        # Get the IUPAC name from the first result
        if compound:
            cid = compound[0].cid
            # c = pcp.Compound.from_cid(cid)
            # smiles = c.canonical_smiles
            return None, compound[0].iupac_name, cid
        else:
            print("No compound found for name: " + name)
            return None,None,None
    except pcp.PubChemHTTPError as e:
        print(f"Error retrieving data from PubChem: {e}")
        return None,None,None
    except Exception as e:
        print(f"Error data from PubChem: {e}")
        return None,None,None

# print("total: %d \n" % len(c_set))
# count = 0
# mapping_result = []
# falie_name = []
# faile_count = 0
# for c in c_set:
#     count += 1
#     s, n,cid = get_compound_name(name2smiles[c])
#     if count % 10 == 0 and count != 0:
#         print("count %d failed:%d" % (count,faile_count))
#         if count % 100 == 0:
#             with open("./pub_result2/pubchempy_falied_name_%d.json" % count, 'w') as f:
#                 json.dump(falie_name,f)
#             with open("./pub_result2/pubchempy_mapping_%d.json" % count, 'w') as f:
#                 json.dump(mapping_result,f)
#             falie_name = []
#             mapping_result = []
#     if cid == None:
#         faile_count += 1
#         falie_name.append(c)
#     else:
#         if n == None: name="No_iupac_name "
#         mapping_result.append([c, cid, n, name2smiles[c]])

def update_text(cid):
    print(cid)
    flag = False
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/'
    req = requests.get(url)
    proper_json = json.loads(req.text)
    print(proper_json)
    Section = proper_json['Record']['Section']
    for item in Section:
        if item['TOCHeading'] == 'Names and Identifiers':
            #Information = item['Section'][0]['Information']
            Section = item['Section']
            for item in Section:
                if item['TOCHeading'] == 'Record Description':
                    Information = item['Information']
                    Value = Information[0]['Value']
                    text = Value['StringWithMarkup'][0]['String']
                    flag = True
                    with open(f'./text/text_{cid}.txt', 'w', encoding='utf-8') as f:
                        f.writelines(text)

# count = 0
# for i in range(100, 8500, 100):
#     file_name = "./pub_result/pubchempy_mapping_" + str(i) + ".json"
#     with open(file_name,"r") as f:
#         datas = json.load(f)
#     for data in datas:
#         count += 1
#         cid = data[1]
#         update_text(cid)
#         if count % 100 == 0:
#             print("finished count:%d" % count)
        


name2id = {}
with open("./reaction_entity.dict") as f:
    lines = f.readlines()
    for line in lines:
        name, temp_id = line.strip().split("\t")
        name2id[name] = int(temp_id)
id2cid = {}
count = 0

for i in range(100, 8500, 100):
    file_name = "./pub_result/pubchempy_mapping_" + str(i) + ".json"
    with open(file_name,"r") as f:
        datas = json.load(f)
    for data in datas:
        cid = data[1]
        id2cid[name2id[data[0]]] = cid

with open("./id2cid.json","w") as f:
    json.dump(id2cid, f)