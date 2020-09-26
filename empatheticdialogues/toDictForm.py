import json

fr = open("train.csv","r")
lines = fr.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].split(",")
    if len(lines[i]) != 8:
        print(lines[i])

curConvId = lines[1][0]
curConvDict = {}
curConvDict["conv_id"] = curConvId
curConvDict["context"] = lines[1][2]
curConvDict["prompt"] = lines[1][3]
curConvDict["selfeval"] = lines[1][6]
curConvDict["utterances"] = []
curConvDict["utterances_idx"] = []
curConvDict["speakers_idx"] = []

allConvs = []

for i in range(1, len(lines)):
    if lines[i][0] == curConvId:
        curConvDict["utterances"].append(lines[i][5])
        curConvDict["utterances_idx"].append(lines[i][1])
        curConvDict["speakers_idx"].append(lines[i][4])
        #print(1, curConvDict)
    else:
        #print(2, curConvDict)
        allConvs.append(curConvDict.copy())
        #print(3, allConvs)
        curConvId = lines[i][0]
        #curConvDict = {}        
        curConvDict["conv_id"] = curConvId
        curConvDict["context"] = lines[i][2]
        curConvDict["prompt"] = lines[i][3]
        curConvDict["selfeval"] = lines[i][6]
        curConvDict["utterances"] = []
        curConvDict["utterances_idx"] = []
        curConvDict["speakers_idx"] = []
        
        curConvDict["utterances"].append(lines[i][5])
        curConvDict["utterances_idx"].append(lines[i][1])
        curConvDict["speakers_idx"].append(lines[i][4])
 
allConvs.append(curConvDict.copy())
with open("train.json", "w+") as fw:
    json.dump(allConvs, fw, ensure_ascii=False)
