# info = {
#     "key" : "value",
#     "name" : "appnacollage",
#     "age" : 45,
#     "is " : True,
#     "subject": ["pathon" , "c" , "java" ],
# }

# x = {
#     "a" : "1",
#     "b" : "2",
#     "c" : "3",
# }
# x["d"] = "4"
# print(x)

# print(info)
# print(info["name"])
# print(info["age"])
# info["age"] = "three" #change value 
# info["surname"] = " google" # add new key
# print(info)



# null_dict = {}
# null_dict["name"] = "baba"
# print(null_dict)


student = {
    "name" : "rahul", 
    "subject" : {
        "phy" : 97,
        "che":98,
        "math": 100,

    }
}

# s = {
#     "a" : "1",
#     "b" : {
#         "b1" : 1 ,
#         "b2" : 2,
# }
# }
# print(s) 

print(student["che"])
# print(student.keys())
# print(list(student.keys()))
# print(len(student))
# print(list( student.values()))
# pairs = (list(student.items()))
# print(pairs[1])
# print(student['name2']) #error
# print(student.get("name2")) # no error --> none value