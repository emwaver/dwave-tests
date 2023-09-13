import os

e_values = [50*(i+1) for i in range(10)]

directory_path = "./output/"
file_list = os.listdir(directory_path)

for e_value in e_values:
    required = 5

    for file_name in file_list:
        if ("e" + str(e_value) + "_") in file_name:
            required -= 1
        if ("c" + str(e_value) + "_") in file_name:
            required -= 1
    if required != 0:
        print(str(e_value) + " -> " + str(required))