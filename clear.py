import os

directory_path = "./elwe_err/"
file_list = os.listdir(directory_path)
error_files = 0

for file_name in file_list:
    file_path = directory_path + file_name
    if os.path.isfile(file_path):
        if os.path.getsize(file_path) > 0:
            error_files += 1
        else:
            os.remove(file_path)


directory_path2 = "./elwe_out/"
file_list2 = os.listdir(directory_path2)
for file_name in file_list2:
    file_path = directory_path2 + file_name
    if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
        os.remove(file_path)

print("error files: " + str(error_files))