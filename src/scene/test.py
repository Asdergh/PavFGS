import pandas as pd


path = "/media/test/T7/ply_collection/gerrard-hall/sparse/images.txt"
DataFrame = pd.read_csv(path)
# with open(path, "r") as file:
#     lines = file.readlines()
#     headers = lines[1].replace(" ", "").replace("#", "")
#     del lines[:4]
    
#     lines = [headers, ] + lines[::2]

# with open(path, "w") as file:
#     file.writelines(lines)

# with open(path, "r") as file:
#     line = file.read()
#     line = line.replace(" ", ",")
    
# with open(path, "w") as file:
#     file.writelines(line)

print(DataFrame.tail())
# print(DataFrame.tail()O)