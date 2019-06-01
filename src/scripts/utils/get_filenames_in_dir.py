from os import listdir
from os.path import isfile, join


path = "3T"
file = open(path + "_filenames.txt", 'w');
for f in listdir(path):
    if isfile(join(path, f)):
        file.write(f[:6]);
        file.write('\n');
file.close();
