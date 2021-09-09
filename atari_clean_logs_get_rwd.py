import os
import re




infile = "log.csv"
outfile = "cleaned_file.csv"

delete_list = ["Reward:", "word_2", "word_n"]
with open(infile) as fin, open(outfile, "w+") as fout:
    for line in fin:
        for word in delete_list:
            line = line.replace(word, "")
        fout.write(line)
