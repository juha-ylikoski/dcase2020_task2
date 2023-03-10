
import glob
import sys


total = 0
num = 0
m = 0
for file in glob.glob(sys.argv[1] + "/*" + sys.argv[2] + "*.csv"):
    with open(file) as f:
        for line in f:
            _, score = line.split(",")
            score = float(score.strip())
            total += score
            num += 1
            m = max(m, score)

print(total / num)
print(m)


