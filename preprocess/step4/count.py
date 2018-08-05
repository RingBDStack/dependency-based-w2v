import codecs
import sys
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--file', type=str, default = None)
parser.add_argument('--quantity', type=int, default=5785)
args = parser.parse_args()
output = codecs.open('weightcn.txt', 'w', 'utf-8')
dic = {}
i = 0
for j in range(1, args.quantity):
    dic[j] = 0
with codecs.open(args.file, 'r', 'utf-8')as f:
    for lines in f:
       line = lines.replace('\n', '')
       if len(line)==0:
           continue
       str1 = line.split(' ')
       i = 0
       for s in str1:
           if (i!=0) and (i%2==0):
               str2 = s.split(',')
               for j in str2:
                   if len(j)==0:
                       continue
                   if int(j) not in dic:
                       dic[int(j)] = 1
                   else:
                       dic[int(j)] = dic[int(j)] + 1
           i = i + 1

items=dic.items()
backitems=[[v[0],v[1]] for v in items]
backitems.sort(reverse=False)
for i in backitems:
    output.write(str(i[1]))
    output.write('\n')
output.close()
