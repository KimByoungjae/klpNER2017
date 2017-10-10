from konlpy.tag import Komoran

komoran = Komoran()



f = open("data/corpus/kor/2017klpner_testset.txt", 'rb')
f2 = open("data/corpus/kor/2017klpner_test_anal.txt", 'w')

lines = f.readlines()

for line in lines:
    if line == b'\n':
        continue
    line = line.decode()
    line = line.strip().split('\t')[1]
    f2.write('; '+line+'\n')
    komo = komoran.pos(line)
    for i, word in enumerate(komo):
        f2.write(str(i+1)+'\t'+word[0]+'\t'+word[1]+'\tO\n')
    f2.write('\n')

f.close()
f2.close()
