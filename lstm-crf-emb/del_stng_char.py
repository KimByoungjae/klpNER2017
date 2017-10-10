import re




f = open("data/morph/morph_vector.txt", 'rb')

a = f.readlines()

f2= open("data/morph/morph_vector_s.txt", "w")

for line in a:
    line = line.decode('utf-8')
    word = line.split('\t')[0]
    word = word.lower()
    if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z]+.*',word) is not None:
        f2.write(line)


f.close()
f2.close()
