

def main():


    f = open("data/coNLL/kor/2016klpNER.base_train", "rb")
    f2 = open("data/coNLL/kor/2017klp_train.txt", "w")



    lines = f.readlines()
    for line in lines:
        line = line.decode('utf-8')
        line = line.strip()
        if line == '':
            f2.write('\n')
            continue
        word_tok = line.split()
        if word_tok[0] == ';' or word_tok[0][0:1] == '$':
            continue
        f2.write(word_tok[1]+'/'+word_tok[2]+'\t'+word_tok[3]+'\n')




    f.close()
    f2.close()



main()
