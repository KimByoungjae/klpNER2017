

def main():


    f = open("data/corpus/kor/2017train_trans.txt", "rb")
    f2 = open("data/corpus/kor/2017train_alter.txt", "w")



    lines = f.readlines()
    for line in lines:
        line = line.decode('utf-8')
        line = line.strip()
        if line == '':
            f2.write('\n')
            continue
        word_tok = line.split('\t')
        if word_tok[0] == ';' or word_tok[0][0:1] == '$':
            continue
        wt = []
        if word_tok[0] == '//SP':
            wt.append('/')
            wt.append('SP')
        elif word_tok[0] == '//SS':
            wt.append('/')
            wt.append('SS')
        else:
            wt = word_tok[0].split('/')
        f2.write('1\t'+wt[0]+'\t'+wt[1]+'\t'+word_tok[1]+'\n')




    f.close()
    f2.close()



main()
