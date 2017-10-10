

def main():


    f = open("data/coNLL/kor/train_trans2016.txt", "rb")
    f2 = open("data/coNLL/kor/train_trans2017.txt", "w")



    lines = f.readlines()
    for line in lines:
        line = line.decode('utf-8')
        line = line.strip()
        if line == '':
            f2.write('\n')
            continue
        word_tok = line.split()
        if word_tok[1] == 'I-PS' or word_tok[1] == 'I-DT' or word_tok[1] == 'I-LC' or word_tok[1] == 'I-TI' or word_tok[1] == 'I-OG':
            f2.write(word_tok[0]+'\tI\n')
        elif word_tok[1] == 'B-PS':
            f2.write(word_tok[0]+'\tB_PS\n')
        elif word_tok[1] == 'B-DT':
            f2.write(word_tok[0]+'\tB_DT\n')
        elif word_tok[1] == 'B-LC':
            f2.write(word_tok[0]+'\tB_LC\n')
        elif word_tok[1] == 'B-TI':
            f2.write(word_tok[0]+'\tB_TI\n')
        elif word_tok[1] == 'B-OG':
            f2.write(word_tok[0]+'\tB_OG\n')
        else:
            f2.write(word_tok[0]+'\t'+word_tok[1]+'\n')

    f.close()
    f2.close()



main()
