import pickle

def main():

    print("pickle")
    f = open("data/morph/char_embed.pkl", 'rb')
    a = pickle.load(f)
    f2 = open("data/morph/morph_vector.txt", 'w')

    morphs = a['char2idx']
    embeds = a['char_embed']

    for morph in morphs:
        f2.write("%s\t" %morph)
        for i in range(60):
            f2.write(str(embeds[morphs[morph]][i]))
            if i == 59:
                f2.write("\n")
            else:
                f2.write(" ")


    f.close()
    f2.close()


main()
