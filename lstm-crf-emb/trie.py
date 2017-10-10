# -*- coding: utf-8 -*-
import sys


# reload(sys)
# sys.setdefaultencoding('utf-8')
def insert_trie(u, tag, lis):
    l = lis
    end = 0
    for i in range(len(u)):
        check = 0
        if (u[i] == u" "):
            continue
        if i == len(u) - 1:
            if tag == "PS":
                end = 1
            elif tag == "LC":
                end = 2
            elif tag == "OG":
                end = 3
            elif tag == "DT":
                end = 4
            elif tag == "TI":
                end = 5
        for letter in l:
            if u[i] == letter[0]:
                check = 1
                if i == len(u) - 1:
                    letter[2] = end
                l = letter[1]
                break
        if check == 0:
            temp = []
            l.append([u[i], temp, end])
            l = temp


def find_trie(u, lis, label):
    l = lis
    check = 0
    start = -1
    end = -1
    savej = -1
    for i in range(len(u)):
        if i <= savej:
            continue
        find = 0
        savej = -1
        for j in range(i, len(u)):
            for k in range(len(u[j])):
                check = 0
                for letter in l:
                    if letter[0] == u[j][k]:
                        l = letter[1]
                        check = 1
                        if k == len(u[j]) - 1 and letter[2] != 0:
                            tag = letter[2]
                            start = i
                            end = j
                            find = 1
                        break
                if check == 0:
                    break
            if check == 0 or j == len(u) - 1:
                l = lis
                if find == 1:
                    t = find_label(start, end, label, tag)
                    savej = end
                    if label[start] != "B_" + t:
                        se = ""
                        for k in range(start, end + 1):
                            se += u[k]

                    # label[start] = "B_" + t
                    if t == "PS":
                        label[start] = 1.0
                    elif t == "OG":
                        label[start] = 2.0
                    elif t == "LC":
                        label[start] = 3.0
                    elif t == "DT":
                        label[start] = 4.0
                    elif t == "TI":
                        label[start] = 5.0

                    if start != end:
                        for i in range(start + 1, end + 1):
                            # label[i] = "I"
                            # label[i] = 6.0

                            if t == "PS":
                                label[i] = 1.0
                            elif t == "OG":
                                label[i] = 2.0
                            elif t == "LC":
                                label[i] = 3.0
                            elif t == "DT":
                                label[i] = 4.0
                            elif t == "TI":
                                label[i] = 5.0
                break


def find_label(start, end, label, tag):
    if tag == 1:
        t = "PS"
    elif tag == 2:
        t = "LC"
    elif tag == 3:
        t = "OG"
    elif tag == 4:
        t = "DT"
    elif tag == 5:
        t = "TI"
    return t


def gazette(lis, filename):
    f = open(filename, "r")

    for line in f.readlines():

        if len(line.strip()) == 0:
            sentences.append(sentence)
            sentence = []
            continue
        line = line.rstrip()

        char = line.split("\t")[0]
        tag = line.split("\t")[1]

        if tag == "DT" or tag == "TI":
            continue
        insert_trie(char, tag, lis)


def gazette_DTTI(lis, filename):
    f = open(filename, "r")

    for line in f.readlines():

        if len(line.strip()) == 0:
            sentences.append(sentence)
            sentence = []
            continue
        line = line.rstrip()
        char = line.split("\t")[0]
        tag = line.split("\t")[1]
        insert_trie(char, tag, lis)


# def test():
# 	lis=[]
# 	insert_trie(unicode("한 해해"),"LC",lis)
# 	s=[unicode("한 해해"),unicode("한 해 해"),unicode("한해 해"),unicode("한해해"),unicode("권"),unicode("을")]
# 	label=["O","O","O","O","O","O","O","O","O"]
# 	find_trie(s,lis,label)
# 	print label
"""
lis=[]
gazette_DTTI(lis,"DT_analysis.txt")
u=[unicode('p'),unicode('년'),unicode('p'),unicode('월'),unicode('p'),unicode('일')]
#insert_trie(unicode("p년p월p일"),"DT",lis)
#insert_trie(unicode("p년p월p일부터"),"DT",lis)
label=['O','O','O','O','O','O','O']
find_trie(u,lis,label)
print label
"""
