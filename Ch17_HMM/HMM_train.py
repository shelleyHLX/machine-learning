

#-*-coding:utf-8
import sys


import codecs
import numpy as np
state_M = 4
word_N = 0

trans_dic = {}
emit_dic = {}
Count_dic = {}
Pi_dic = {}
word_set = set()
state_list = ['B','M','E','S']
line_num = -1

INPUT_DATA = "RenMinData.txt_utf8"
PROB_START = "prob_start.npy"  #
PROB_EMIT = "prob_emit.npy"  # 发射概率 # B->我# {'B':{'我':0.34, '门':0.54}，'S':{'我':0.34, '门':0.54}}
PROB_TRANS = "prob_trans.npy"  # 转移概率 BMES*BMES
# PROB_START {'B': 0.5820149148537713, 'M': 0.0, 'E': 0.0, 'S': 0.41798844132394497}

def init():
    global state_M
    global word_N
    for state in state_list:
        trans_dic[state] = {}
        for state1 in state_list:
            trans_dic[state][state1] = 0.0
    for state in state_list:
        Pi_dic[state] = 0.0
        emit_dic[state] = {}
        Count_dic[state] = 0

def getList(input_str):
    """

    :param input_str: 过年，一个词，
    :return: BE,BMMMS,S,BMMS
    """
    outpout_str = []
    if len(input_str) == 1:
        outpout_str.append('S')
    elif len(input_str) == 2:
        outpout_str = ['B','E']
    else:
        M_num = len(input_str) -2
        M_list = ['M'] * M_num
        outpout_str.append('B')
        outpout_str.extend(M_list)
        outpout_str.append('E')
    return outpout_str

def Output():
    # start_fp = codecs.open(PROB_START,'a', 'utf-8')
    # emit_fp = codecs.open(PROB_EMIT,'a', 'utf-8')
    # trans_fp = codecs.open(PROB_TRANS,'a', 'utf-8')

    print ("len(word_set) = %s " % (len(word_set)))
    for key in Pi_dic.keys():
        '''
        if Pi_dic[key] != 0:
            Pi_dic[key] = -1*math.log(Pi_dic[key] * 1.0 / line_num)
        else:
            Pi_dic[key] = 0
        '''
        Pi_dic[key] = Pi_dic[key] * 1.0 / line_num
    # print >>start_fp,Pi_dic
    np.save(PROB_START, Pi_dic)

    for key in trans_dic:
        for key1 in trans_dic[key]:
            '''
            if A_dic[key][key1] != 0:
                A_dic[key][key1] = -1*math.log(A_dic[key][key1] / Count_dic[key])
            else:
                A_dic[key][key1] = 0
            '''
            trans_dic[key][key1] = trans_dic[key][key1] / Count_dic[key]
    # print >>trans_fp,A_dic
    # for k, v in A_dic:
    #     trans_fp.write(k+' '+str(v)+'\n')
    np.save(PROB_TRANS, trans_dic)

    for key in emit_dic:
        for word in emit_dic[key]:
            '''
            if B_dic[key][word] != 0:
                B_dic[key][word] = -1*math.log(B_dic[key][word] / Count_dic[key])
            else:
                B_dic[key][word] = 0
            '''
            emit_dic[key][word] = emit_dic[key][word] / Count_dic[key]

    # print >> emit_fp,B_dic
    np.save(PROB_EMIT, emit_dic)
    # for k, v in B_dic:
    #     emit_fp.write(k+' '+str(v)+'\n')
    # start_fp.close()
    # emit_fp.close()
    # trans_fp.close()


def main():
    # python HMM_train.py
    # if len(sys.argv) != 2:
    #     print ("Usage [%s] [input_data] " % (sys.argv[0]))
    #     sys.exit(0)
    ifp = codecs.open('RenMinData.txt_utf8', 'r', 'utf-8')
    init()
    global word_set
    global line_num
    for line in ifp.readlines():
        line_num += 1
        if line_num % 10000 == 0:
            print (line_num)

        line = line.strip()
        if not line:continue
        # line = line.decode("utf-8","ignore")
        # １９８６年 ，
        # 十亿 中华 儿女 踏上 新 的 征 程 。
        # 过去 的 一 年 ，

        word_list = []  # [过,去,的,一,年,]一个一个的字
        for i in range(len(line)):
            if line[i] == " ":
                continue
            word_list.append(line[i])
        word_set = word_set | set(word_list)


        lineArr = line.split(" ")  #　【过去，的，一，年，】
        line_state = []  # BMMS, BMMS,S,BE
        for item in lineArr:
            line_state.extend(getList(item))
        # pdb.set_trace()
        if len(word_list) != len(line_state):
            print (sys.stderr,"[line_num = %d][line = %s]" % (line_num, line))
        else:
            for i in range(len(line_state)):
                if i == 0:
                    Pi_dic[line_state[0]] += 1
                    Count_dic[line_state[0]] += 1
                else:
                    trans_dic[line_state[i-1]][line_state[i]] += 1
                    Count_dic[line_state[i]] += 1
                    # if not B_dic[line_state[i]].has_key(word_list[i]):
                    if word_list[i] not in emit_dic[line_state[i]].keys():
                        emit_dic[line_state[i]][word_list[i]] = 0.0
                    else:
                        emit_dic[line_state[i]][word_list[i]] += 1
    Output()
    ifp.close()
if __name__ == "__main__":
    main()
    print(Pi_dic)
    print(Count_dic)
    a = np.load(PROB_EMIT, allow_pickle=True).item()
    print('emit')
    print(a)
    b = np.load(PROB_TRANS, allow_pickle=True).item()
    print('trans')
    print(b)
    c = np.load(PROB_START, allow_pickle=True).item()
    print('start')
    print(c)
"""
{'B': {'B': 0.0, 'M': 0.1167175117318146, 'E': 0.8832824882681853, 'S': 0.0}, 
'M': {'B': 0.0, 'M': 0.2777743117140081, 'E': 0.0, 'S': 0.7222256882859919}, 
'E': {'B': 0.46951648068515556, 'M': 0.0, 'E': 0.0, 'S': 0.5304835193148444}, 
'S': {'B': 0.3607655156767958, 'M': 0.0, 'E': 0.0, 'S': 0.47108435638736734}}

"""