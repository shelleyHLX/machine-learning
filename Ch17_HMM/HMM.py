#-*-coding:utf-8

import numpy as np

prob_start = np.load("prob_start.npy", allow_pickle=True).item()
prob_trans = np.load("prob_trans.npy", allow_pickle=True).item()
prob_emit = np.load("prob_emit.npy", allow_pickle=True).item()


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}] #tabular
    path = {}
    for y in states: #init
        V[0][y] = start_p[y] * emit_p[y].get(obs[0],0)
        path[y] = [y]
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            (prob,state ) = max([(V[t-1][y0] * trans_p[y0].get(y,0) * emit_p[y].get(obs[t],0) ,y0) for y0 in states if V[t-1][y0]>0])
            V[t][y] =prob
            newpath[y] = path[state] + [y]
        path = newpath
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return (prob, path[state])

def cut(sentence):
    #pdb.set_trace()
    prob, pos_list =  viterbi(sentence,('B','M','E','S'), prob_start, prob_trans, prob_emit)
    return (prob,pos_list)

def pos2word(test_str, pos_list):
    res = ''
    for i in range(len(pos_list)):
        if pos_list[i]=='B':
            res = res+test_str[i]
            # print(test_str[i], end='')
        elif pos_list[i]=='E':
            res = res + test_str[i]+'/'
            # print(test_str[i],'/', end='')
        elif pos_list[i]=='M':
            res = res + test_str[i]
            # print(test_str[i], end='')
        else:
            res = res + test_str[i] + '/'
            # print(test_str[i], '/', end='')
    print(res.strip('/'))

if __name__ == "__main__":
    test_str = u"计算机和电脑"
    prob,pos_list = cut(test_str)
    print (test_str)
    print (pos_list)
    pos2word(test_str, pos_list)
    test_str = u"他说的确实在理."
    prob,pos_list = cut(test_str)
    print (test_str)
    print (pos_list)
    pos2word(test_str, pos_list)
    test_str = u"毛主席万岁。"
    prob,pos_list = cut(test_str)
    print (test_str)
    print (pos_list)
    pos2word(test_str, pos_list)
    test_str = u"我有一台电脑。"
    prob,pos_list = cut(test_str)
    print (test_str)
    print (pos_list)
    pos2word(test_str, pos_list)