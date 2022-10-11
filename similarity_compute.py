import os, re
from gensim.models.fasttext import FastText
from antlr4 import *
from scipy.spatial import distance
import numpy as np
from solidityparser.SolidityLexer import SolidityLexer
from solidityparser.SolidityParser import SolidityParser


loading = False
if loading:
    model = FastText.load("FastText/fasttext50/fasttext.model")

def get3TokenSeq(path1, path2):
    with open(path1) as f:
        lines = f.readlines()
    buglines = [t[0] for t in enumerate(lines) if '// fault line' in t[1]]
    bugline = buglines[0]
    with open(path2, 'w') as again:
        for i in range(3):
            path3 = '{0}-{1}.sol'.format(path2.strip('.sol'), i+1)
            with open(path3, 'w') as f:
                for line in lines[bugline - 1 + i]:
                    f.write(line)
            parser = SolidityParser(CommonTokenStream(SolidityLexer(FileStream(path3))))
            tree = parser.sourceUnit()
            code_line = tree.toCodeSequence()
            regex = r'(\[)[0-9\s]*(\])'
            code_line = re.sub(regex, '', code_line)
            regex2 = "<missing [^>]+>"
            code_line = re.sub(regex2, '', code_line)
            tokenSeq = [x for x in code_line.split(' ') if not x == '' and not x == '<EOF>']
            seq = ' '.join(tokenSeq)
            again.write(seq)
            again.write('\n')
    for i in range(3):
        path4 = '{0}-{1}.sol'.format(path2.strip('.sol'), i + 1)
        os.remove(path4)

def getLineVector(line, dimension):
    token_list = line.strip('\n').split(' ')
    vector = np.zeros((dimension,), dtype="float64")
    token_num = 0
    global model
    for token in token_list:
        if token in model:
            token_num += 1
            vector = np.add(vector, model[token])
    lineVector = np.array([vector]) / token_num
    return lineVector

def getSimilarity(v, e):
    numerator = distance.cdist(v.reshape(1, 150), e, 'euclidean')
    denominator = np.linalg.norm(e, axis=1) + np.linalg.norm(v)
    return 1 - np.divide(numerator, denominator)

def get_similarity(contract_code_path):
    five_list = []
    bts = [0, 1, 2, 3, 4]
    for bt in bts:
        tmp_dir = "dataset_vul/newALLBUGS/tmp/tmp_tokenseq3/"
        os.makedirs(tmp_dir, exist_ok=True)
        context1_npy_path = "FastText/bugEmbedding/{}context1.npy".format(bt)
        bug_npy_path = "FastText/bugEmbedding/{}bug.npy".format(bt)
        context2_npy_path = "FastText/bugEmbedding/{}context2.npy".format(bt)
        context1_npy = np.load(context1_npy_path)
        bug_npy = np.load(bug_npy_path)
        context2_npy = np.load(context2_npy_path)

        tmp_3tokenseq_path = tmp_dir + contract_code_path.split('/')[-1]
        get3TokenSeq(contract_code_path, tmp_3tokenseq_path)
        with open(tmp_3tokenseq_path) as tf:
            threelines = tf.readlines()
        context1_vector = getLineVector(threelines[0], 150)
        bug_vector = getLineVector(threelines[1], 150)
        context2_vector = getLineVector(threelines[2], 150)

        sims_list = []
        for vs in zip(context1_npy, bug_npy, context2_npy):
            sims1 = getSimilarity(vs[0], context1_vector)
            sims2 = getSimilarity(vs[1], bug_vector)
            sims3 = getSimilarity(vs[2], context2_vector)
            w1, w2, w3 = (1, 2, 1)
            sims = (w1 * sims1 + w2 * sims2 + w3 * sims3) / (w1 + w2 + w3)
            sims_list.append(sims[0][0])

        os.remove(tmp_3tokenseq_path)
        five_list.append(max(sims_list))

    return max(five_list)
