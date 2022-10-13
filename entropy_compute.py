import os
import re
import subprocess
from antlr4 import *
from solidityparser.SolidityLexer import SolidityLexer
from solidityparser.SolidityParser import SolidityParser

def get_entropy(contract_path, use_cache):
    if use_cache == False:
        options = '-ENTROPY -BACKOFF -TEST -FILES'
    else:
        options = '-ENTROPY -BACKOFF -TEST -CACHE -CACHE_ORDER 3 -CACHE_DYNAMIC_LAMBDA -FILE_CACHE -FILES'
    # options_window5000 = '-ENTROPY -BACKOFF -TEST -CACHE -CACHE_ORDER 3 -CACHE_DYNAMIC_LAMBDA -WINDOW_CACHE -WINDOW_SIZE 5000 -FILES'

    completion = 'entropy_compute/code/completion'
    scope_file = 'entropy_compute/data/trainset/fold0.train.scope'
    grams_file = 'entropy_compute/data/trainset/fold0.train.3grams'  # n-grams file

    tmp_dir = "dataset_vul/newALLBUGS/tmp/tmp_function/"
    tmp_test = "dataset_vul/newALLBUGS/tmp/test_function/"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(tmp_test, exist_ok=True)
    parser_js = "solidity-extractor/function.js"

    cp = subprocess.run("cd {} && node {} {} >/dev/null".format(tmp_dir, parser_js, contract_path), shell=True, stdout=subprocess.PIPE)
    if cp.returncode:
        return -9999
    name = contract_path.split('/')[-1]
    tmp_path = tmp_dir + name
    with open(tmp_path) as f:
        codestring = f.read()

    tree = SolidityParser(CommonTokenStream(SolidityLexer(InputStream(codestring)))).functionDefinition()
    output = tree.toCodeSequence()
    regex = r'(\[)[0-9\s]*(\])'
    output2 = re.sub(regex, '', output)
    output3 = output2.split(' ')
    tokens = [char for char in output3 if char != '' and char != '<EOF>']
    sourceCode = " ".join(tokens)
    with open(tmp_path, 'w') as f:
        f.write(sourceCode)

    test_file = tmp_test + name
    with open(test_file, 'w') as f2:
        f2.write(tmp_path)

    # compute entropy
    order = 3
    cp2 = subprocess.run('{} {} -NGRAM_FILE {} -NGRAM_ORDER {} -SCOPE_FILE {} -INPUT_FILE {}'.format
                         (completion, options, grams_file, order, scope_file, test_file), shell=True, stdout=subprocess.PIPE)
    if cp2.returncode:
        raise (IOError, 'code/completion fail')
    lines = cp2.stdout.decode().split('\n')
    for line in lines:
        if 'Entropy: ' in line:
            return float(line.strip('Entropy: '))


if __name__ == '__main__':
    import numpy as np
    es = []
    dir1 = 'dataset_vul/classify2/sum/'
    for name in os.listdir(dir1):
        path = dir1 + name
        e = get_entropy(path, True)
        es.append(e)
        print(name, e)
    emean = np.mean(es)

    es2 = []
    for name in os.listdir(dir1):
        path = dir1 + name
        e = get_entropy(path, False)
        es2.append(e)
        print(name, e)
    emean2 = np.mean(es2)
