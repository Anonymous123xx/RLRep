from utils2 import *
import random

def get_fault_line(_path) -> list:
    with open(_path) as f:
        lines = f.readlines()
    bugline = finds(_path)[-1]
    fault_code_sequence = lines[bugline].strip()
    parser = SolidityParser(CommonTokenStream(SolidityLexer(InputStream(fault_code_sequence))))
    tree = parser.sourceUnit()
    code_line = tree.toCodeSequence()
    regex = r'(\[)[0-9\s]*(\])'
    code_line = re.sub(regex, '', code_line)
    regex2 = "<missing [^>]+>"
    code_line = re.sub(regex2, '', code_line)
    tokenSeq = [x for x in code_line.split(' ') if not x == '' and not x == '<EOF>']
    return tokenSeq

def finds(contract_path):
    with open(contract_path) as cf:
        org_lines = cf.readlines()
    bugline = -1
    for index, line in enumerate(org_lines):
        if '// fault line' in line:
            bugline = index
    if bugline == -1:
        raise Exception
    preceding = org_lines[:bugline]
    following = org_lines[bugline+1:]
    return preceding, following, bugline

def compile_ok(repair_path):
    with open(repair_path) as f:
        lines = f.readlines()
    try:
        for line in lines:
            if 'pragma solidity' in line:
                num = re.compile(r'(\d+)\s*\.\s*(\d+)\s*\.\s*(\d+)')
                v1, v2, v3 = re.search(num, line).group(1), re.search(num, line).group(2), re.search(num, line).group(3)
                version = v1 + '.' + v2 + '.' + v3
                break
        solcx.compile_files(repair_path, solc_version=version)
        return True
    except:
        return False


def insert_token(mutation_path, token_w2i, gen, logger):
    logger.info('inserting ...')
    name = os.path.split(mutation_path)[1].split('.sol')[0]
    addr = name.split('__')[0]
    repair_dir = 'dataset_vul/newALLBUGS/validation/genetic/{}/{}/'.format(addr, gen)
    preceding, following, _ = finds(mutation_path)
    code = get_fault_line(mutation_path)

    n = 0
    for i in range(len(code)):
        for j in token_w2i.keys():
            n += 1
            new_code = code[:]
            new_code.insert(i, j)
            new_code = [' '.join(new_code)]
            new_code = [c.strip() + '\n' for c in new_code]
            new_code[0] = new_code[0].strip() + '  // fault line\n'
            new_code = preceding + new_code + following
            repair_path = repair_dir + name + '__insert' + str(n) + '.sol'
            with open(repair_path, 'w') as newf:
                for line in new_code:
                    newf.write(line)
            if compile_ok(repair_path) == False:
                os.remove(repair_path)

def replace_token(mutation_path, token_w2i, gen, logger):
    logger.info('replacing ...')
    name = os.path.split(mutation_path)[1].split('.sol')[0]
    addr = name.split('__')[0]
    repair_dir = 'dataset_vul/newALLBUGS/validation/genetic/{}/{}/'.format(addr, gen)
    preceding, following, _ = finds(mutation_path)
    code = get_fault_line(mutation_path)

    n = 0
    for i in range(len(code)):
        for j in token_w2i.keys():
            n += 1
            new_code = code[:]
            if new_code[i] != j:
                new_code[i] = j
            else:
                continue
            new_code = [' '.join(new_code)]
            new_code = [c.strip() + '\n' for c in new_code]
            new_code[0] = new_code[0].strip() + '  // fault line\n'
            new_code = preceding + new_code + following
            repair_path = repair_dir + name + '__replace' + str(n) + '.sol'
            with open(repair_path, 'w') as newf:
                for line in new_code:
                    newf.write(line)
            if compile_ok(repair_path) == False:
                os.remove(repair_path)

def move_line(mutation_path, gen, logger):
    logger.info('moving ...')
    name = os.path.split(mutation_path)[1].split('.sol')[0]
    addr = name.split('__')[0]
    repair_dir = 'dataset_vul/newALLBUGS/validation/genetic/{}/{}/'.format(addr, gen)
    with open(mutation_path) as f:
        lines = f.readlines()
    bug_index = finds(mutation_path)[-1]
    lines1 = lines[:]
    repair_path1 = repair_dir + name + '__move1' + '.sol'
    lines1[bug_index], lines1[bug_index-1] = lines1[bug_index-1], lines1[bug_index]
    with open(repair_path1, 'w') as f1:
        for line in lines1:
            f1.write(line)
    if compile_ok(repair_path1) == False:
        os.remove(repair_path1)
    lines2 = lines[:]
    repair_path2 = repair_dir + name + '__move2' + '.sol'
    lines2[bug_index], lines2[bug_index+1] = lines2[bug_index+1], lines2[bug_index]
    with open(repair_path2, 'w') as f1:
        for line in lines2:
            f1.write(line)
    if compile_ok(repair_path2) == False:
        os.remove(repair_path2)

def insert_move_token(mutation_path, token_w2i, gen, logger):
    logger.info('inserting, moving ...')
    name = os.path.split(mutation_path)[1].split('.sol')[0]
    addr = name.split('__')[0]
    repair_dir = 'dataset_vul/newALLBUGS/validation/genetic/{}/{}/'.format(addr, gen)
    preceding, following, bug_index = finds(mutation_path)
    code = get_fault_line(mutation_path)

    n = 0
    for i in range(len(code)):
        for j in token_w2i.keys():
            n += 1
            new_code = code[:]
            new_code.insert(i, j)
            new_code = [' '.join(new_code)]
            new_code = [c.strip() + '\n' for c in new_code]
            new_code[0] = new_code[0].strip() + '  // fault line\n'
            new_code = preceding + new_code + following

            new_code1 = new_code[:]
            new_code1[bug_index], new_code1[bug_index-1] = new_code1[bug_index-1], new_code1[bug_index]
            repair_path = repair_dir + name + '__insert' + str(n) + '__move1' + '.sol'
            with open(repair_path, 'w') as newf:
                for line in new_code1:
                    newf.write(line)
            if compile_ok(repair_path) == False:
                os.remove(repair_path)

            new_code2 = new_code[:]
            new_code2[bug_index], new_code2[bug_index+1] = new_code2[bug_index+1], new_code2[bug_index]
            repair_path = repair_dir + name + '__insert' + str(n) + '__move2' + '.sol'
            with open(repair_path, 'w') as newf:
                for line in new_code2:
                    newf.write(line)
            if compile_ok(repair_path) == False:
                os.remove(repair_path)

def replace_move_token(mutation_path, token_w2i, gen, logger):
    logger.info('replacing, moving ...')
    name = os.path.split(mutation_path)[1].split('.sol')[0]
    addr = name.split('__')[0]
    repair_dir = 'dataset_vul/newALLBUGS/validation/genetic/{}/{}/'.format(addr, gen)
    preceding, following, bug_index = finds(mutation_path)
    code = get_fault_line(mutation_path)

    n = 0
    for i in range(len(code)):
        for j in token_w2i.keys():
            n += 1
            new_code = code[:]
            if new_code[i] != j:
                new_code[i] = j
            else:
                continue
            new_code = [' '.join(new_code)]
            new_code = [c.strip() + '\n' for c in new_code]
            new_code[0] = new_code[0].strip() + '  // fault line\n'
            new_code = preceding + new_code + following
            new_code1 = new_code[:]
            new_code1[bug_index], new_code1[bug_index-1] = new_code1[bug_index-1], new_code1[bug_index]
            repair_path = repair_dir + name + '__repmov' + str(n) + '.sol'
            with open(repair_path, 'w') as newf:
                for line in new_code1:
                    newf.write(line)
            if compile_ok(repair_path) == False:
                os.remove(repair_path)
            new_code2 = new_code[:]
            new_code2[bug_index], new_code2[bug_index+1] = new_code2[bug_index+1], new_code2[bug_index]
            repair_path = repair_dir + name + '__repmov' + str(n) + '.sol'
            with open(repair_path, 'w') as newf:
                for line in new_code2:
                    newf.write(line)
            if compile_ok(repair_path) == False:
                os.remove(repair_path)

def insert_replace_token(mutation_path, token_i2w, gen, logger):
    logger.info('inserting, replacing...')
    name = os.path.split(mutation_path)[1].split('.sol')[0]
    addr = name.split('__')[0]
    repair_dir = 'dataset_vul/newALLBUGS/validation/genetic/{}/{}/'.format(addr, gen)
    preceding, following, bug_index = finds(mutation_path)
    code = get_fault_line(mutation_path)

    n = 0
    while n <= 10000:
        n += 1
        new_code = code[:]
        i = random.randint(0, len(code))
        j_index = random.randint(0, len(token_i2w)-1)
        j = token_i2w[j_index]
        new_code.insert(i,j)
        i = random.randint(0, len(code))
        j_index = random.randint(0, len(token_i2w)-1)
        j = token_i2w[j_index]
        if new_code[i] != j:
            new_code[i] = j
        else:
            continue
        new_code = [' '.join(new_code)]
        new_code = [c.strip() + '\n' for c in new_code]
        new_code[0] = new_code[0].strip() + '  // fault line\n'
        new_code = preceding + new_code + following
        repair_path = repair_dir + name + '__insrep' + str(n) + '.sol'
        with open(repair_path, 'w') as newf:
            for line in new_code:
                newf.write(line)
        if compile_ok(repair_path) == False:
            os.remove(repair_path)

def insert_replace_move_token(mutation_path, token_i2w, gen, logger):
    logger.info('replacing, moving, inserting...')
    name = os.path.split(mutation_path)[1].split('.sol')[0]
    addr = name.split('__')[0]
    repair_dir = 'dataset_vul/newALLBUGS/validation/genetic/{}/{}/'.format(addr, gen)
    preceding, following, bug_index = finds(mutation_path)
    code = get_fault_line(mutation_path)

    n = 0
    while n <= 10000:
        n += 1
        new_code = code[:]
        i = random.randint(0, len(code))
        j_index = random.randint(0, len(token_i2w)-1)
        j = token_i2w[j_index]
        new_code.insert(i, j)
        i = random.randint(0, len(code))
        j_index = random.randint(0, len(token_i2w)-1)
        j = token_i2w[j_index]
        if new_code[i] != j:
            new_code[i] = j
        else:
            continue
        new_code = [' '.join(new_code)]
        new_code = [c.strip() + '\n' for c in new_code]
        new_code[0] = new_code[0].strip() + '  // fault line\n'
        new_code = preceding + new_code + following
        new_code1 = new_code[:]
        new_code1[bug_index], new_code1[bug_index-1] = new_code1[bug_index-1], new_code1[bug_index]
        repair_path = repair_dir + name + '__insrepmov' + str(n) + '.sol'
        with open(repair_path, 'w') as newf:
            for line in new_code1:
                newf.write(line)
        if compile_ok(repair_path) == False:
            os.remove(repair_path)
        new_code2 = new_code[:]
        new_code2[bug_index], new_code2[bug_index+1] = new_code2[bug_index+1], new_code2[bug_index]
        repair_path = repair_dir + name + '__insrepmov' + str(n) + '.sol'
        with open(repair_path, 'w') as newf:
            for line in new_code2:
                newf.write(line)
        if compile_ok(repair_path) == False:
            os.remove(repair_path)

def mutation_token(mutation_path, code_w2i, code_i2w, gen, logger):
    insert_token(mutation_path, code_w2i, gen, logger)
    replace_token(mutation_path, code_w2i, gen, logger)
    move_line(mutation_path, gen, logger)
    insert_move_token(mutation_path, code_w2i, gen, logger)
    replace_move_token(mutation_path, code_w2i, gen, logger)
    insert_replace_token(mutation_path, code_i2w, gen, logger)
    insert_replace_move_token(mutation_path, code_i2w, gen, logger)


def fitness_function(compilable_contract_dir, original_contract, logger):
    patch = {}
    patch_remove = []
    for compile_contract_name in os.listdir(compilable_contract_dir):
        compilable_contract = compilable_contract_dir + compile_contract_name
        reward = 0
        error, now_error = detect(original_contract, compilable_contract, 60, trainset=False)
        if error == -1 or now_error == -1:
            os.remove(compilable_contract)
            continue
        else:
            if now_error <= error:
                reward += 0.025
            else:
                reward -= 0.025
                if compilable_contract not in patch_remove:
                    patch_remove.append(compilable_contract)
        # similarity compute
        contract_sims = get_similarity(original_contract, first=False)
        repair_sims = get_similarity(compilable_contract, first=False)
        if repair_sims < contract_sims:
            reward += 0.014
        else:
            reward -= 0.014
            if compilable_contract not in patch_remove:
                patch_remove.append(compilable_contract)
        # entropy compute
        original_entropy = get_entropy(original_contract, cache, first=False)
        repair_entropy = get_entropy(compilable_contract, cache, first=False)

        if repair_entropy == -9999:
            logger.info('error: parser.js'.format(compilable_contract))
            if compilable_contract not in patch_remove:
                patch_remove.append(compilable_contract)
            patch[compilable_contract] = -9999
            continue

        if abs(original_entropy-3.2) > abs(repair_entropy-3.2):
            reward += 0.014
        else:
            reward -= 0.014
            if compilable_contract not in patch_remove:
                patch_remove.append(compilable_contract)

        patch[compilable_contract] = reward
    return patch, patch_remove

