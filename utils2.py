import logging, pickle
import random, shutil, solcx
from smartBugs import *
from similarity_compute import *
from entropy_compute import get_entropy


def get_logger(paths):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(paths)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def code_ast2index(in_w2i, config, path_code, path_ast):
    in1_w2i, in2_w2i = in_w2i
    for tup in get_one(path_code, path_ast):
        in1, in2, contract_name = tup
        in1 = [in1_w2i[x] if x in in1_w2i else in1_w2i['<UNK>'] for x in in1]
        in2 = [in2_w2i[x] if x in in2_w2i else in2_w2i['<UNK>'] for x in in2]
        in1 = in1[:config.MAX_INPUT_SIZE]
        in2 = in2[:config.MAX_INPUT_SIZE]
        if len(in1) == 0 or len(in2) == 0:
            continue
        yield in1, in2, contract_name

def get_one(path_code, path_ast):
    name_list = [name for name in os.listdir(path_code) if name.endswith('.sol')]
    for name in name_list:
        function_file = os.path.join(path_code, name)
        tree = SolidityParser(CommonTokenStream(SolidityLexer(FileStream(function_file)))).functionDefinition()
        output = tree.toCodeSequence()
        regex = r'(\[)[0-9\s]*(\])'
        output2 = re.sub(regex, '', output)
        output3 = output2.split(' ')
        code_tokens_list = [token for token in output3 if token != '' and token != '<EOF>']
        in_code = code_tokens_list
        with open(os.path.join(path_ast, name)) as f:
            ast_token_seq = f.read()
        in_ast = ast_token_seq.split(' ')
        addr = name.rstrip('.sol')
        yield in_code, in_ast, addr

def get_batch(path_code, path_ast, config, in_w2i, pretrain):
    tag_path = 'dataset_vul/newALLBUGS/pretrain_label/label190.pkl'
    with open(tag_path, 'rb') as label:
        label_dict = pickle.load(label)
    batch_size = config.BATCH_SIZE if pretrain == False else config.PRE_BATCH_SIZE
    batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    for code_sequence, ast_sequence, contract in code_ast2index(in_w2i, config, path_code, path_ast):
        batch_in1.append(code_sequence)
        batch_in2.append(ast_sequence)
        batch_in3.append(contract)
        if len(batch_in1) >= batch_size:
            if pretrain:
                batch_out = [label_dict[contract_name] for contract_name in batch_in3]
                yield batch_in1, batch_in2, batch_in3, batch_out
            else:
                yield batch_in1, batch_in2, batch_in3
            batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    if len(batch_in1) > 0:
        if pretrain:
            batch_out = [label_dict[contract_name] for contract_name in batch_in3]
            yield batch_in1, batch_in2, batch_in3, batch_out
        else:
            yield batch_in1, batch_in2, batch_in3

def fitness_function2(original_contract, repair_contract) -> float:
    reward = 0
    with open(repair_contract) as f:
        repair_contract_code = f.read()
    with open(original_contract) as f:
        original_contract_code = f.read()
    repair_contract_list = repair_contract_code.split('\n')
    try:
        for line in repair_contract_list:
            if 'pragma solidity' in line:
                num = re.compile(r'(\d+)\s*\.\s*(\d+)\s*\.\s*(\d+)')
                v1, v2, v3 = re.search(num, line).group(1), re.search(num, line).group(2), re.search(num, line).group(3)
                version = v1 + '.' + v2 + '.' + v3
                break
        solcx.compile_files(repair_contract, solc_version=version)
    except:
        reward -= 0.02
        with open(repair_contract, 'w') as f:
            f.write(original_contract_code)
        return reward
    error, now_error = detect(original_contract, repair_contract, 60)
    if error == -1 or now_error == -1:
        pass
    else:
        if now_error < error:
            reward += 0.025
        else:
            reward -= 0.025
    cache = True
    original_entropy = get_entropy(original_contract, cache, first=False)
    repair_entropy = get_entropy(repair_contract, cache, first=False)
    if abs(original_entropy - 3.2) > abs(repair_entropy - 3.2):
        reward += 0.02
    else:
        reward -= 0.02
    if reward > 1:
        reward = 1
    if reward == 0.0:
        reward = -0.0001
    return reward

def get_action():
    action_map = [
        ### operator mutation
        'replace + to -',
        'replace + to *',
        'replace + to /',
        'replace - to +',
        'replace - to *',
        'replace - to /',
        'replace * to +',
        'replace * to -',
        'replace * to /',
        'replace / to +',
        'replace / to -',
        'replace / to *',
        'replace > to <',
        'replace > to >=',
        'replace > to <=',
        'replace < to >',
        'replace < to >=',
        'replace < to <=',
        'replace >= to >',
        'replace >= to <',
        'replace >= to <=',
        'replace <= to >',
        'replace <= to <',
        'replace <= to >=',
        'replace && to ||',
        'replace || to &&',
        'replace == to !=',
        'replace != to ==',
        'replace == to >=',
        'replace == to <=',
        'replace != to >=',
        'replace != to <=',
        ### operand mutation
        'replace ope1 to ope2',
        'replace ope1 to ope3',
        'replace ope1 to ope1+1',
        'replace ope1 to 0',
        'replace 0 to ope1',
        ### data location keyword replacement
        'replace storage to memory',
        'replace memory to storage',
        ### address variable replacement
        'replace tx.origin to msg.sender',
        ### transfer function replacement
        'replace call.value to transfer',
        'replace call.value to send',
        'replace send to transfer',
        'replace call to transfer',
        ### variable type keyword replacement
        'replace uint* to uint64',
        'replace uint* to uint256',
        'replace bytes* to bytes8',
        'replace bytes* to bytes32',
        ### ether unit replacement
        'replace wei to finney',
        'replace wei to szabo',
        'replace wei to Ether',
        'replace finney to Ether',
        'replace finney to szabo',
        'replace finney to wei',
        'replace szabo to Ether',
        'replace szabo to wei',
        'replace szabo to finney',
        'replace Ether to szabo',
        'replace Ether to wei',
        'replace Ether to finney',
        ### time unit replacement
        'replace seconds to minutes',
        'replace seconds to hours',
        'replace seconds to days',
        'replace seconds to weeks',
        'replace minutes to seconds',
        'replace minutes to hours',
        'replace minutes to days',
        'replace minutes to weeks',
        'replace hours to seconds',
        'replace hours to minutes',
        'replace hours to days',
        'replace hours to weeks',
        'replace weeks to seconds',
        'replace weeks to minutes',
        'replace weeks to hours',
        'replace weeks to days',
        'replace days to seconds',
        'replace days to minutes',
        'replace days to hours',
        'replace days to weeks',
        ### revert/assert/require injection
        'insert revert(statement)',
        'insert assert(statement)',
        'insert require(statement)',
        ### access domain keyword replacement
        'replace public to internal',
        'replace public to private',
        'replace public to external',
        'replace external to internal',
        'replace external to private',
        'replace external to public',
        ### require statement
        'insert require(ope1!=0)',
        'insert require(ope1!=0)',
        ### function state keyword
        'add function state keyword view',
        'add function state keyword pure',
        'function state keyword change',
        ### conditional criterion
        'add conditional criterion && ope1 of require',
        'add conditional criterion || ope1 of require',
        'add conditional criterion && ope1 of if',
        'add conditional criterion || ope1 of if',
        'add conditional criterion && ope1 of assert',
        'add conditional criterion || ope1 of assert',
        ### move bug line and next line
        'move internal state changes',
        'endtoken',
    ]
    return action_map

def insert_bugline(lines, bugline, newline, order):
    if order == "after":
        newlines = lines[:bugline+1]
        newlines.append(newline)
        newlines.extend(lines[bugline+1:])
    if order == "before":
        newlines = lines[:bugline]
        newlines.append(newline)
        newlines.extend(lines[bugline:])
    return newlines

def write_newlines(repair_path, wlines):
    with open(repair_path, 'w') as rf:
        for line in wlines:
            rf.write(line)

def replace_random(line, str1, str2):
    if len(str1) is 1:
        i = 0
        matchs = []
        while i < len(line):
            if line[i] is str1:
                if line[i+1] is str1:
                    i += 2
                    continue
                else:
                    matchs.append(i)
            i += 1
        index = matchs[random.randint(0, len(matchs)-1)]
        _newline = line[:index] + str2 + line[index+1:]
        return _newline

    elif len(str1) is 2:
        i = 0
        matchs = []
        while i < len(line):
            if line[i] is str1[0] and line[i+1] is str1[1]:
                matchs.append(i)
                i += 2
            else:
                i += 1
        index = matchs[random.randint(0, len(matchs)-1)]
        _newline = line[:index] + str2 + line[index+2:]
        return _newline

def find_lines(lines, fault, _path) -> list:
    targetline = []
    if fault:
        string = '// fault line'
    else:
        string = '// fixed line'
    for i, line in enumerate(lines):
        if string in line:
            targetline.append(i)
    if fault and not targetline:
        raise Exception('"{}" has no fault line'.format(_path))
    return targetline

def choose_action(contract, action_nums, train, gitdif=False) -> float:
    fail = -0.03
    fail2 = -0.03
    action_map = get_action()
    repair_dir = 'dataset_vul/newALLBUGS/repair_contract/'
    if train == True:
        contract_dir = 'dataset_vul/newALLBUGS/contract/'
    else:
        contract_dir = 'dataset_vul/newALLBUGS/validation/contract/'
    contract_path = contract_dir + contract + '.sol'
    repair_path = repair_dir + contract + '.sol'
    shutil.copy(contract_path, repair_path)
    with open(repair_path) as sf:
        lines = sf.readlines()

    if action_nums == []:
        write_newlines(repair_path, lines)
        return fail2

    for number, action_num in enumerate(action_nums):
        action = action_map[action_num]
        bugline = find_lines(lines, True, repair_path)[0]
        fixline_list = find_lines(lines, False, repair_path)
        com = re.compile(
            r'\s*([^()\s+\-=\*]+)\s*(\+\+|-=|\*=|\\=|\+=|-=|-|\*|\\|=|\+)\s*([^()\s+\-=\*;]+)*\s*(\++|-=|\*=|\\=|\+=|-=|-|\*|\\|=|\+)*\s*([^()\s+\-=\*;]+)*'
        )
        matches_bugline = re.search(com, lines[bugline])

        if 0 <= action_num <= 31:
            action = action.split(' ')
            tglines = [line for line in fixline_list if action[1] in lines[line]]
            if action[1] in lines[bugline].split('// fault')[0]:
                tglines.append(bugline)
            if not tglines:
                write_newlines(repair_path, lines)
                return fail
            random_index = random.randint(0, len(tglines)-1)
            tmp = lines[tglines[random_index]].split("// fixed line")[0] if '// fixed' in lines[tglines[random_index]] else lines[tglines[random_index]].split("// fault line")[0]
            try:
                tmp2 = replace_random(tmp, action[1], action[3])
            except:
                return fail
            lines[tglines[random_index]] = tmp2 + "// fixed line\n" if '// fixed' in lines[tglines[random_index]] else tmp2 + "// fault line\n"


        if action_num == 32:
            try:
                ope1 = matches_bugline.group(1)
                ope2 = matches_bugline.group(3)
                if ope2 is None:
                    write_newlines(repair_path, lines)
                    return fail
                tglines = [line for line in fixline_list if ope1 in lines[line]]
                if len(tglines) > 0:
                    random_index = random.randint(0, len(tglines)-1)
                    tmp = lines[tglines[random_index]].split("// fixed line")[0] if '// fixed' in lines[tglines[random_index]] else lines[tglines[random_index]].split("// fault line")[0]
                    tmp2 = tmp.replace(ope1, ope2, 1)
                    lines[tglines[random_index]] = tmp2 + "// fixed line\n" if '// fixed' in lines[tglines[random_index]] else tmp2 + "// fault line\n"
                else:
                    tmp = lines[bugline].split("// fixed line")[0] if '// fixed' in lines[bugline] else lines[bugline].split("// fault line")[0]
                    tmp2 = tmp.replace(ope1, ope2, 1)
                    lines[bugline] = tmp2 + "// fixed line\n" if '// fixed' in lines[bugline] else tmp2 + "// fault line\n"
            except:
                write_newlines(repair_path, lines)
                return fail

        if action_num == 33:
            try:
                ope1 = matches_bugline.group(1)
                ope3 = matches_bugline.group(5)
                if ope3 is None:
                    write_newlines(repair_path, lines)
                    return fail
                tglines = [line for line in fixline_list if ope1 in lines[line]]
                if len(tglines) > 0:
                    random_index = random.randint(0, len(tglines) - 1)
                    tmp = lines[tglines[random_index]].split("// fixed line")[0] if '// fixed' in lines[tglines[random_index]] else lines[tglines[random_index]].split("// fault line")[0]
                    tmp2 = tmp.replace(ope1, ope3, 1)
                    lines[tglines[random_index]] = tmp2 + "// fixed line\n" if '// fixed' in lines[tglines[random_index]] else tmp2 + "// fault line\n"
                else:
                    tglines.append(bugline)
                    random_index = random.randint(0, len(tglines)-1)
                    tmp = lines[tglines[random_index]].split("// fixed line")[0] if '// fixed' in lines[tglines[random_index]] else lines[tglines[random_index]].split("// fault line")[0]
                    tmp2 = tmp.replace(ope1, ope3, 1)
                    lines[tglines[random_index]] = tmp2 + "// fixed line\n" if '// fixed' in lines[tglines[random_index]] else tmp2 + "// fault line\n"
            except:
                write_newlines(repair_path, lines)
                return fail

        if action_num == 34:
            try:
                ope1 = matches_bugline.group(1)
                tglines = [line for line in fixline_list if ope1 in lines[line]]
                if len(tglines) > 0:
                    random_index = random.randint(0, len(tglines) - 1)
                    tmp = lines[tglines[random_index]].split("// fixed line")[0] if '// fixed' in lines[tglines[random_index]] else lines[tglines[random_index]].split("// fault line")[0]
                    tmp2 = tmp.replace(ope1, ope1 + '+1', 1)
                    lines[tglines[random_index]] = tmp2 + "// fixed line\n" if '// fixed' in lines[tglines[random_index]] else tmp2 + "// fault line\n"
                else:
                    tglines.append(bugline)
                    random_index = random.randint(0, len(tglines)-1)
                    tmp = lines[tglines[random_index]].split("// fixed line")[0] if '// fixed' in lines[tglines[random_index]] else lines[tglines[random_index]].split("// fault line")[0]
                    tmp2 = tmp.replace(ope1, ope1+'+1', 1)
                    lines[tglines[random_index]] = tmp2 + "// fixed line\n" if '// fixed' in lines[tglines[random_index]] else tmp2 + "// fault line\n"
            except:
                write_newlines(repair_path, lines)
                return fail

        if action_num == 35:
            try:
                ope1 = matches_bugline.group(1)
                tglines = [line for line in fixline_list if ope1 in lines[line]]
                if len(tglines) > 0:
                    random_index = random.randint(0, len(tglines)-1)
                    tmp = lines[tglines[random_index]].split("// fixed line")[0] if '// fixed' in lines[tglines[random_index]] else lines[tglines[random_index]].split("// fault line")[0]
                    tmp2 = tmp.replace(ope1, '0', 1)
                    lines[tglines[random_index]] = tmp2 + "// fixed line\n" if '// fixed' in lines[tglines[random_index]] else tmp2 + "// fault line\n"
                else:
                    tglines.append(bugline)
                    random_index = random.randint(0, len(tglines)-1)
                    tmp = lines[tglines[random_index]].split("// fixed line")[0] if '// fixed' in lines[tglines[random_index]] else lines[tglines[random_index]].split("// fault line")[0]
                    tmp2 = tmp.replace(ope1, '0', 1)
                    lines[tglines[random_index]] = tmp2 + "// fixed line\n" if '// fixed' in lines[tglines[random_index]] else tmp2 + "// fault line\n"
            except:
                write_newlines(repair_path, lines)
                return fail

        if action_num == 36:
            try:
                ope1 = matches_bugline.group(1)
                fixline_list.append(bugline)
                zero_com = re.compile(r'(\b(0)\b)')
                tglines = [line for line in fixline_list if re.search(zero_com, lines[line])]
                if len(tglines) > 0:
                    random_index = random.randint(0, len(tglines)-1)
                    tmp = lines[tglines[random_index]].split("// fixed line")[0] if '// fixed' in lines[tglines[random_index]] else lines[tglines[random_index]].split("// fault line")[0]
                    tmp2 = re.sub(zero_com, ope1, tmp, 1)
                    lines[tglines[random_index]] = tmp2 + "// fixed line\n" if '// fixed' in lines[tglines[random_index]] else tmp2 + "// fault line\n"
                elif re.search(zero_com, lines[bugline]):
                    tglines.append(bugline)
                    random_index = random.randint(0, len(tglines)-1)
                    tmp = lines[tglines[random_index]].split("// fixed line")[0] if '// fixed' in lines[tglines[random_index]] else lines[tglines[random_index]].split("// fault line")[0]
                    tmp2 = re.sub(zero_com, ope1, tmp, 1)
                    lines[tglines[random_index]] = tmp2 + "// fixed line\n" if '// fixed' in lines[tglines[random_index]] else tmp2 + "// fault line\n"
                else:
                    write_newlines(repair_path, lines)
                    return fail
            except:
                write_newlines(repair_path, lines)
                return fail


        if action_num == 37:
            tmp = lines[bugline].split("// fault line")[0]
            storage_com = re.compile(r'\bstorage\b')
            if not re.search(storage_com, tmp):
                write_newlines(repair_path, lines)
                return fail
            tmp2 = re.sub(storage_com, 'memory', tmp, 1)
            lines[bugline] = tmp2 + "// fault line\n"


        if 38 <= action_num <= 39:
            action = action.split(' ')
            tmp = lines[bugline].split("// fault line")[0]
            if action[1] not in lines[bugline]:
                write_newlines(repair_path, lines)
                return fail
            tmp2 = tmp.replace(action[1], action[3], 1)
            lines[bugline] = tmp2 + "// fault line\n"


        if action_num == 40:
            try:
                com = re.compile(r'.*(call\.value\s*\(([^)]*)\)\s*\(\s*\))')
                matches = re.match(com, lines[bugline])
                string, address = matches.group(1), matches.group(2)
                parts = lines[bugline].split(string)
                newline = parts[0] + 'transfer ({})'.format(address) + parts[1]
                lines[bugline] = newline
            except:
                write_newlines(repair_path, lines)
                return fail

        if action_num == 41:
            try:
                com = re.compile(r'.*(call\.value\s*\(([^)]*)\)\s*\(\s*\))')
                matches = re.match(com, lines[bugline])
                string, address = matches.group(1), matches.group(2)
                parts = lines[bugline].split(string)
                newline = parts[0] + 'send ({})'.format(address) + parts[1]
                lines[bugline] = newline
            except:
                write_newlines(repair_path, lines)
                return fail

        if action_num == 42:
            send_com = re.compile(r'\.\s*send\s*\(')
            if not re.search(send_com, lines[bugline]):
                write_newlines(repair_path, lines)
                return fail
            lines[bugline] = re.sub(send_com, '.transfer(', lines[bugline], 1)

        if action_num == 43:
            call_com = re.compile(r'\.\s*call\s*\(')
            if not re.search(call_com, lines[bugline]):
                write_newlines(repair_path, lines)
                return fail
            lines[bugline] = re.sub(call_com, '.transfer(', lines[bugline], 1)


        if action_num == 44:
            try:
                uintX = re.search(re.compile(r'(uint(\d+))'), lines[bugline])
                if not int(uintX.group(2)) < 64:
                    write_newlines(repair_path, lines)
                    return fail
                lines[bugline] = lines[bugline].replace(uintX.group(1), 'uint64', 1)
            except:
                write_newlines(repair_path, lines)
                return fail

        if action_num == 45:
            try:
                uintX = re.search(re.compile(r'(uint(\d+))'), lines[bugline])
                if not 64 <= int(uintX.group(2)) < 256:
                    write_newlines(repair_path, lines)
                    return fail
                lines[bugline] = lines[bugline].replace(uintX.group(1), 'uint256', 1)
            except:
                write_newlines(repair_path, lines)
                return fail

        if action_num == 46:
            try:
                bytesX = re.search(re.compile(r'(bytes(\d+))'), lines[bugline])
                if not 0 < int(bytesX.group(2)) < 8:
                    write_newlines(repair_path, lines)
                    return fail
                lines[bugline] = lines[bugline].replace(bytesX.group(1), 'bytes8', 1)
            except:
                write_newlines(repair_path, lines)
                return fail

        if action_num == 47:
            try:
                bytesX = re.search(re.compile(r'(bytes(\d+))'), lines[bugline])
                if not 8 <= int(bytesX.group(2)) < 32:
                    write_newlines(repair_path, lines)
                    return fail
                lines[bugline] = lines[bugline].replace(bytesX.group(1), 'bytes32', 1)
            except:
                write_newlines(repair_path, lines)
                return fail

        if 48 <= action_num <= 59:
            action = action.split(' ')
            if not action[1] in lines[bugline]:
                write_newlines(repair_path, lines)
                return fail
            lines[bugline] = lines[bugline].replace(action[1], action[3])


        if 60 <= action_num <= 79:
            action = action.split(' ')
            if not action[1] in lines[bugline]:
                write_newlines(repair_path, lines)
                return fail
            lines[bugline] = lines[bugline].replace(action[1], action[3])


        if action_num == 80:
            lines[bugline] = 'if( !({}) ) revert;  // fault line\n'.format(lines[bugline].strip('\n').strip().split('// fault line')[0].strip().strip(';'))

        if action_num == 81:
            lines[bugline] = 'assert( {} );  // fault line\n'.format(lines[bugline].strip('\n').strip().split('// fault line')[0].strip().strip(';'))

        if action_num == 82:
            lines[bugline] = 'require( {} );  // fault line\n'.format(lines[bugline].strip('\n').strip().split('// fault line')[0].strip().strip(';'))


        if 83 <= action_num <= 88:
            action = action.split(' ')
            repla = ' {} '.format(action[1])
            if repla not in lines[bugline]:
                write_newlines(repair_path, lines)
                return fail
            lines[bugline] = lines[bugline].replace(action[1], action[3], 1)


        if 89 <= action_num <= 90:
            try:
                ope1 = matches_bugline.group(1)
                if action_num == 89:
                    newline = 'require({} != 0);  // fixed line\n'.format(ope1)
                    lines = insert_bugline(lines, bugline, newline, "after")
                if action_num == 90:
                    newline = 'require({} != 0);  // fixed line\n'.format(ope1)
                    lines = insert_bugline(lines, bugline, newline, "before")
            except:
                write_newlines(repair_path, lines)
                return fail


        if 91 <= action_num <= 92:
            try:
                if 'view' not in lines[bugline] and 'pure' not in lines[bugline] and 'constant' not in lines[bugline]:
                    f_matches = re.search(re.compile(r'(\s*function\s*[^(]*(\([^()]*\)))'), lines[bugline])
                    part1 = f_matches.group(1)
                    part2 = lines[bugline].split(part1)[-1]
                    if action_num == 91:
                        lines[bugline] = part1 + ' view' + part2
                    if action_num == 92:
                        lines[bugline] = part1 + ' pure' + part2
                else:
                    write_newlines(repair_path, lines)
                    return fail
            except:
                write_newlines(repair_path, lines)
                return fail

        if action_num == 93:
            tmp = lines[bugline].split("// fault line")[0]
            storage_com = re.compile(r'\bview\b')
            if not re.search(storage_com, tmp):
                write_newlines(repair_path, lines)
                return fail
            tmp2 = re.sub(storage_com, 'pure', tmp, 1)
            lines[bugline] = tmp2 + "// fault line\n"


        if 94 <= action_num <= 95:
            try:
                ope1 = matches_bugline.group(1)
                fixline_list.append(bugline)
                tglines = [line for line in fixline_list if 'require' in lines[line] or 'assert' in lines[line] or 'if ' in lines[line]]
                random_index = random.randint(0, len(tglines)-1)
                tmp = lines[fixline_list[random_index]]
                require_com = re.compile(r'\s*require\s*(\((.+)\))')
                _statement = re.search(require_com, tmp).group(1)
                statement = re.search(require_com, tmp).group(2)
                head, tail = tmp.split(_statement)[0], tmp.split(_statement)[1]

                if action_num == 94:
                    tmp2 = head + '( {} && {} != 0 )'.format(statement, ope1) + tail
                    lines[fixline_list[random_index]] = tmp2
                if action_num == 95:
                    tmp2 = head + '( {} || {} != 0 )'.format(statement, ope1) + tail
                    lines[fixline_list[random_index]] = tmp2

            except:
                write_newlines(repair_path, lines)
                return fail

        if 96 <= action_num <= 97:
            try:
                ope1 = matches_bugline.group(1)
                tmp = lines[bugline]
                if_com = re.compile(r'\s*if\s*(\(([^()]+)\))')
                _statement = re.search(if_com, tmp).group(1)
                statement = re.search(if_com, tmp).group(2)
                head, tail = tmp.split(_statement)[0], tmp.split(_statement)[1]

                if action_num == 96:
                    tmp2 = head + '( {} && {} != 0 )'.format(statement, ope1) + tail
                    lines[bugline] = tmp2
                if action_num == 97:
                    tmp2 = head + '( {} || {} != 0 )'.format(statement, ope1) + tail
                    lines[bugline] = tmp2

            except:
                write_newlines(repair_path, lines)
                return fail

        if 98 <= action_num <= 99:
            try:
                ope1 = matches_bugline.group(1)
                tmp = lines[bugline]
                assert_com = re.compile(r'\s*assert\s*(\(([^()]+)\))')
                _statement = re.search(assert_com, tmp).group(1)
                statement = re.search(assert_com, tmp).group(2)
                head, tail = tmp.split(_statement)[0], tmp.split(_statement)[1]

                if action_num == 98:
                    tmp2 = head + '( {} && {} != 0 )'.format(statement, ope1) + tail
                    lines[bugline] = tmp2
                if action_num == 99:
                    tmp2 = head + '( {} || {} != 0 )'.format(statement, ope1) + tail
                    lines[bugline] = tmp2

            except:
                write_newlines(repair_path, lines)
                return fail


        if action_num == 100:
            if 'call' in lines[bugline] and 'value' in lines[bugline] and '=' in lines[bugline+1]:
                tmpline = lines[bugline]
                lines[bugline] = lines[bugline+1]
                lines[bugline+1] = tmpline
            else:
                write_newlines(repair_path, lines)
                return fail

        if number == len(action_nums)-1 or action_num == 101:
            write_newlines(repair_path, lines)
            if gitdif == True:
                return -99999
            else:
                rew = fitness_function2(contract_path, repair_path, action_nums)
                return rew

def detect(original_contract, repair_contract, limited) -> tuple:
    error = smart(original_contract, limited)
    now_error = smart(repair_contract, limited)
    return error, now_error


if __name__ == '__main__':
    pass
