#!/usr/bin/env python3
import shutil

import docker
import json
import os
import re
import sys
import tarfile
import yaml
import tempfile
from shutil import copyfile, rmtree

from solidity_parser import parser

from src.output_parser.Conkas import Conkas
from src.output_parser.HoneyBadger import HoneyBadger
from src.output_parser.Maian import Maian
from src.output_parser.Manticore import Manticore
from src.output_parser.Mythril import Mythril
from src.output_parser.Osiris import Osiris
from src.output_parser.Oyente import Oyente
from src.output_parser.Securify import Securify
from src.output_parser.Slither import Slither
from src.output_parser.Smartcheck import Smartcheck
from src.output_parser.Solhint import Solhint

from time import time

client = docker.from_env()

"""
get solidity compiler version
"""
def get_solc_version(file, logs):
    try:
        with open(file, 'r', encoding='utf-8') as fd:
            sourceUnit = parser.parse(fd.read())  # 解析合约，编译合约
            solc_version = sourceUnit['children'][0]['value']
            solc_version = solc_version.strip('^')
            solc_version = solc_version.split('.')
            return (int(solc_version[1]), int(solc_version[2]))
    except:
        print('\x1b[1;33m' + 'WARNING: could not parse solidity file to get solc version' + '\x1b[0m')
        # logs.write('WARNING: could not parse solidity file to get solc version \n')
    return (None, None)


"""
pull images
"""
def pull_image(image, logs):
    try:
        print('pulling ' + image + ' image, this may take a while...')
        logs.write('pulling ' + image + ' image, this may take a while...\n')
        image = client.images.pull(image)
        print('image pulled')
        # logs.write('image pulled\n')

    except docker.errors.APIError as err:
        print(err)
        # logs.write(err + '\n')


"""
mount volumes
"""
def mount_volumes(dir_path, logs):
    try:
        volume_bindings = {os.path.abspath(dir_path): {'bind': '/data', 'mode': 'rw'}}
        return volume_bindings
    except os.error as err:
        print(err)
        # logs.write(err + '\n')


"""
stop container
"""
def stop_container(container, logs):
    try:
        if container is not None:
            container.stop(timeout=0)
    except (docker.errors.APIError) as err:
        print(err)
        # logs.write(str(err) + '\n')


"""
remove container
"""
def remove_container(container, logs):
    try:
        if container is not None:
            container.remove()
    except (docker.errors.APIError) as err:
        print(err)
        # logs.write(err + '\n')


"""
write output
"""
def parse_results(output, tool, file_name, container, cfg, logs, results_folder, start, end, sarif_outputs, file_path_in_repo, output_version):
    output_folder = os.path.join(results_folder, file_name)  # 构建结果的输出文件夹

    results = {
        'contract': file_name,
        'tool': tool,
        'start': start,
        'end': end,
        'duration': end - start,
        'analysis': None
    }
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # with open(os.path.join(output_folder, 'result.log'), 'w', encoding='utf-8') as f:
    #     f.write(output)  # 把日志信息写入

    if 'output_in_files' in cfg:
        try:
            with open(os.path.join(output_folder, 'result.tar'), 'wb') as f:
                output_in_file = cfg['output_in_files']['folder']
                bits, stat = container.get_archive(output_in_file)
                for chunk in bits:
                    f.write(chunk)
        except Exception as e:
            # print(output)
            # print(e)
            print('\x1b[1;31m' + 'ERROR: could not get file from container. file not analysed.' + '\x1b[0m')
            # logs.write('ERROR: could not get file from container. file not analysed.\n')

    try:
        sarif_holder = sarif_outputs[file_name]
        if tool == 'oyente':
            results['analysis'] = Oyente().parse(output)  # Oyente解析出来的结果
            # Sarif Conversion
            sarif_holder.addRun(Oyente().parseSarif(results, file_path_in_repo))
        elif tool == 'osiris':
            results['analysis'] = Osiris().parse(output)
            sarif_holder.addRun(Osiris().parseSarif(results, file_path_in_repo))
        elif tool == 'honeybadger':
            results['analysis'] = HoneyBadger().parse(output)
            sarif_holder.addRun(HoneyBadger().parseSarif(results, file_path_in_repo))
        elif tool == 'smartcheck':
            results['analysis'] = Smartcheck().parse(output)
            sarif_holder.addRun(Smartcheck().parseSarif(results, file_path_in_repo))
        elif tool == 'solhint':
            results['analysis'] = Solhint().parse(output)
            sarif_holder.addRun(Solhint().parseSarif(results, file_path_in_repo))
        elif tool == 'maian':
            results['analysis'] = Maian().parse(output)
            sarif_holder.addRun(Maian().parseSarif(results, file_path_in_repo))
        elif tool == 'mythril':
            results['analysis'] = json.loads(output)
            sarif_holder.addRun(Mythril().parseSarif(results, file_path_in_repo))
        elif tool == 'securify':
            if len(output) > 0 and output[0] == '{':
                results['analysis'] = json.loads(output)
            elif os.path.exists(os.path.join(output_folder, 'result.tar')):
                tar = tarfile.open(os.path.join(output_folder, 'result.tar'))
                try:
                    output_file = tar.extractfile('results/results.json')
                    results['analysis'] = json.loads(output_file.read())
                    sarif_holder.addRun(Securify().parseSarif(results, file_path_in_repo))
                except Exception as e:
                    print('pas terrible')
                    output_file = tar.extractfile('results/live.json')
                    results['analysis'] = {
                        file_name: {
                            'results': json.loads(output_file.read())["patternResults"]
                        }
                    }
                    sarif_holder.addRun(Securify().parseSarifFromLiveJson(results, file_path_in_repo))
        elif tool == 'slither':
            if os.path.exists(os.path.join(output_folder, 'result.tar')):
                tar = tarfile.open(os.path.join(output_folder, 'result.tar'))
                output_file = tar.extractfile('output.json')
                results['analysis'] = json.loads(output_file.read())
                sarif_holder.addRun(Slither().parseSarif(results, file_path_in_repo))
        elif tool == 'manticore':
            if os.path.exists(os.path.join(output_folder, 'result.tar')):
                tar = tarfile.open(os.path.join(output_folder, 'result.tar'))
                m = re.findall('Results in /(mcore_.+)', output)
                results['analysis'] = []
                for fout in m:
                    output_file = tar.extractfile('results/' + fout + '/global.findings')
                    results['analysis'].append(Manticore().parse(output_file.read().decode('utf8')))
                sarif_holder.addRun(Manticore().parseSarif(results, file_path_in_repo))
        elif tool == 'conkas':
            results['analysis'] = Conkas().parse(output)
            sarif_holder.addRun(Conkas().parseSarif(results, file_path_in_repo))

        sarif_outputs[file_name] = sarif_holder

    except Exception as e:
        # print(output)
        print(e)
        # ignore
        pass

    if output_version == 'v1' or output_version == 'all':
        with open(os.path.join(output_folder, 'result.json'), 'w') as f:
            json.dump(results, f, indent=2)

    if output_version == 'v2' or output_version == 'all':
        with open(os.path.join(output_folder, 'result.sarif'), 'w') as sarifFile:
            json.dump(sarif_outputs[file_name].printToolRun(tool=tool), sarifFile, indent=2)

    return results  # 补充一个检测结果返回


# analyse solidity files
def analyse_files(tool, file, logs, now, sarif_outputs, output_version, import_path):
    try:
        cfg_path = os.path.abspath('config/tools/' + tool + '.yaml')
        with open(cfg_path, 'r', encoding='utf-8') as ymlfile:
            try:
                cfg = yaml.safe_load(ymlfile)
            except yaml.YAMLError as exc:
                print(exc)
                logs.write(exc)

        # create result folder with time
        results_folder = '/home/pc/disk1/guohy/SolidifI/smartbugs/results/' + tool + '/' + now
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        # os.makedirs(os.path.dirname(results_folder), exist_ok=True)

        # check if config file as all required fields
        if 'default' not in cfg['docker_image'] or cfg['docker_image'] == None:
            logs.write(tool + ': default docker image not provided. please check you config file.\n')
            sys.exit(tool + ': default docker image not provided. please check you config file.')
        elif 'cmd' not in cfg or cfg['cmd'] == None:
            logs.write(tool + ': commands not provided. please check you config file.\n')
            sys.exit(tool + ': commands not provided. please check you config file.')

        if import_path == "FILE":
            import_path = file
            file_path_in_repo = file
        else:
            file_path_in_repo = file.replace(import_path, '')  # file path relative to project's root directory

        file_name = os.path.basename(file)
        file_name = os.path.splitext(file_name)[0]  # 提取合约文件名

        working_dir = tempfile.mkdtemp()  # 创建临时文件
        copyfile(file, os.path.join(working_dir, os.path.basename(file)))
        file = os.path.join(working_dir, os.path.basename(file))

        # bind directory path instead of file path to allow imports in the same directory
        volume_bindings = mount_volumes(working_dir, logs)  # 绑定目录路径而不是文件路径以允许在同一目录中导入

        start = time()

        (solc_version, solc_version_minor) = get_solc_version(file, logs)  # 获取solc编译器版本

        if isinstance(solc_version, int) and solc_version < 5 and 'solc<5' in cfg['docker_image']:
            image = cfg['docker_image']['solc<5']
        # if there's no version or version >5, choose default
        else:
            image = cfg['docker_image']['default']

        if not client.images.list(image):
            pull_image(image, logs)

        cmd = cfg['cmd']
        if '{contract}' in cmd:
            cmd = cmd.replace('{contract}', '/data/' + os.path.basename(file))
        else:
            cmd += ' /data/' + os.path.basename(file)  # 构建指令，路径里有data？？
        container = None
        try:
            container = client.containers.run(image, cmd, detach=True,  # cpu_quota=150000,
                                              volumes=volume_bindings)
            try:
                container.wait(timeout=(30 * 60))
            except Exception as e:
                pass
            output = container.logs().decode('utf8').strip()
            if output.count('Solc experienced a fatal error') >= 1 or output.count('compilation failed') >= 1:
                # print(contract_path)
                # bad_debt_path = '/home/pc/guohy/SolidifI/smartbugs/dataset_vul/bad_debt/{}'.format(contract_path.split('/')[-1])
                # shutil.move(contract_path, bad_debt_path)
                # print('\x1b[1;31m' + 'ERROR: Solc experienced a fatal error. Check the results file for more info' + '\x1b[0m')
                # logs.write('ERROR: Solc experienced a fatal error. Check the results file for more info\n')  # 执行检测工具
                pass
            end = time()

            analyze_result = parse_results(output, tool, file_name, container, cfg, logs, results_folder, start, end, sarif_outputs,
                                           file_path_in_repo, output_version)  # 整理分析结果，写入json文件中,补充一个analyze result，这是分析后返回的结果
        finally:
            stop_container(container, logs)
            remove_container(container, logs)
            rmtree(working_dir)
        return analyze_result  # 补充一个返回的工具检查结果
    except (docker.errors.APIError, docker.errors.ContainerError, docker.errors.ImageNotFound) as err:
        print(err)
        # logs.write(err + '\n')
