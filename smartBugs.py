import argparse
import os
import pathlib
import sys
import yaml
import time as sleeptime
from datetime import timedelta
from multiprocessing import Manager, Pool, Process
from src.docker_api.docker_api import analyse_files
from src.interface.cli import create_parser, getRemoteDataset, isRemoteDataset, DATASET_CHOICES, TOOLS_CHOICES
from src.output_parser.SarifHolder import SarifHolder
from time import time, localtime, strftime
import multiprocessing

cfg_dataset_path = os.path.abspath('config/dataset/dataset.yaml')
with open(cfg_dataset_path, 'r') as ymlfile:
    try:
        cfg_dataset = yaml.safe_load(ymlfile)
    except yaml.YAMLError as exc:
        print(exc)

output_folder = strftime("%Y%m%d_%H%M", localtime())
os.makedirs('results/logs/', exist_ok=True)
pathlib.Path('results/logs/').mkdir(parents=True, exist_ok=True)
logs = open('results/logs/SmartBugs_' + output_folder + '.log', 'w')

def analyse(args):
    global logs, output_folder
    (tool, file, sarif_outputs, import_path, output_version, nb_task, nb_task_done, total_execution, start_time, pipi0) = args
    try:
        start = time()
        nb_task_done.value += 1
        analyze_result = analyse_files(tool, file, logs, output_folder, sarif_outputs, output_version, import_path)
        total_execution.value += time() - start
        duration = str(timedelta(seconds=round(time() - start)))
        task_sec = nb_task_done.value / (time() - start_time)
        remaining_time = str(timedelta(seconds=round((nb_task - nb_task_done.value) / task_sec)))
        sys.stdout.write('\x1b[1;37m' + 'Done [%d/%d, %s]: ' % (nb_task_done.value, nb_task, remaining_time) + '\x1b[0m')
        sys.stdout.write('\x1b[1;34m' + file + '\x1b[0m')
        sys.stdout.write('\x1b[1;37m' + ' [' + tool + '] in ' + duration + ' ' + '\x1b[0m' + '\n')
    except Exception as e:
        print(e)
        raise e
    pipi0.send(analyze_result)
    return analyze_result

def exec_cmd(args: argparse.Namespace, ltime):
    global logs, output_folder
    files_to_analyze = []
    for file in args.file:
        if os.path.basename(file).endswith('.sol'):
            files_to_analyze.append(file)
        elif os.path.isdir(file):
            if args.import_path == "FILE":
                args.import_path = file
            for root, dirs, files in os.walk(file):
                for name in files:
                    if name.endswith('.sol'):
                        files_to_analyze.append(os.path.join(root, name))
        else:
            print('%s is not a directory or a solidity file' % file)

    start_time = time()
    manager = Manager()
    nb_task_done = manager.Value('i', 0)
    total_execution = manager.Value('f', 0)
    nb_task = len(files_to_analyze) * len(args.tool)
    sarif_outputs = manager.dict()
    tasks = []
    file_names = []
    pipi = multiprocessing.Pipe()
    for file in files_to_analyze:
        for tool in args.tool:
            results_folder = 'results/' + tool + '/' + output_folder
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            tasks.append((tool, file, sarif_outputs, args.import_path, args.output_version, nb_task, nb_task_done, total_execution, start_time, pipi[0]))
        file_names.append(os.path.splitext(os.path.basename(file))[0])
    for file_name in file_names:
        sarif_outputs[file_name] = SarifHolder()

    p = Process(target=analyse, args=(tasks[0],))
    p.start()
    p.join(ltime)
    if p.is_alive():
        print("pass time")
        p.terminate()
        sleeptime.sleep(0.1)
        return {}
    contract_inspection_reuslts = pipi[1].recv()
    return contract_inspection_reuslts

def smart(contract_path, ltime):
    error = 0
    tools = ['mythril', 'oyente', 'securify', 'slither', 'mythril']
    for tool in tools:
        sys.argv[1:] = ['--tool', tool, '--file', contract_path]
        args = create_parser()
        execution_result = exec_cmd(args, ltime)
        if tool == 'oyente':
            if execution_result is None or execution_result == {} or 'analysis' not in execution_result.keys() or execution_result['analysis'] == []:
                return -1
            for obj in execution_result['analysis']:
                for obj2 in obj['errors']:
                    if obj2['message'] == 'Integer Overflow.':
                        error += 1
        elif tool == 'slither':
            if execution_result is None or execution_result == {} or 'analysis' not in execution_result.keys() or execution_result['analysis'] == [] or execution_result['analysis'] is None:
                return -1
            error = len([obj for obj in execution_result['analysis'] if obj['check'] == 'tx-origin'])
        elif tool == 'securify':
            if execution_result is None or execution_result == {} or 'analysis' not in execution_result.keys() or execution_result['analysis'] is None or execution_result['analysis'] == {}:
                return -1
            for cont in execution_result['analysis']:
                for ts in execution_result['analysis'][cont]['results']:
                    if ts == 'TODReceiver' or ts == 'TODTransfer' or ts == 'TODAmount':
                        error += len(execution_result['analysis'][cont]['results'][ts]['violations']) + len(execution_result['analysis'][cont]['results'][ts]['conflicts'])
        elif tool == 'mythril':
            if execution_result is None or execution_result == {} or 'analysis' not in execution_result.keys() or execution_result['analysis'] == {}:
                return -1
            if execution_result['analysis']['success'] == False:
                return -1
            for obj in execution_result['analysis']['issues']:
                if obj['title'] == 'Message call to external contract' or obj['title'] == 'State access after external call' or obj['title'] == 'DAO':
                    error += 1
            for obj in execution_result['analysis']['issues']:
                if obj['title'] == 'Unchecked CALL return value':
                    error += 1
            else:
                return -1
    return error
