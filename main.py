import os

from utils2 import *

class Config_RL_multistep:
    def __init__(self):
        self.EMB_SIZE = 512
        self.ENC_SIZE = 512
        self.DEC_SIZE = 512
        self.MODEL_SIZE = 512
        self.ATTN_SIZE = 512
        self.NUM_LAYER = 2
        self.SEED = 1234
        self.DICT_SIZE_1 = 860
        self.ACTION_NUM = len(get_action())
        self.DICT_SIZE_2 = 350
        self.PRE_BATCH_SIZE = 16
        self.BATCH_SIZE = 64
        self.MAX_INPUT_SIZE = 400
        self.MAX_OUTPUT_SIZE = 5
        self.PRE_EPOCH = 20
        self.EPOCH = 100
        self.DROPOUT = 0.25
        self.LR = 1e-5
        self.START_TOKEN = 0
        self.END_TOKEN = 101


if __name__ == "__main__":
    model_name = 'multistep_RLRep'
    path = sys.argv[1]  # dataset_vul/newALLBUGS
    os.makedirs('{}/log'.format(path), exist_ok=True)
    logger = get_logger('{}/log/{}_logging.txt'.format(path, model_name))
    with open('{}/code_w2i.pkl'.format(path), 'rb') as f, open('{}/ast_w2i.pkl'.format(path), 'rb') as f2:
        code_w2i = pickle.load(f)
        ast_w2i = pickle.load(f2)
    with open('{}/code_i2w.pkl'.format(path), 'rb') as f3, open('{}/ast_i2w.pkl'.format(path), 'rb') as f4:
        code_i2w = pickle.load(f3)
        ast_i2w = pickle.load(f4)
    action = []
    best_positive_repair = 10
    config = Config_RL_multistep()

    start = -1
    if model_name == 'multistep_RLRep':
        from multistep_RLRep import *
        in_w2i = (code_w2i, ast_w2i)
        model = Model(config)
        beam_search_use = 5
        if not start == -1:
            model_path = ''
            model.load(model_path)
            model.set_trainer()

        os.makedirs('{}/model'.format(path), exist_ok=True)
        for epoch in range(start+1, config.PRE_EPOCH):
            loss = 0
            code_dir = '{}/pretrain/threelines-tokenseq'.format(path)
            ast_dir = '{}/pretrain/ast'.format(path)
            for step, batch in enumerate(get_batch(code_dir, ast_dir, config, in_w2i, pretrain=True)):
                batch_in1, batch_in2, batch_in3, batch_out = batch
                loss += model.pretrain(batch[:-2], batch[-1], 'actor')
                logger.info('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, step, loss / (step + 1)))
            model.save('{}/model/{}_{}'.format(path, model_name, epoch))

        for epoch in range(config.PRE_EPOCH, config.EPOCH):
            loss_actor = 0
            loss_critic = 0.
            code_dir = '{}/threelines-tokenseq'.format(path)
            ast_dir = '{}/ast'.format(path)
            for step, batch in enumerate(get_batch(code_dir, ast_dir, config, in_w2i, pretrain=False)):
                batch_in1, batch_in2, batch_in3 = batch
                loss = model(batch[:-1], True, batch[-1])
                loss_actor += loss
                logger.info('Epoch: {}, Batch: {}, Loss: actor:{}'.format(epoch, step, loss_actor / (step + 1),))

            preds, ats, names, rewards = [], [], [], []
            valid_code_dir = "{}/validation/threelines-tokenseq".format(path)
            valid_ast_dir = "{}/validation/ast".format(path)
            for step, batch in enumerate(get_batch(valid_code_dir, valid_ast_dir, config, in_w2i, pretrain=False)):
                batch_in1, batch_in2, batch_in3 = batch
                if beam_search_use > 0:
                    pred = model(batch[:-1], False, size=beam_search_use)[0]
                else:
                    pred = model(batch[:-1], False, size=beam_search_use)
                for at in pred:
                    ats.append(at)
                for name in batch_in3:
                    names.append(name)
            for tup in zip(ats, names):
                reward = choose_action(tup[1], tup[0], False)
                rewards.append(reward)
                os.makedirs('dataset_vul/newALLBUGS/validation/result/', exist_ok=True)
                os.makedirs("dataset_vul/newALLBUGS/validation/result/{}/original_contract".format(epoch), exist_ok=True)
                os.makedirs("dataset_vul/newALLBUGS/validation/result/{}/repair_contract".format(epoch), exist_ok=True)
                src_path = "dataset_vul/newALLBUGS/validation/contract/{}.sol".format(name)
                copy_paths = "dataset_vul/newALLBUGS/validation/result/{0}/original_contract/{1}.sol".format(epoch, name)
                repair_paths = "dataset_vul/newALLBUGS/repair_contract/{}.sol".format(name)
                copy_paths2 = "dataset_vul/newALLBUGS/validation/result/{0}/repair_contract/{1}.sol".format(epoch, name)
                shutil.copy(src_path, copy_paths)
                shutil.copy(repair_paths, copy_paths2)

            positive_repair = 0
            for x in rewards:
                if x >= 0:
                    positive_repair += 1
            if positive_repair > best_positive_repair:
                print('model update. the {0}-th epoch. positive reward / total : {1} / {2}'.format(epoch, positive_repair, len(names)))
                best_positive_repair = positive_repair
                model.save('{}/model/{}_{}'.format(path, model_name, epoch))
            else:
                print('model NOT update. the {0}-th epoch. positive reward / total : {1} / {2}'.format(epoch, positive_repair, len(names)))



