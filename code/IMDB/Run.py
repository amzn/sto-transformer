"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

from torch import optim
from datetime import datetime
from common.CumulativeTrainer import *
from IMDB.Config import *
from IMDB.Utils import *
from IMDB.IMDBDataset import *
from IMDB.Run_evaluation import *

import pdb
if args.model_type == 'tf':
    from IMDB.Model_transformers import *
elif args.model_type == 'tf-sto':
    from IMDB.Model_sto_transformers import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def train(args, run_id=args.local_rank):
    if args.train_epoch_start > 0:  # load a model and continue to train
        file = os.path.join(OUTPUT_MODEL_DIR, str(args.train_epoch_start) + '.pkl')

        if os.path.exists(file):
            model = IMDB(args.emb_dim, args.output_dim, vocab)
            model.load_state_dict(torch.load(file, map_location='cpu'))
        else:
            print('ERR: do not have %s' % args.train_epoch_start)

    else:

        model = IMDB(args.emb_dim, args.output_dim, vocab)
        #pdb.set_trace()
        init_params(model)

    train_size = len(train_dataset)
    model_bp_count = (args.n_epoch * train_size) / (args.n_gpu * args.train_batch_size * args.accumulation_steps) # global_batch_step
    model_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.warmup > 0:
        model_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(model_optimizer, round(args.warmup*model_bp_count), int(model_bp_count) + 100)
    else:
        model_scheduler = None

    model_trainer = CumulativeTrainer(model, tokenizer, None, args.local_rank, args.n_gpu, accumulation_steps=args.accumulation_steps, max_grad_norm=args.max_grad_norm)

    # patience, max_dev_acc, report_test_acc, max_epoch = args.early_stop, 0, 0, 0
    patience, max_epoch, max_dev_acc, report_test_acc = args.early_stop, -1, -1, -1

    for i in range(args.train_epoch_start, args.n_epoch):
        train_loss_list = model_trainer.train_epoch('train', train_dataset, collate_fn, args.train_batch_size, i, model_optimizer, model_scheduler)
        dev_list_output, dev_loss_list = model_trainer.predict('infer', valid_dataset, collate_fn, args.test_batch_size, i)
        test_list_output, test_loss_list = model_trainer.predict('infer', test_dataset, collate_fn, args.test_batch_size, i)
        test_ood_list_output, test_ood_loss_list = model_trainer.predict('infer', test_ood_dataset, collate_fn, args.test_batch_size, i)

        dev_acc = eval_from_list_output(dev_list_output)
        test_acc = eval_from_list_output(test_list_output)
        test_ood_acc = eval_from_list_output(test_ood_list_output)

        # Records to tensorboard - start
        writer.add_scalars(main_tag='Loss_%s' % run_id,
                           tag_scalar_dict={'Train': np.mean(train_loss_list),
                                            'Dev': np.mean(dev_loss_list),
                                            'Test': np.mean(test_loss_list),
                                            'Test_OOD': np.mean(test_ood_loss_list)
                                            },
                           global_step=i)
        writer.add_scalars(main_tag='Accuracy_%s' % run_id,
                           tag_scalar_dict={'Dev': dev_acc,
                                            'Test': test_acc,
                                            'Test_OOD': test_ood_acc
                                            },
                           global_step=i)
        # records to tensorboard - end

        # Evaluate & update best model
        if dev_acc > max_dev_acc:
            max_epoch, max_dev_acc, report_test_acc, report_test_ood_acc =i, dev_acc, test_acc, test_ood_acc
            model_trainer.serialize(max_epoch, output_path=OUTPUT_MODEL_DIR, use_multiple_gpu=True if args.n_gpu > 1 else False)
            print('Save model: rank=%s, max_epoch=%s, max_dev_acc=%s, report_test_acc=%s,\n path=%s' % (args.local_rank, max_epoch, max_dev_acc, test_acc, OUTPUT_MODEL_DIR))
        else:
            patience -= 1

        if patience < 0 or i == args.n_epoch:
            print('INFO: early stop at epoch=%s' % i)
            break

    # save & test best model
    print('RESULT(train)', '-' * 50)
    print('run_id=%s, max_epoch=%s, max_dev_acc=%s, report_test_acc=%s' % (args.local_rank, max_epoch, max_dev_acc, report_test_acc))
    torch.save(dev_list_output, os.path.join(OUTPUT_RESULT_DIR, 'valid.%s.%s' % (max_epoch, args.local_rank)))
    torch.save(test_list_output, os.path.join(OUTPUT_RESULT_DIR, 'test.%s.%s' % (max_epoch, args.local_rank)))
    torch.save(test_ood_list_output, os.path.join(OUTPUT_RESULT_DIR, 'test_ood.%s.%s' % (max_epoch, args.local_rank)))

    return max_epoch, report_test_acc, report_test_ood_acc


def test_max_epoch(args, run_id=args.local_rank, epoch=None):
    # one epoch test
    if epoch is not None:
        print('epoch=%s' % epoch)
        file = os.path.join(OUTPUT_MODEL_DIR, str(epoch) + '.pkl')
        print('model=%s' % file)
        if os.path.exists(file):
            model = IMDB(args.emb_dim, args.output_dim, vocab)
            model.load_state_dict(torch.load(file, map_location='cpu'))

            model_trainer = CumulativeTrainer(model, tokenizer, None, args.local_rank, args.n_gpu, accumulation_steps=args.accumulation_steps, max_grad_norm=args.max_grad_norm)

            dev_list_output, _ = model_trainer.predict('infer', valid_dataset, collate_fn, args.test_batch_size, epoch)
            test_list_output, _ = model_trainer.predict('infer', test_dataset, collate_fn, args.test_batch_size, epoch)
            test_ood_list_output, _ = model_trainer.predict('infer', test_ood_dataset, collate_fn, args.test_batch_size, epoch)

            dev_acc = eval_from_list_output(dev_list_output)
            test_acc = eval_from_list_output(test_list_output)
            test_ood_acc = eval_from_list_output(test_ood_list_output)

            print('RESULT(test)', '-' * 50)
            print('run_id=%s, max_epoch=%s, max_dev_acc=%s, report_test_acc=%s, report_test_ood_acc=%s' % (run_id, epoch, dev_acc, test_acc, test_ood_acc))
            return epoch, test_acc, test_ood_acc
        else:
            print('Err: do not exit...', file)
            return None


def test(args, run_id=args.local_rank):
    # multiple epochs test
    # max_dev_acc, report_test_acc, max_epoch = 0, 0, 0
    max_epoch, max_dev_acc, report_test_acc, report_test_ood_acc = -1, -1, -1, -1
    dev_list_output, test_list_output = [], []
    for i in range(args.infer_epoch_start, args.n_epoch):
        print('epoch=%s' % i)
        file = os.path.join(OUTPUT_MODEL_DIR, str(i) + '.pkl')
        print('model=%s' % file)
        if os.path.exists(file):
            model = IMDB(args.emb_dim, args.output_dim, vocab)
            model.load_state_dict(torch.load(file, map_location='cpu'))

            model_trainer = CumulativeTrainer(model, tokenizer, None, args.local_rank, args.n_gpu, accumulation_steps=args.accumulation_steps, max_grad_norm=args.max_grad_norm)

            # Dev & Test infer:
            # [golden_data, pred_output]
            dev_list_output, _ = model_trainer.predict('infer', valid_dataset, collate_fn, args.test_batch_size, i)
            test_list_output, _ = model_trainer.predict('infer', test_dataset, collate_fn, args.test_batch_size, i)
            test_ood_list_output, _ = model_trainer.predict('infer', test_ood_dataset, collate_fn, args.test_batch_size, i)

            dev_acc = eval_from_list_output(dev_list_output)
            test_acc = eval_from_list_output(test_list_output)
            test_ood_acc = eval_from_list_output(test_ood_list_output)

            writer.add_scalars(main_tag='Accuracy_%s' % run_id,
                               tag_scalar_dict={'Dev': dev_acc,
                                                'Test': test_acc,
                                                'Test_OOD': test_ood_acc
                                                },
                               global_step=i)
            print('run_id=%s, dev_acc=%s, test_acc=%s' % (run_id, dev_acc, test_acc))
            if dev_acc > max_dev_acc:
                max_epoch, max_dev_acc, report_test_acc, report_test_ood_acc  = i, dev_acc, test_acc, test_ood_acc
        else:
            print('Err: model does not exist: %s' % file)
    print('RESULT(test)', '-'*50)
    print('run_id=%s, max_epoch=%s, max_dev_acc=%s, report_test_acc=%s' % (run_id, max_epoch, max_dev_acc, report_test_acc))
    torch.save(dev_list_output, os.path.join(OUTPUT_RESULT_DIR, 'valid.%s.%s' % (max_epoch, args.local_rank)))
    torch.save(test_list_output, os.path.join(OUTPUT_RESULT_DIR, 'test.%s.%s' % (max_epoch, args.local_rank)))
    torch.save(test_ood_list_output, os.path.join(OUTPUT_RESULT_DIR, 'test_ood.%s.%s' % (max_epoch, args.local_rank)))
    return max_epoch, report_test_acc, report_test_ood_acc


if __name__ == '__main__':
    if FLAG_OUTPUT:
        print(args)
        writer = SummaryWriter(log_dir=LOG_DIR)

    # 1. Load vocab
    vocab_path = '%s/%s.%s.vocab-%s.pkl' % (DATA_DIR, args.model_name, args.model_type.split('-')[0], args.train_valid_split_ratio)
    if os.path.exists(vocab_path):
        vocab = torch.load(vocab_path)
    else:
        print('ERR: can not load vocab.')

    # 2. Load tokenizer
    if args.model_type == 'tf' or args.model_type == 'tf-sto':
        tokenizer = imdb_tokenize
    elif args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model).tokenize
    else:
        print('ERR: do not have a tokenizer.')

    # 3. Load data_sample_tensor
    if args.mode != 'pre':
        all_train_sample_tensor = wrap_dataset(None, None, 'train')
        all_valid_sample_tensor = wrap_dataset(None, None, 'valid')
        all_test_sample_tensor = wrap_dataset(None, None, 'test')
        all_test_ood_sample_tensor = wrap_dataset(create_ood_torchtext_dataset(), vocab, 'test_ood')

        if args.debug:
            train_sample_tensor = all_train_sample_tensor[:cut_data_index]
            valid_sample_tensor = all_valid_sample_tensor[:cut_data_index]
            test_sample_tensor = all_test_sample_tensor[:cut_data_index]
            test_ood_sample_tensor = all_test_ood_sample_tensor[:cut_data_index]
        else:
            train_sample_tensor = all_train_sample_tensor
            valid_sample_tensor = all_valid_sample_tensor
            test_sample_tensor = all_test_sample_tensor
            test_ood_sample_tensor = all_test_ood_sample_tensor

        train_dataset = IMDBDataset(None, None, sample_tensor=train_sample_tensor)
        valid_dataset = IMDBDataset(None, None, sample_tensor=valid_sample_tensor)
        test_dataset = IMDBDataset(None, None, sample_tensor=test_sample_tensor)
        test_ood_dataset = CRDataset(None, None, sample_tensor=test_ood_sample_tensor)

    # 4. Run
    start = datetime.now()
    if args.mode == 'pre':
        from IMDB.IMDBDataset import prepare_dataset
        prepare_dataset(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        # test(args)
        test_max_epoch(args, epoch=args.test_epoch)
    elif args.mode == 'eval':
        if FLAG_OUTPUT:
            eval_from_files(OUTPUT_RESULT_DIR)
    elif args.mode == 'uncertain-train-test':
        test_acc_list, test_ood_acc_list = [], []
        max_epoch, test_acc_0, test_ood_acc_0 = train(args)
        test_acc_list.append(test_acc_0)
        test_ood_acc_list.append(test_ood_acc_0)
        args.dropout = 0
        for run_i in range(1, args.n_run):
            print('inference_run_id=', run_i, '=' * 50)
            set_random_seed()
            _, test_acc, test_ood_acc = test_max_epoch(args, run_id=run_i, epoch=max_epoch)
            test_acc_list.append(test_acc)
            test_ood_acc_list.append(test_ood_acc)
        print('=' * 50)
        print(test_acc_list)
        print('Uncertain: max_epoch=%s, mean_test_acc=%s, std_test_acc=%s, mean_test_ood_acc=%s, std_test_ood_acc=%s' % (max_epoch, np.mean(test_acc_list), np.std(test_acc_list), np.mean(test_ood_acc_list), np.std(test_ood_acc_list)))

    elif args.mode == 'uncertain-test':
        test_acc_list, test_ood_acc_list = [], []
        max_epoch = args.test_epoch     # the best epoch
        args.dropout = 0
        for run_i in range(0, args.n_run):
            print('inference_run_id=', run_i, '=' * 50)
            set_random_seed()
            _, test_acc, test_ood_acc = test_max_epoch(args, run_id=run_i, epoch=max_epoch)
            test_acc_list.append(test_acc)
            test_ood_acc_list.append(test_ood_acc)
        print('=' * 50)
        print(test_acc_list)
        print('Uncertain: max_epoch=%s, mean_test_acc=%s, std_test_acc=%s, mean_test_ood_acc=%s, std_test_ood_acc=%s' % (max_epoch, np.mean(test_acc_list), np.std(test_acc_list), np.mean(test_ood_acc_list), np.std(test_ood_acc_list)))
    else:
        pass

    end = datetime.now()
    if FLAG_OUTPUT:
        writer.flush()
        writer.close()
    print('run time:%.2f mins'% ((end-start).seconds/60))