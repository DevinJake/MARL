#!/usr/bin/env python3
import os
import random
import argparse
import logging
import numpy as np
import sys
from tensorboardX import SummaryWriter

from libbots import data, model, utils

import torch
import torch.optim as optim
import torch.nn.functional as F

SAVES_DIR = "../data/saves"

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHES = 70
MAX_TOKENS = 40

log = logging.getLogger("train")

TEACHER_PROB = 1.0

TRAIN_QUESTION_PATH = '../data/auto_QA_data/mask_even/PT_train.question'
TRAIN_ACTION_PATH = '../data/auto_QA_data/mask_even/PT_train.action'
DIC_PATH = '../data/auto_QA_data/share.question'

def run_test(test_data, net, end_token, device="cpu"):
    bleu_sum = 0.0
    bleu_count = 0
    for p1, p2 in test_data:
        input_seq = model.pack_input(p1, net.emb, device)
        enc = net.encode(input_seq)
        # Return logits (N*outputvocab), res_tokens (1*N)
        _, tokens = net.decode_chain_argmax(enc, input_seq.data[0:1],
                                            seq_len=data.MAX_TOKENS,
                                            stop_at_token=end_token)
        bleu_sum += utils.calc_bleu(tokens, p2[1:])
        bleu_count += 1
    return bleu_sum / bleu_count


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

    # command line parameters
    sys.argv = ['train_crossent.py', '--cuda', '--n=crossent_even']

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", required=True, help="Category to use for training. "
    #                                                   "Empty string to train on full processDataset")
    parser.add_argument("--cuda", action='store_true', default=False,
                        help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    log.info("Device info: %s", str(device))

    saves_path = os.path.join(SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    # 得到配对的input-output pair和对应的词汇表（词汇表放在一起），这里可以换成自己的pair和词典！
    # phrase_pairs, emb_dict = data.load_data(genre_filter=args.data)
    phrase_pairs, emb_dict = data.load_data_from_existing_data(TRAIN_QUESTION_PATH, TRAIN_ACTION_PATH, DIC_PATH, MAX_TOKENS)
    # Index -> word.
    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}
    log.info("Obtained %d phrase pairs with %d uniq words",
             len(phrase_pairs), len(emb_dict))
    data.save_emb_dict(saves_path, emb_dict)
    end_token = emb_dict[data.END_TOKEN]
    # 将tokens转换为emb_dict中的indices;
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    rand = np.random.RandomState(data.SHUFFLE_SEED)
    rand.shuffle(train_data)
    log.info("Training data converted, got %d samples", len(train_data))
    train_data, test_data = data.split_train_test(train_data)
    log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))

    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)
    # 转到cuda
    net.cuda()
    log.info("Model: %s", net)

    writer = SummaryWriter(comment="-" + args.name)

    optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    best_bleu = None
    for epoch in range(MAX_EPOCHES):
        losses = []
        bleu_sum = 0.0
        bleu_count = 0
        dial_shown = False
        random.shuffle(train_data)
        for batch in data.iterate_batches(train_data, BATCH_SIZE):
            optimiser.zero_grad()
            # input_idx：一个batch的输入句子的tokens对应的ID矩阵；
            # output_idx：一个batch的输出句子的tokens对应的ID矩阵；
            ''' 猜测input_seq：一个batch输入的所有tokens的embedding，大小为358*50；
            tensor([[-1.0363, -1.6041,  0.1451,  ..., -1.0645,  0.2387,  1.2934],
        [-1.0363, -1.6041,  0.1451,  ..., -1.0645,  0.2387,  1.2934],
        [-1.0363, -1.6041,  0.1451,  ..., -1.0645,  0.2387,  1.2934],
        ...,
        [ 0.5198, -0.3963,  1.4022,  ...,  1.0182,  0.2710, -1.5520],
        [ 2.1937, -0.5535, -0.9000,  ..., -0.1032,  0.3514, -1.2759],
        [-0.8078,  0.1575,  1.1064,  ...,  0.1365,  0.4121, -0.4211]],
       device='cuda:0')'''
            input_seq, out_seq_list, _, out_idx = model.pack_batch(batch, net.emb, device)
            enc = net.encode(input_seq)

            net_results = []
            net_targets = []
            for idx, out_seq in enumerate(out_seq_list):
                ref_indices = out_idx[idx][1:]
                enc_item = net.get_encoded_item(enc, idx)
                # teacher forcing做训练；
                if random.random() < TEACHER_PROB:
                    r = net.decode_teacher(enc_item, out_seq)
                    blue_temp = model.seq_bleu(r, ref_indices)
                    bleu_sum += blue_temp
                    # Get predicted tokens.
                    seq = torch.max(r.data, dim=1)[1]
                    seq = seq.cpu().numpy()
                # argmax做训练；
                else:
                    r, seq = net.decode_chain_argmax(enc_item, out_seq.data[0:1],
                                                     len(ref_indices))
                    blue_temp = utils.calc_bleu(seq, ref_indices)
                    bleu_sum += blue_temp
                net_results.append(r)
                net_targets.extend(ref_indices)
                bleu_count += 1

                if not dial_shown:
                    # data.decode_words transform IDs to tokens.
                    ref_words = [utils.untokenize(data.decode_words(ref_indices, rev_emb_dict))]
                    log.info("Reference: %s", " ~~|~~ ".join(ref_words))
                    log.info("Predicted: %s, bleu=%.4f", utils.untokenize(data.decode_words(seq, rev_emb_dict)), blue_temp)
                    dial_shown = True
            results_v = torch.cat(net_results)
            results_v = results_v.cuda()
            targets_v = torch.LongTensor(net_targets).to(device)
            targets_v = targets_v.cuda()
            loss_v = F.cross_entropy(results_v, targets_v)
            loss_v = loss_v.cuda()
            loss_v.backward()
            optimiser.step()

            losses.append(loss_v.item())
        bleu = bleu_sum / bleu_count
        bleu_test = run_test(test_data, net, end_token, device)
        log.info("Epoch %d: mean loss %.3f, mean BLEU %.3f, test BLEU %.3f",
                 epoch, np.mean(losses), bleu, bleu_test)
        writer.add_scalar("loss", np.mean(losses), epoch)
        writer.add_scalar("bleu", bleu, epoch)
        writer.add_scalar("bleu_test", bleu_test, epoch)
        if best_bleu is None or best_bleu < bleu_test:
            if best_bleu is not None:
                out_name = os.path.join(saves_path, "pre_bleu_%.3f_%02d.dat" %
                                        (bleu_test, epoch))
                torch.save(net.state_dict(), out_name)
                log.info("Best BLEU updated %.3f", bleu_test)
            best_bleu = bleu_test

        if epoch % 10 == 0:
            out_name = os.path.join(saves_path, "epoch_%03d_%.3f_%.3f.dat" %
                                    (epoch, bleu, bleu_test))
            torch.save(net.state_dict(), out_name)
        print ("------------------Epoch " + str(epoch) + ": training is over.------------------")
    writer.close()
