import json
import os
import sys
import random
import shutil
from collections import defaultdict
import argparse

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets.fingerprint import Hasher
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

from modeling_provence import DebertaV2ForCompressionAndRanking
from utils import load_trec_dict, evaluate_retrieval_simple

# setting seed for reproducibility
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# saving csv results
class Logger:
    def __init__(self, out_folder):
        os.makedirs(out_folder, exist_ok=True)
        self.out_folder = out_folder
        self.writer = SummaryWriter(os.path.join(self.out_folder, "tensorboard"))
        self.train_log = open(os.path.join(out_folder, "train_log.csv"), "w")
        self.train_log.write("step,loss,compression_loss,ranking_loss,acc\n")
        self.eval_log = open(os.path.join(out_folder, "eval_log.csv"), "w")
        self.eval_log.write("epoch,acc,pre,rec,f1,loss\n")
        self.log = open(os.path.join(out_folder, "log.txt"), "w")
        self.log.write(" ".join(sys.argv) + "\n")

    def log_train(self, *vals):
        self.train_log.write(
            ",".join(["%.3f" % v if type(v) != int else "%d" % v for v in vals]) + "\n"
        )
        self.train_log.flush()

    def log_tb(self, d, step):
        for k, v in d.items():
            self.writer.add_scalar(k, v, step)

    def log_eval(self, *vals):
        self.eval_log.write(
            ",".join(["%.3f" % v if type(v) != int else "%d" % v for v in vals]) + "\n"
        )
        self.eval_log.flush()

    def log_text(self, s):
        self.log.write(s + "\n")
        self.log.flush()

    def save_preds(self, epoch, texts_pred, texts_label):
        with open(
            os.path.join(self.out_folder, f"preds_epoch{epoch}.txt"), "w"
        ) as fout:
            for text_pred, text_label in zip(texts_pred, texts_label):
                fout.write("Pred:\n")
                fout.write(" ".join(text_pred).replace(" ##", "") + "\n")
                fout.write("Oracle:\n")
                fout.write(" ".join(text_label).replace(" ##", "") + "\n")
                fout.write("_" * 50 + "\n")


def tokenize_with_labels(item, tokenizer):
    tokenized_context = []
    labels = []
    for idx, sent in enumerate(item["context"]):
        tokenized_sent = tokenizer.tokenize(sent if idx == 0 else " " + sent)
        tokenized_context += tokenized_sent
        label = int(idx in item["selected_sents"])
        labels += [label] * len(tokenized_sent)

    tokenized_query = tokenizer.tokenize(item["query"])
    return tokenized_query, tokenized_context, labels


class DatasetComp(Dataset):
    def __init__(self, train_data, tokenizer, max_len):
        self.len = len(train_data)
        self.data = train_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        tokenized_query, tokenized_sentence, labels = tokenize_with_labels(
            self.data[index], self.tokenizer
        )
        
        tokenized_sentence = (
            ["[CLS]"] + tokenized_query + ["[SEP]"] + tokenized_sentence + ["[SEP]"]
        )  # add special tokens
        labels = [0] * (len(tokenized_query) + 2) + labels + [0]
        
        if len(tokenized_sentence) > self.max_len:
            # truncate
            tokenized_sentence = tokenized_sentence[:self.max_len]
            labels = labels[:self.max_len]
        else:
            tokenized_sentence = tokenized_sentence + [
                "[PAD]" for _ in range(self.max_len - len(tokenized_sentence))
            ]
            labels = labels + [0 for _ in range(self.max_len - len(labels))]
        
        attn_mask = [1 if tok != "[PAD]" else 0 for tok in tokenized_sentence]
        
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        return {
            "ids": torch.tensor(ids, dtype=torch.long).to(device, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long).to(
                device, dtype=torch.long
            ),
            "targets": torch.tensor(labels, dtype=torch.long).to(
                device, dtype=torch.long
            ),
            "ranking_labels": torch.tensor(self.data[index]["ranking_labels"]).to(
                device
            ),
            "q_id": self.data[index]["q_id"],
            "d_id": self.data[index]["d_id"],
        }

    def __len__(self):
        return self.len


class DatasetRank(Dataset):
    def __init__(self, reranking_file):
        self.data = []
        with open(reranking_file) as handler:
            for line in handler.readlines():
                q_id, d_id, q, d = line.split("\t")
                self.data.append((q_id, d_id, q, d))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class RankingLoader(DataLoader):
    def __init__(self, tokenizer, max_length, **kwargs):
        self.max_length = max_length
        self.tokenizer = tokenizer
        super().__init__(collate_fn=self.collate_fn, **kwargs, pin_memory=True)

    def collate_fn(self, batch):
        q_id, d_id, q, d = zip(*batch)
        example = self.tokenizer(
            list(q),
            list(d),
            add_special_tokens=True,
            padding="longest",  # pad to max sequence length in batch
            truncation="only_second",  # truncates to self.max_length, only second arg (document)
            max_length=self.max_length,
            return_attention_mask=True,
        )

        sample = {
            **{k: torch.tensor(v) for k, v in example.items()},
            **{"q_id": q_id, "d_id": d_id},
        }
        return sample


def train_epoch(
    training_type, estimates=[0, 0, 0, 0, 0], ranking_eval=None, loss_weight=None
):
    tr_loss, tr_comp_loss, tr_ranking_loss, tr_accuracy, nb_tr_steps = (
        estimates[0],
        estimates[1],
        estimates[2],
        estimates[3],
        estimates[4],
    )
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()

    for idx, batch in enumerate(training_loader):
        if idx % args.accum == 0:
            optimizer.zero_grad()

        ids = batch["ids"]
        mask = batch["mask"]
        targets = batch["targets"]
        targets_comp = None if training_type == "ranking" else batch["targets"]
        targets_ranking = (
            None if training_type == "compression" else batch["ranking_labels"]
        )

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=targets_comp,
            ranking_labels=targets_ranking,
            loss_weight=loss_weight,
        )
        loss, comp_loss, ranking_loss, tr_logits = (
            outputs.loss,
            outputs.compression_loss,
            outputs.ranking_loss,
            outputs.compression_logits,
        )

        # adding mean for dataparallel training
        loss = loss.mean()

        tr_loss += loss.item()
        tr_comp_loss += 0 if training_type == "ranking" else comp_loss.mean().item()
        tr_ranking_loss += (
            0 if training_type == "compression" else ranking_loss.mean().item()
        )

        nb_tr_steps += 1

        with torch.no_grad():
            # compute training accuracy
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(
                -1, 2
            )  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(
                active_logits, axis=1
            )  # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = (
                mask.view(-1) == 1
            )  # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_preds.extend(predictions)
            tr_labels.extend(targets)

            tmp_tr_accuracy = accuracy_score(
                targets.cpu().numpy(), predictions.cpu().numpy()
            )
            tr_accuracy += tmp_tr_accuracy

        if idx % 800 == 0:
            loss_step = tr_loss / nb_tr_steps
            loss_c_step = tr_comp_loss / nb_tr_steps
            loss_r_step = tr_ranking_loss / nb_tr_steps
            logger.log_train(
                nb_tr_steps, loss_step, loss_c_step, loss_r_step, tr_accuracy
            )
            logger.log_tb(
                {
                    "train/loss": loss_step,
                    "train/loss_compression": loss_c_step,
                    "train/loss_ranking": loss_r_step,
                    "train/batch_loss": loss.item(),
                    "train/accuracy": tr_accuracy / nb_tr_steps,
                },
                nb_tr_steps,
            )

        if idx % 3000 == 0:
            if ranking_eval is not None:
                ndcg_10, ndcg_30 = valid_ranking(
                    model=model, testing_loader=ranking_eval[0], qrel=ranking_eval[1]
                )
                logger.log_tb(
                    {"val/ndcg@10_iter/": ndcg_10, "val/ndcg@30_iter": ndcg_30},
                    nb_tr_steps,
                )
                model.train()
        if nb_tr_steps * batch["ids"].shape[0] % 600000 == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(
                f"{logger.out_folder}/model_step-{nb_tr_steps}/"
            )
            tokenizer.save_pretrained(f"{logger.out_folder}/model_step-{nb_tr_steps}/")

        # backward pass

        loss.backward()
        if idx % args.accum == args.accum - 1:
            optimizer.step()

        if args.debug:
            break

    return [tr_loss, tr_comp_loss, tr_ranking_loss, tr_accuracy, nb_tr_steps]


def valid_ranking(model, testing_loader, qrel):
    model.eval()
    run = defaultdict(dict)
    with torch.no_grad():
        for batch in testing_loader:
            for k, v in batch.items():
                if k not in ["q_id", "d_id"]:
                    batch[k] = v.to(device)

            ranking_scores = model(
                **{k: v for k, v in batch.items() if k not in {"q_id", "d_id"}}
            ).ranking_scores
            for q_id, d_id, s in zip(
                batch["q_id"], batch["d_id"], ranking_scores.detach().cpu().tolist()
            ):
                run[str(q_id)][str(d_id)] = s
    metrics = evaluate_retrieval_simple(run, qrel, {"ndcg_cut"})
    ndcg_10 = sum([d["ndcg_cut_10"] for d in metrics.values()]) / len(metrics)
    ndcg_30 = sum([d["ndcg_cut_30"] for d in metrics.values()]) / len(metrics)
    return ndcg_10, ndcg_30


def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels, eval_texts_pred, eval_texts_label = [], [], [], []

    with torch.no_grad():
        for batch in testing_loader:

            ids = batch["ids"]
            mask = batch["mask"]
            targets = batch["targets"]
            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs.loss, outputs.compression_logits

            loss = loss.mean()

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)

            # compute evaluation accuracy
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(
                -1, 2
            )  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(
                active_logits, axis=1
            )  # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = (
                mask.view(-1) == 1
            )  # active accuracy is also of shape (batch_size * seq_len,)
            targets_all = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(targets_all)
            eval_preds.extend(predictions)

            for row, labels_, predictions_, mask_ in zip(
                ids, targets, torch.argmax(eval_logits, axis=-1), mask
            ):
                l = sum(mask_)
                tokens = tokenizer.convert_ids_to_tokens(row.tolist())[:l]
                res_pred = []
                for i in range(l):
                    res_pred.append(tokens[i])
                    if i < l - 1 and predictions_[i] == 0 and predictions_[i + 1] == 1:
                        res_pred.append("[[[")
                    if i < l - 1 and predictions_[i] == 1 and predictions_[i + 1] == 0:
                        res_pred.append("]]]")
                eval_texts_pred.append(res_pred)
                res_lab = []
                for i in range(l):
                    res_lab.append(tokens[i])
                    if i < l - 1 and labels_[i] == 0 and labels_[i + 1] == 1:
                        res_lab.append("[[[")
                    if i < l - 1 and labels_[i] == 1 and labels_[i + 1] == 0:
                        res_lab.append("]]]")
                eval_texts_label.append(res_lab)

    labels = [id.item() for id in eval_labels]
    predictions = [id.item() for id in eval_preds]

    acc = accuracy_score(labels, predictions)
    pre = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps

    return (
        np.array(labels),
        np.array(predictions),
        eval_texts_pred,
        eval_texts_label,
        acc,
        pre,
        rec,
        f1,
        eval_loss,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--model", type=str, default="naver/trecdl22-crossencoder-debertav3"
    )
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=2000)
    parser.add_argument("--train_batch_size", type=int, default=48)
    parser.add_argument("--eval_batch_size", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-06)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--data_path", required=True, nargs="+")
    parser.add_argument("--run_path", required=True, nargs="+")
    parser.add_argument("--ranking_validation_data", required=True)
    parser.add_argument("--ranking_validation_qrels", required=True)
    parser.add_argument("--loss_weight", default=None, type=float)
    parser.add_argument("--nb_queries", default=None, type=int)
    parser.add_argument("--training_type", required=True)
    parser.add_argument("--exp_folder", type=str, required=True)
    args = parser.parse_args()

    assert len(args.data_path) == len(args.run_path)
    assert (args.training_type) in ("joint", "compression", "ranking")

    final_dir = os.path.join(args.exp_folder, f"{Hasher.hash(str(vars(args)))}")
    if os.path.exists(final_dir):
        raise OSError(f"Experiment {final_dir} already exists!")
    tmp_dir = os.path.join(args.exp_folder, f"tmp_{Hasher.hash(str(vars(args)))}")

    print(
        f"Storing exps in {tmp_dir}",
        "\n\n\n",
        "======================================\n",
    )
    logger = Logger(tmp_dir)
    json.dump(
        vars(args), open(os.path.join(logger.out_folder, "training_args.json"), "w")
    )
    logger.writer.add_text("hps", json.dumps(vars(args), indent=4), global_step=0)

    run = {}
    for run_path in args.run_path:
        run_temp = load_trec_dict(run_path)
        if args.nb_queries is not None:
            q_ids = random.shuffle(list(run_temp.keys()))[: args.nb_queries]
            run_temp = {k: v for k, v in run_temp.items() if k in q_ids}
        run = run | run_temp

    data_ = []
    for data_path in args.data_path:
        for f in os.listdir(data_path):
            if f.endswith(".json"):
                f_name = f.split(".")[0]
                q_id, d_id = f_name.split("_")
                if q_id in run:
                    with open(os.path.join(data_path, f)) as fin:
                        item = json.load(fin)
                        item["q_id"] = q_id
                        item["d_id"] = d_id
                        item["ranking_labels"] = run[q_id][d_id]
                        print(item)
                        print("\n\n\n")
                        data_.append(item)
                else:
                    pass

    # simplest data filtering
    data = []
    for item in data_:
        if (
            "No answer" not in item["response"]
            and len(set(item["selected_sents"]).difference({0})) == 0
        ):
            continue
        if item["selected_sents"] == [0]:
            item["selected_sents"] = []
        data.append(item)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = DebertaV2ForCompressionAndRanking.from_pretrained(args.model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    training_set = DatasetComp(data[: -args.val_size], tokenizer, args.max_len)
    testing_set = DatasetComp(data[-args.val_size :], tokenizer, args.max_len)

    training_loader = DataLoader(
        training_set, batch_size=args.train_batch_size, shuffle=True, num_workers=0
    )
    testing_loader = DataLoader(
        testing_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0
    )
    # loading ranking validation:
    testing_ranking_dataset = DatasetRank(args.ranking_validation_data)
    testing_ranking_loader = RankingLoader(
        dataset=testing_ranking_dataset,
        tokenizer=tokenizer,
        max_length=args.max_len,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    qrel = json.load(open(args.ranking_validation_qrels))

    # compute and log val metrics at init:
    labels, predictions, texts_pred, texts_label, acc, pre, rec, f1, loss = valid(
        model, testing_loader
    )
    ndcg_10, ndcg_30 = valid_ranking(
        model=model, testing_loader=testing_ranking_loader, qrel=qrel
    )
    logger.log_eval(0, acc, pre, rec, f1, loss)
    logger.log_tb(
        {
            "val/acc": acc,
            "val/prec": pre,
            "val/recall": rec,
            "val/f1": f1,
            "val/loss": loss,
            "val/ndcg@10": ndcg_10,
            "val/ndcg@30": ndcg_30,
        },
        0,
    )

    estimates = [0, 0, 0, 0, 0]
    for epoch in range(1, args.epochs + 1):
        print(f"Training epoch: {epoch}")
        estimates = train_epoch(
            args.training_type,
            estimates,
            ranking_eval=(testing_ranking_loader, qrel),
            loss_weight=args.loss_weight,
        )
        labels, predictions, texts_pred, texts_label, acc, pre, rec, f1, loss = valid(
            model, testing_loader
        )
        ndcg_10, ndcg_30 = valid_ranking(
            model=model, testing_loader=testing_ranking_loader, qrel=qrel
        )
        logger.log_eval(epoch, acc, pre, rec, f1, loss)
        logger.log_tb(
            {
                "val/acc": acc,
                "val/prec": pre,
                "val/recall": rec,
                "val/f1": f1,
                "val/loss": loss,
                "val/ndcg@10": ndcg_10,
                "val/ndcg@30": ndcg_30,
            },
            epoch,
        )
        logger.save_preds(epoch, texts_pred, texts_label)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(f"{logger.out_folder}/model_epoch-{epoch}/")
        tokenizer.save_pretrained(f"{logger.out_folder}/model_epoch-{epoch}/")

    # adding final HF saving:
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(f"{logger.out_folder}/model/")
    tokenizer.save_pretrained(f"{logger.out_folder}/model/")
    shutil.move(tmp_dir, final_dir)