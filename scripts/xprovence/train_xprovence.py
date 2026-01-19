import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets.fingerprint import Hasher
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import json
import os
import sys
import random
import shutil
from collections import defaultdict
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from utils import load_trec_dict, evaluate_retrieval_simple
from modeling_xprovence import XLMRobertaForCompressionAndRanking

from datetime import timedelta

# setting seed for repro
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

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


def tokenize_and_preserve_labels(item, tokenizer):
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
        # step 1: tokenize (and adapt corresponding labels)
        tokenized_query, tokenized_sentence, labels = tokenize_and_preserve_labels(
            self.data[index], self.tokenizer
        )
        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = (
            ["<s>"] + tokenized_query + ["</s>"] + tokenized_sentence + ["</s>"]
        )  # add special tokens
        labels = [0] * (len(tokenized_query) + 2) + labels + [0]
        # step 3: truncating/padding
        maxlen = self.max_len
        if len(tokenized_sentence) > maxlen:
            # truncate
            tokenized_sentence = tokenized_sentence[:maxlen]
            labels = labels[:maxlen]
        else:
            tokenized_sentence = tokenized_sentence + [
                "<pad>" for _ in range(maxlen - len(tokenized_sentence))
            ]
            labels = labels + [0 for _ in range(maxlen - len(labels))]
        # step 4: obtain the attention mask
        attn_mask = [1 if tok != "<pad>" else 0 for tok in tokenized_sentence]
        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long),
            "targets": torch.tensor(labels, dtype=torch.long),
            "ranking_labels": torch.tensor(self.data[index]["ranking_labels"]),
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

    # Reset DistributedSampler for shuffling
    if args.distributed:
        training_loader.sampler.set_epoch(epoch)

    for idx, batch in enumerate(training_loader):
        if idx % args.accum == 0:
            optimizer.zero_grad()

        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

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

        # backward pass
        loss.backward()
        if idx % args.accum == args.accum - 1:
            if args.distributed:
                torch.cuda.synchronize()
            optimizer.step()

        tr_loss += loss.item()
        tr_comp_loss += 0 if training_type == "ranking" else comp_loss.item()
        tr_ranking_loss += 0 if training_type == "compression" else ranking_loss.item()

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
            # now, use mask to determine where we should compare predictions with targets (includes <s> and </s> token predictions)
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
            if args.distributed:
                torch.cuda.synchronize()
                dist.barrier()

        if (not args.distributed) or (dist.get_rank() == 0):
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

        if idx % 3000 == 0 and (not args.distributed or dist.get_rank() == 0):
            if ranking_eval is not None:
                ndcg_10, ndcg_30 = valid_ranking(
                    model=model, testing_loader=ranking_eval[0], qrel=ranking_eval[1]
                )
                logger.log_tb(
                    {"val/ndcg@10_iter/": ndcg_10, "val/ndcg@30_iter": ndcg_30},
                    nb_tr_steps,
                )
                model.train()
        if nb_tr_steps * batch["ids"].shape[0] % 600000 == 0 and (
            not args.distributed or dist.get_rank() == 0
        ):
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(
                f"{logger.out_folder}/model_step-{nb_tr_steps}/"
            )
            tokenizer.save_pretrained(f"{logger.out_folder}/model_step-{nb_tr_steps}/")

        if args.debug:
            break

    return [tr_loss, tr_comp_loss, tr_ranking_loss, tr_accuracy, nb_tr_steps]


def valid_ranking(model, testing_loader, qrel):
    model.eval()
    run = defaultdict(dict)
    with torch.no_grad():
        for batch in testing_loader:
            for k, v in batch.items():
                if k not in ["q_id", "d_id"] and isinstance(v, torch.Tensor):
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
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            ids = batch["ids"]
            mask = batch["mask"]
            targets = batch["targets"]
            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs.loss, outputs.compression_logits

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
            # now, use mask to determine where we should compare predictions with targets (includes <s> and </s> token predictions)
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


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    # Set NCCL environment variables before initialization
    os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30 minutes
    os.environ.setdefault('TORCH_NCCL_TRACE_BUFFER_SIZE', '100000')
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    # Initialize with longer timeout
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(minutes=30)  # 30 minute timeout
    )
    torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--tokenizer_name", type=str, default=None)
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
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--post_process", action="store_true")  # TODO: to remove

    # Distributed training parameters
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Use distributed training"
    )

    args = parser.parse_args()

    if args.tokenizer_name is None:
        args.tokenizer_name = args.model

    assert len(args.data_path) == len(args.run_path)
    assert (args.training_type) in ("joint", "compression", "ranking")

    # Setup distributed training if enabled
    if args.distributed:
        setup_distributed()

    experiment_name = args.exp_name if args.exp_name else Hasher.hash(str(vars(args)))
    final_dir = os.path.join(args.exp_folder, f"{experiment_name}")
    if os.path.exists(final_dir) and (not args.distributed or dist.get_rank() == 0):
        raise OSError(f"Experiment {final_dir} already exists!")
    tmp_dir = os.path.join(args.exp_folder, f"tmp_{experiment_name}")

    if not args.distributed or dist.get_rank() == 0:
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
    print("Loading data...")
    run = {}
    if not (args.training_type == "compression"):
        for run_path in os.listdir(args.run_path[0]):
            run_path = os.path.join(args.run_path[0], run_path)
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
                if (q_id in run) or (args.training_type == "compression"):
                    try:
                        with open(os.path.join(data_path, f)) as fin:
                            item = json.load(fin)
                            if args.post_process:
                                # TODO: to remove for final version
                                new_sents = []
                                new_labels = []
                                id_ = 0
                                for i, sent in enumerate(item["context"]):
                                    if sent == ".":
                                        pass
                                    else:
                                        if sent[-2:] == " .":
                                            sent = sent[:-2] + "."
                                        new_sents.append(sent)
                                        if i in item["selected_sents"]:
                                            new_labels.append(id_)
                                        id_ += 1
                                item["context"] = new_sents
                                item["selected_sents"] = new_labels
                            item["q_id"] = q_id
                            item["d_id"] = d_id
                            item["ranking_labels"] = 0 if (args.training_type == "compression") else run[q_id][d_id]
                            # print(item)
                            # print("\n\n\n")
                            data_.append(item)
                    except Exception as e:
                        print(f"Error loading {f}: {e}")
                        continue

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

    print(f"Loaded {len(data)} items from {len(args.data_path)} data files.")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = XLMRobertaForCompressionAndRanking.from_pretrained(args.model)

    # Select device and create model
    if args.distributed:
        device = torch.device(f"cuda:{args.gpu}")
        model = model.to(device)
        model = DistributedDataParallel(
            model, device_ids=[args.gpu] , find_unused_parameters=True
        )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    training_set = DatasetComp(data[: -args.val_size], tokenizer, args.max_len)
    testing_set = DatasetComp(data[-args.val_size :], tokenizer, args.max_len)

    if args.distributed:
        train_sampler = DistributedSampler(training_set)

        training_loader = DataLoader(
            training_set,
            batch_size=args.train_batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
        )
        testing_loader = DataLoader(
            testing_set,
            batch_size=args.eval_batch_size,
            # sampler=test_sampler,
            num_workers=0,
            pin_memory=True,
        )
    else:
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
    if not args.distributed or dist.get_rank() == 0:
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
        if not args.distributed or dist.get_rank() == 0:
            print(f"Training epoch: {epoch}")
        estimates = train_epoch(
            args.training_type,
            estimates,
            ranking_eval=(testing_ranking_loader, qrel),
            loss_weight=args.loss_weight,
        )

        # Add barrier before validation
        if args.distributed:
            dist.barrier()

        if not args.distributed or dist.get_rank() == 0:
            labels, predictions, texts_pred, texts_label, acc, pre, rec, f1, loss = (
                valid(model, testing_loader)
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
    if not args.distributed or dist.get_rank() == 0:
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(f"{logger.out_folder}/model/")
        tokenizer.save_pretrained(f"{logger.out_folder}/model/")
        shutil.move(tmp_dir, final_dir)
