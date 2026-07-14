import torch
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.generators.generator import Generator


class LLMPisco(Generator):
    """
    Bergen generator for PISCO (pisco.model.PISCO) compress-then-generate models.

    PISCO differs from COCOM: it has no prepare_encoder_inputs / fixed mem-count. Instead each
    doc is tokenized, length-proportional <MEM> tokens are appended (n_mems = len//compr_rate + 1),
    the decoder prompt embeds "Document j:<MEM>*n_mems[j]" per doc, and generation runs via
    compress() -> replace_embeddings() -> decoder.generate(inputs_embeds=...). We reuse PISCO's
    own FineTuningCollator to build inputs so eval formatting is identical to finetuning (with
    label=None for the generation prompt).
    """

    def __init__(self,
                 model_name: str,                 # PISCO checkpoint path
                 batch_size: int,
                 max_new_tokens: int = 128,
                 max_length: int = None,
                 topk_docs: int = 5,
                 compressor_max_length: int = 128,
                 decoder_max_length: int = 2048,
                 query_dependent: bool = False,
                 device_map=None,
                 **kwargs):
        # Initialize CUDA BEFORE importing the model: Qwen3.5's fla (flash-linear-attention)
        # kernel detects + LOCKS its device backend at import time. Under bergen (spawn start
        # method) CUDA may be uninitialized when fla imports -> it locks to CPU -> at forward
        # "module 'torch.cpu' has no attribute 'device'". Touching CUDA first forces the cuda backend.
        if torch.cuda.is_available():
            torch.cuda.init()
            _ = torch.zeros(1, device="cuda")
        from pisco.model import PISCO
        from pisco.collator import FineTuningCollator

        Generator.__init__(self, model_name=model_name, batch_size=batch_size,
                           max_new_tokens=max_new_tokens, max_length=max_length)

        self.model = PISCO.from_pretrained(model_name)
        self.model.eval()
        # Qwen3.5's gated-delta-rule (fla) kernel only runs on GPU; force CUDA and verify the
        # move stuck (bergen uses spawn; a silent CPU fallback -> "torch.cpu has no attribute device").
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        dev_now = next(self.model.parameters()).device
        print(f"Loaded PISCO from {model_name}: compr_rate={self.model.compr_rate}, "
              f"bidirectional={getattr(self.model.config, 'bidirectional', None)}, device={dev_now}")
        assert dev_now.type == "cuda", f"PISCO not on CUDA (got {dev_now}); fla kernel needs GPU."

        self.topk_docs = topk_docs
        self.max_new_tokens = max_new_tokens
        self.decoder_tokenizer = self.model.decoder_tokenizer
        self.compressor_tokenizer = self.model.compressor_tokenizer
        # Reuse PISCO's finetuning collator purely for input formatting (label=None -> gen prompt).
        self.collator = FineTuningCollator(
            compressor_tokenizer=self.compressor_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
            compr_rate=self.model.compr_rate,
            compressor_max_length=compressor_max_length,
            decoder_max_length=decoder_max_length,
            query_dependent=query_dependent,
            topk_docs=topk_docs,
        )

    @torch.no_grad()
    def generate(self, model_input):
        device = next(self.model.parameters()).device
        cii = model_input["compressor_input_ids"].to(device)
        cam = model_input["compressor_attention_mask"].to(device)
        dii = model_input["decoder_input_ids"].to(device)
        dam = model_input["decoder_attention_mask"].to(device)
        embeddings = self.model.compress(cii, cam)
        dec_embeds = self.model.replace_embeddings(embeddings, dii)
        out = self.model.decoder.generate(
            inputs_embeds=dec_embeds, attention_mask=dam,
            do_sample=False, top_p=None, max_new_tokens=self.max_new_tokens,
        )
        # inputs_embeds -> generate returns only the new tokens
        texts = self.decoder_tokenizer.batch_decode(out, skip_special_tokens=True)
        cleaned = []
        for t in texts:
            if "</think>" in t:
                t = t.split("</think>", 1)[1]
            cleaned.append(t.strip())
        return cleaned

    def collate_fn(self, examples, eval=False):
        # bergen example: {q_id, query, doc:[...], label, ranking_label}
        q_ids = [e["q_id"] for e in examples]
        query = [e["query"] for e in examples]
        label = [[e["label"]] if isinstance(e["label"], str) else e["label"] for e in examples]
        ranking_label = [e.get("ranking_label") for e in examples]

        # Build inputs via PISCO's collator logic, but with label=None (generation prompt).
        all_comp_ids, all_dec_texts = [], []
        from pisco.collator_utils import add_memory_tokens_to_inputs
        for e in examples:
            docs = [self.collator.clean_text(d) for d in e["doc"][: self.topk_docs]]
            qy = self.collator.clean_text(e["query"])
            doc_ids = self.compressor_tokenizer(
                docs, padding="do_not_pad", return_tensors=None,
                truncation=True, max_length=self.collator.compressor_max_length,
            )["input_ids"]
            doc_ids, n_mems = add_memory_tokens_to_inputs(
                doc_ids, self.compressor_tokenizer, self.model.compr_rate
            )
            all_comp_ids.extend(doc_ids)
            doc_text = "".join(
                f"Document {j}:" + self.decoder_tokenizer.mem_token * n_mems[j]
                for j in range(len(docs))
            )
            # Build the GENERATION prompt directly (system+user, add_generation_prompt). We can't
            # reuse the collator's compute_prompt_and_prefix_length here: with label=None it derives
            # the prefix from messages[:-1], dropping the user turn -> Qwen template "No user query".
            messages = [
                {"role": "system", "content": self.collator.system_prompt},
                {"role": "user", "content": self.collator.user_prompt.replace("[documents]", doc_text).replace("[question]", qy)},
            ]
            prompt = self.decoder_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            all_dec_texts.append(prompt)

        comp = self.collator.compressor_pad(all_comp_ids)
        dec = self.decoder_tokenizer(
            all_dec_texts, return_tensors="pt", padding="longest",
            add_special_tokens=False, truncation=True,
            max_length=self.collator.decoder_max_length,
        )
        model_input = {
            "compressor_input_ids": comp["input_ids"],
            "compressor_attention_mask": comp["attention_mask"],
            "decoder_input_ids": dec["input_ids"],
            "decoder_attention_mask": dec["attention_mask"],
        }
        return {"model_input": model_input, "q_id": q_ids, "query": query,
                "instruction": all_dec_texts, "label": label, "ranking_label": ranking_label}

    def eval(self, dataset):
        assert len(dataset) > 0, "Empty dataset"
        self.model.eval()
        dl = DataLoader(dataset, batch_size=self.batch_size,
                        collate_fn=lambda l: self.collate_fn(l, eval=True), num_workers=0)
        query_ids, queries, instructions, responses, labels, ranking_labels = [], [], [], [], [], []
        for d in tqdm(dl, desc="Generating (PISCO)"):
            query_ids += d["q_id"]
            queries += d["query"]
            instructions += d["instruction"]
            labels += d["label"]
            ranking_labels += d["ranking_label"]
            responses += self.generate(d["model_input"])
        return query_ids, queries, instructions, responses, labels, ranking_labels
