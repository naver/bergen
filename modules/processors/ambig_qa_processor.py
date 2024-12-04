from ..dataset_processor import *
import datasets
from datasets import load_dataset
from collections import defaultdict
import urllib.request


# UB means Upper Bound for our experiments
# FIXME rename
class AmbigQA_UB(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = "ambig_qa_ub"
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        # dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        # ds = load_dataset("sewon/ambig_qa", "light",num_proc=self.num_proc)['train']
        def my_gen():
            sid = "ambigqa_ub"
            i = 0
            # for split in ['train','dev','test']:
            # we keep only dev and test for now
            for split in ["dev", "test"]:
                ds = load_dataset(
                    "erbacher/AmbigNQ-clarifying-question", num_proc=self.num_proc
                )[split]
                for x in ds:
                    # FIXME for some reason the fields are stored as string instead of list
                    qs = eval(x["intent"])
                    answers = eval(x["answer"])
                    if x["ambig"] is True:
                        for q, a in zip(qs, answers):
                            qid = sid + str(i)
                            i += 1
                            # print(q,a)
                            if q is not None and a is not None:
                                yield {"id": qid, "content": q, "label": [a]}

        return datasets.Dataset.from_generator(my_gen)


class AmbigQAInstructV1(Processor):
    """ """

    def __init__(self, *args, **kwargs):
        dataset_name = "ambig_qa_InstructV1"
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        def my_gen():
            i = 0
            # LOCAL FILE => to change after
            file = json.load(
                open(
                    "/beegfs/scratch/team/nlp/llm_hackathon/instruction_following_retrieval/data_/ambigqa_basic_queries_prompts_gpt4.json"
                )
            )
            d = {}
            for item in file:
                key_ = list(item.keys())[0]
                d[key_] = item[key_]
            for split in ["dev", "test"]:
                ds = load_dataset(
                    "erbacher/AmbigNQ-clarifying-question", num_proc=self.num_proc
                )[split]
                for x in ds:
                    if x["ambig"]:
                        # FIXME for some reason the fields are stored as string instead of list
                        q = x[
                            "question"
                        ]  # ambiguous question (to which we are going to append an instruction)
                        answer = eval(x["answer"])[
                            0
                        ]  # hard-coded (bc how we constructed the instructions for now ==> the first answer is the correct one)
                        try:
                            instruction = d[f"ambigqa_{i}"]["instruction"]
                        except KeyError:
                            print(d[f"ambigqa_{i}"])
                            instruction = ""
                        # assert d[f"ambigqa_{i}"]["original query"] == q, print(
                        #     d[f"ambigqa_{i}"]["original query"], q
                        # )
                        yield {
                            "id": f"ambigqa_{i}",
                            "content": q.strip() + " " + instruction.strip(),
                            "label": [answer],
                        }
                        i += 1

        return datasets.Dataset.from_generator(my_gen)


class AmbigQAInstructV2(Processor):
    """ """

    def __init__(self, *args, **kwargs):
        dataset_name = "ambig_qa_InstructV2"
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        def my_gen():
            i = 0
            # LOCAL FILE => to change after
            file = json.load(
                open(
                    "/beegfs/scratch/team/nlp/llm_hackathon/instruction_following_retrieval/data_/ambigqa_queries_with_docs_prompts_gpt4.json"
                )
            )
            d = {}
            for item in file:
                key_ = list(item.keys())[0]
                d[key_] = item[key_]
            for split in ["dev", "test"]:
                ds = load_dataset(
                    "erbacher/AmbigNQ-clarifying-question", num_proc=self.num_proc
                )[split]
                for x in ds:
                    if x["ambig"]:
                        # FIXME for some reason the fields are stored as string instead of list
                        q = x[
                            "question"
                        ]  # ambiguous question (to which we are going to append an instruction)
                        answer = eval(x["answer"])[
                            0
                        ]  # hard-coded (bc how we constructed the instructions for now ==> the first answer is the correct one)
                        try:
                            instruction = d[f"ambigqa_ub{i}"]["instruction"]
                        except KeyError:
                            # print(d[f"ambigqa_ub{i}"])
                            print(f"ambigqa_ub{i}")
                            instruction = ""
                        yield {
                            "id": f"ambigqa_ub{i}",
                            "content": q.strip() + " " + instruction.strip(),
                            "label": [answer],
                        }
                        i += 1

        return datasets.Dataset.from_generator(my_gen)


class AmbigQABase(Processor):
    """This dataset is to evaluate the perf wo/ instructions
    we have an ambiguous query (by construction), for which we consider an un-ambiguous answer (== the first answer == from the first desambiguated query)
    """

    def __init__(self, *args, **kwargs):
        dataset_name = "ambig_qa_base"
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        def my_gen():
            i = 0
            d = {}
            for split in ["dev", "test"]:
                ds = load_dataset(
                    "erbacher/AmbigNQ-clarifying-question", num_proc=self.num_proc
                )[split]
                for x in ds:
                    if x["ambig"]:
                        # FIXME for some reason the fields are stored as string instead of list
                        q = x[
                            "question"
                        ]  # ambiguous question (to which we are going to append an instruction)
                        answer = eval(x["answer"])[
                            0
                        ]  # hard-coded (bc how we constructed the instructions for now ==> the first answer is the correct one)
                        yield {
                            "id": f"ambigqa_{i}",
                            "content": q.strip(),
                            "label": [answer],
                        }
                        i += 1

        return datasets.Dataset.from_generator(my_gen)
