from ..dataset_processor import *
import datasets
from datasets import load_dataset
from collections import defaultdict
import urllib.request


class AmbigQAQueries(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = "ambigqa-queries"
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = "naver/ambigqa-queries"
        dataset = load_dataset(hf_name, num_proc=self.num_proc)[self.split]
        dataset = dataset.rename_column("question", "content")
        return dataset


class AmbigQAInstructFromDict(Processor):
    """basic class, instructions are contained in local json file
    TOREMOVE when we push a dataset on HF
    """

    def __init__(self, *args, **kwargs):
        dataset_name = "ambigqa-v3-long-cat2-25q"
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        def my_gen():
            # LOCAL FILE => to change after
            # data = json.load(
            #     open(
            #         "/beegfs/scratch/user/hdejean/bergen_branches_test/TAR/scripts/prompt_type3_v2_gpt4.json"
            #     )
            # )
            # data = json.load(
            #     open(
            #         "/beegfs/scratch/user/hdejean/bergen_branches_test/TAR/scripts/prompt_type3_500q_gpt4.json"
            #     )
            # )  # type3_500q
            # data = json.load(
            #     open(
            #         "/beegfs/scratch/user/hdejean/bergen_branches_test/TAR/scripts/prompt_Nadia_v1_categ2_500_gpt4.json"
            #     )
            # )
            data = json.load(
                open(
                    "/beegfs/scratch/user/tformal/evian/save/generated_instructions_40_categ2_with-doc_medium-size_gpt4.json"
                )
            )
            hf_name = "naver/ambigqa-queries"
            dataset = load_dataset(hf_name, num_proc=self.num_proc)[self.split]
            d = {}
            for item in data:
                key_ = list(item.keys())[0]
                d[key_] = item[key_]
            for sample in dataset:
                id_ = sample["id"]
                if id_ in d:
                    # q = sample["question"]
                    q = sample["nq_question"]
                    answer = sample["answer"]
                    try:
                        instruction = d[id_]["instruction"]
                    except KeyError:
                        print(id_)
                        print(d[id_])
                    yield {
                        "id": id_,
                        "content": q.strip(),
                        "instruction": instruction.strip(),
                        "label": [answer],
                    }

        return datasets.Dataset.from_generator(my_gen)


class AmbigQAFromDict(Processor):
    """temp, to remove"""

    def __init__(self, *args, **kwargs):
        dataset_name = "ambigqa-500q"
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        def my_gen():
            data = json.load(
                open(
                    "/beegfs/scratch/user/hdejean/bergen_branches_test/TAR/scripts/prompt_Nadia_v1_categ3_500_gpt4.json"
                )
            )  # only used to gather the same ids
            hf_name = "naver/ambigqa-queries"
            dataset = load_dataset(hf_name, num_proc=self.num_proc)[self.split]
            d = {}
            for item in data:
                key_ = list(item.keys())[0]
                d[key_] = item[key_]
            for sample in dataset:
                id_ = sample["id"]
                if id_ in d:
                    q = sample["question"]
                    answer = sample["answer"]
                    yield {
                        "id": id_,
                        "content": q.strip(),
                        "label": [answer],
                    }

        return datasets.Dataset.from_generator(my_gen)


# class AmbigQAInstructV1(Processor):
#     """ """

#     def __init__(self, *args, **kwargs):
#         dataset_name = "ambig_qa_InstructV1"
#         super().__init__(*args, **kwargs, dataset_name=dataset_name)

#     def process(self):
#         def my_gen():
#             i = 0
#             # LOCAL FILE => to change after
#             file = json.load(
#                 open(
#                     "/beegfs/scratch/team/nlp/llm_hackathon/instruction_following_retrieval/data_/ambigqa_basic_queries_prompts_gpt4.json"
#                 )
#             )
#             d = {}
#             for item in file:
#                 key_ = list(item.keys())[0]
#                 d[key_] = item[key_]
#             for split in ["dev", "test"]:
#                 ds = load_dataset(
#                     "erbacher/AmbigNQ-clarifying-question", num_proc=self.num_proc
#                 )[split]
#                 for x in ds:
#                     if x["ambig"]:
#                         # FIXME for some reason the fields are stored as string instead of list
#                         q = x[
#                             "question"
#                         ]  # ambiguous question (to which we are going to append an instruction)
#                         answer = eval(x["answer"])[
#                             0
#                         ]  # hard-coded (bc how we constructed the instructions for now ==> the first answer is the correct one)
#                         try:
#                             instruction = d[f"ambigqa_{i}"]["instruction"]
#                         except KeyError:
#                             print(d[f"ambigqa_{i}"])
#                             instruction = ""
#                         # assert d[f"ambigqa_{i}"]["original query"] == q, print(
#                         #     d[f"ambigqa_{i}"]["original query"], q
#                         # )
#                         yield {
#                             "id": f"ambigqa_{i}",
#                             "content": q.strip(),
#                             "instruction": instruction.strip(),
#                             "label": [answer],
#                         }
#                         i += 1

#         return datasets.Dataset.from_generator(my_gen)


# class AmbigQAInstructV2(Processor):
#     """ """

#     def __init__(self, *args, **kwargs):
#         dataset_name = "ambig_qa_InstructV2"
#         super().__init__(*args, **kwargs, dataset_name=dataset_name)

#     def process(self):
#         def my_gen():
#             i = 0
#             # LOCAL FILE => to change after
#             file = json.load(
#                 open(
#                     "/beegfs/scratch/team/nlp/llm_hackathon/instruction_following_retrieval/data_/ambigqa_queries_with_docs_prompts_gpt4.json"
#                 )
#             )
#             d = {}
#             for item in file:
#                 key_ = list(item.keys())[0]
#                 d[key_] = item[key_]
#             for split in ["dev", "test"]:
#                 ds = load_dataset(
#                     "erbacher/AmbigNQ-clarifying-question", num_proc=self.num_proc
#                 )[split]
#                 for x in ds:
#                     if x["ambig"]:
#                         # FIXME for some reason the fields are stored as string instead of list
#                         q = x[
#                             "question"
#                         ]  # ambiguous question (to which we are going to append an instruction)
#                         answer = eval(x["answer"])[
#                             0
#                         ]  # hard-coded (bc how we constructed the instructions for now ==> the first answer is the correct one)
#                         try:
#                             instruction = d[f"ambigqa_ub{i}"]["instruction"]
#                         except KeyError:
#                             # print(d[f"ambigqa_ub{i}"])
#                             print(f"ambigqa_ub{i}")
#                             instruction = ""
#                         yield {
#                             "id": f"ambigqa_ub{i}",
#                             "content": q.strip(),
#                             "instruction": instruction.strip(),
#                             "label": [answer],
#                         }
#                         i += 1

#         return datasets.Dataset.from_generator(my_gen)


# class AmbigQABase(Processor):
#     """This dataset is to evaluate the perf wo/ instructions
#     we have an ambiguous query (by construction), for which we consider an un-ambiguous answer (== the first answer == from the first desambiguated query)
#     """

#     def __init__(self, *args, **kwargs):
#         dataset_name = "ambig_qa_base"
#         super().__init__(*args, **kwargs, dataset_name=dataset_name)

#     def process(self):
#         def my_gen():
#             i = 0
#             d = {}
#             for split in ["dev", "test"]:
#                 ds = load_dataset(
#                     "erbacher/AmbigNQ-clarifying-question", num_proc=self.num_proc
#                 )[split]
#                 for x in ds:
#                     if x["ambig"]:
#                         # FIXME for some reason the fields are stored as string instead of list
#                         q = x[
#                             "question"
#                         ]  # ambiguous question (to which we are going to append an instruction)
#                         answer = eval(x["answer"])[
#                             0
#                         ]  # hard-coded (bc how we constructed the instructions for now ==> the first answer is the correct one)
#                         yield {
#                             "id": f"ambigqa_{i}",
#                             "content": q.strip(),
#                             "label": [answer],
#                         }
#                         i += 1

#         return datasets.Dataset.from_generator(my_gen)
