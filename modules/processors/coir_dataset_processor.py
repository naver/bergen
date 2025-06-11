from ..dataset_processor import *
import datasets


class CoIRQueryProcessor(Processor):
    def __init__(self, task_name, *args, **kwargs):
        self.task_name = task_name
        super().__init__(*args, **kwargs, dataset_name=self.task_name)

    def process(self):
        dataset = datasets.load_dataset(
            f"CoIR-Retrieval/{self.task_name}-queries-corpus"
        )
        filtered_queries = dataset["queries"].filter(
            lambda example: example["partition"] == self.split
        )
        filtered_queries = filtered_queries.rename_column("_id", "id")
        filtered_queries = filtered_queries.rename_column("text", "content")
        filtered_queries = filtered_queries.remove_columns(
            [
                col
                for col in filtered_queries.column_names
                if col not in ["id", "content"]
            ]
        )
        print("number of queries:", len(filtered_queries))
        return filtered_queries


class CoIRDocumentProcessor(Processor):
    def __init__(self, task_name, *args, **kwargs):
        self.task_name = task_name
        super().__init__(*args, **kwargs, dataset_name=self.task_name)

    def process(self):
        documents = datasets.load_dataset(
            f"CoIR-Retrieval/{self.task_name}-queries-corpus"
        )["corpus"]
        documents = documents.rename_column("_id", "id")
        documents = documents.rename_column("text", "content")
        documents = documents.remove_columns(
            [col for col in documents.column_names if col not in ["id", "content"]]
        )
        print("size of collection:", len(documents))
        return documents
