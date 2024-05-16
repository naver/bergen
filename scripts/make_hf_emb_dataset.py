
if __name__ == "__main__":
    import argparse
    import datasets
    import glob
    from tqdm import tqdm
    import torch
    import pyarrow as pa
    import pandas as pd 

    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_folder', type=str, required=True)
    parser.add_argument('--num_porc', desc='Number of processes used for building and saving dataset.', type=int, default=35)
    args = parser.parse_args()

    class StreamDatasetBuilder(datasets.GeneratorBasedBuilder):
        def _info(self):
            return datasets.DatasetInfo(
                description='dataset',
                features=datasets.Features(
                    {
                        "embedding":  datasets.Array2D(shape=(1, 4096), dtype='float16'),
                    }
                ),
                supervised_keys=None,
                homepage="",
                citation='',
            )

        def _split_generators(self, dl_manager):
            emb_files = glob.glob(f'{args.embeddings_folder}/*.pt')
            sorted_emb_files = sorted(emb_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": sorted_emb_files}),
            ]

        def _generate_examples(self, filepaths):
            id_ = 0
            for filepath in filepaths:
                embeds = torch.load(filepath)
                for emb in embeds:
                    yield id_, {'embedding': emb.unsqueeze(0)}
                    id_ += 1

    dataset_builder = StreamDatasetBuilder(name='Stream')
    dataset_builder.download_and_prepare(num_proc=args.num_proc)
    dataset = dataset_builder.as_dataset(split="train")
    dataset.save_to_disk(f'{args.embeddings_folder.rstrip("/")}.hf', num_proc=args.num_proc)

    

    

    exit()


    ## could not make arrowbased builder work which would directly would generate chunks instead of single examples

    class StreamDatasetBuilder(datasets.ArrowBasedBuilder):

        def _info(self):
            print('info')
            return datasets.DatasetInfo(
                description='dataset',
                features=datasets.Features(
                    {
                        "embedding":  datasets.Array2D(shape=(1, 4096), dtype='float16'),
                    }
                ),
                supervised_keys=None,
                homepage="",
                citation='',
            )

        def _split_generators(self, dl_manager):
            print('split')
            emb_files = glob.glob(f'{args.embeddings_folder}/*.pt')
            sorted_emb_files = sorted(emb_files, key=lambda x: int(''.join(filter(str.isdigit, x))))[:2]
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": sorted_emb_files}),
            ]

        def _generate_tables(self, filepaths):
            print(filepaths)
            def stream_embedding_chunks(filepaths):
                id_ = 0
                for filepath in filepaths:
                    embeds = torch.load(filepath)
                    for emb in embeds:
                        yield id_, {'embedding': emb.unsqueeze(0)}
                        id_ += 1

            for chunk in stream_embedding_chunks():
                yield self._dict_to_table(chunk)

        def _dict_to_table(self, data):
            return pa.Table.from_pandas(pd.DataFrame(data))

    stream_builder = StreamDatasetBuilder()
    stream_builder.download_and_prepare(force_download=True)
    dataset = stream_builder.as_dataset(split="train")  
