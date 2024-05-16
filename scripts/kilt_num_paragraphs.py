import datasets
from tqdm import tqdm
from collections import defaultdict
import numpy as np

d = datasets.load_dataset('kilt_wikipedia')
print(d)
# just split in existing paragraphs

def predefined_paragraphs(sample):
    return { "num_paragraphs": sum(1 if "BULLET::::" not in el else 0 for el in sample['text']['paragraph'])}


def merge_predefined_paragraphs(sample, bool_print=False):
    paragraph_ids = sample['anchors']['paragraph_id']
    pa_dict = defaultdict(list)
    new_paragraph = list()

    wiki_id = sample['wikipedia_id']
    id_dict = defaultdict(list)

    for p, _id in zip(sample['text']['paragraph'], paragraph_ids):
        spl = p.split()
        if "BULLET::::" not in p and len(spl) >= 1:
            pa_dict[_id].append(p)
            id_dict[_id].append(_id)
    for k in pa_dict:
        p = ' '.join(pa_dict[k])
        new_paragraph.append(p)

    return {'merged_paragraphs': new_paragraph, "num_paragraphs": len(new_paragraph), "lengths": [len(el.split()) for el in new_paragraph]}



def to_predefined_100w(sample, num_words=100):
    # karpukhin
    #passages = [x.strip() for x in sample["text"]["paragraph"] if "BULLET::::" not in x]
    #doc = " ".join(passages)
    # new_passages = [doc[i:i + num_words] for i in range(0, len(doc), num_words)]
    #wiki_ids = [wiki_id] * len(new_passages)

    title = sample['wikipedia_title']
    wiki_id = sample['wikipedia_id']
    paragraph_ids = sample['anchors']['paragraph_id']
    paragraph_dict = defaultdict(list)

    id_dict = defaultdict(list)

    for predefined_paragraph, _id in zip(sample['text']['paragraph'], paragraph_ids):
        spl = predefined_paragraph.split()
        if "BULLET::::" not in predefined_paragraph and len(spl) >= 5:
            paragraph_dict[_id].append(predefined_paragraph)
            id_dict[_id].append(_id)

    final_paragraphs = list()
    for k in paragraph_dict:
        predefined_paragraph = ' '.join(paragraph_dict[k])
        # now split to 100w
        predefined_paragraph = predefined_paragraph.replace('Section::::', 'Section:')
        words = predefined_paragraph.split()
        for i in range(0, len(words), num_words):
            para = title + '. ' + ' '.join(words[i:i + num_words])
            final_paragraphs.append( para)

    return {'merged_paragraphs': final_paragraphs, "num_paragraphs": len(final_paragraphs), "lengths": [len(el.split()) for el in final_paragraphs]}


def to_100w(sample, num_words=100):
    wiki_id = sample['wikipedia_id']
    title = sample['wikipedia_title']
    # karpukhin
    passages = [x.strip() for x in sample["text"]["paragraph"] if "BULLET::::" not in x]
    doc = " ".join(passages)
    doc = doc.replace('Section::::', 'Section:')
    words = doc.split()
    final_paragraphs = [title + '. ' + " ".join(words[i:i + num_words]) for i in range(0, len(words), num_words)]
    wiki_ids = [wiki_id] * len(final_paragraphs)

    return {'merged_paragraphs': final_paragraphs, "num_paragraphs": len(final_paragraphs), "lengths": [len(el.split()) for el in final_paragraphs]}

# merge predefined paragraphs
# results: 21,985,886 mean 582 min 2 max 96001 median 247.0 ; min. 1 word: 20,091,230  mean 629 min 4 max 95222 median 295.0 ; min. 10 words: 15,004,527 mean 803 min 21 max 95118 median 464.0

# predefined paragraphs
#a = d['full'].map(merge_predefined_paragraphs, num_proc=40)

# split to predefined paragraphs according to paragraph_id and then split long paragraphs into 200w
a = d['full'].map(to_predefined_100w, num_proc=40)

# split paragraphs into 100w
#a = d['full'].map(to_100w, num_proc=40)
print("num_paragraphs", sum(a['num_paragraphs']))


# split in predefined paragraphs
#a = d['full'].map(predefined_paragraphs, num_proc=16)
lengths = [item for sublist in a['lengths'] for item in sublist]
print('mean', round(np.mean(lengths)), 'min', np.min(lengths), 'max', np.max(lengths), 'median', np.median(lengths))
exit()
for p in sum(a['merged_paragraphs'], []):
    print('len', len(p.split(' ')))
    print(p)
    print('-'*15)
exit()
# 100w split ~24m passages
passage_num = 0
for sample in tqdm(d['full']):
    text = [x.strip() for x in sample["text"]["paragraph"] if "BULLET::::" not in x and "Section::::" not in x]
    word_num = len(" ".join(text).split())
    passage_num += word_num//100 + int(bool(word_num%100))
print(passage_num)
exit()

