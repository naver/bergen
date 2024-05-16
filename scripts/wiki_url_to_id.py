import datasets
import requests
import concurrent
from tqdm import tqdm
import pickle 

def fetch_wiki_id(inp):
    wiki_link, wiki_title = inp
    response = requests.get(wiki_api.format(wiki_title))
    if response.status_code == 200:
        data = response.json()
        # hack because of weird dict structure returned json of the wiki_api
        pages = data.get("query", {}).get("pages", {})
        if pages:
            wiki_id = next(iter(pages.keys()))
            return (wiki_link, wiki_id)
        else:
            wiki_object_has_no_pages += 1
            return None
    else:
        count_not_found += 1
        print(f"wiki page {wiki_link} could not be fetched from wiki!")
        return None


datasets_ = datasets.load_dataset('din0s/asqa')
wiki_api = "https://en.wikipedia.org/w/api.php?action=query&format=json&titles={}"
count_not_found = 0
no_url = 0
wiki_object_has_no_pages = 0
wiki_titles = list()
split = 'dev'

# iterate over dataset splits
dataset = datasets_[split]
# for each example
for example in dataset:
    # get wiki obj
    wiki_objects = example['wikipages']
    # we can have multiple wiki objects
    for wiki_object in wiki_objects:
        if wiki_object['url'] != None:
            # get wiki url
            wiki_link = wiki_object['url']
            # get title
            wiki_title = wiki_link.split("/")[-1]
            # fetch id by title
            wiki_titles.append((wiki_link, wiki_title))
        else:
            no_url += 1
# remove duplivates            
wiki_titles = list(set(wiki_titles))

# multithread
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    # Map the function to each element in the list using threads
    wiki_tuples = list(tqdm(executor.map(fetch_wiki_id, wiki_titles), total=len(wiki_titles)))
print('Wiki pages that could not be machted:', count_not_found)
print('Wiki json has no pages object:', wiki_object_has_no_pages)
print('Examples in dataset without url field:', no_url)

url2iw = dict(wiki_tuples)
print(len(url2iw))
pickle.dump(url2iw, open('wiki2id_asqa.p', 'wb'))   
