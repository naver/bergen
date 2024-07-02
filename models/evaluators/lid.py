import fasttext
from huggingface_hub import hf_hub_download

class LID:
    def __init__(self, target_lang):
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.model = fasttext.load_model(model_path)
        self.target_lang = target_lang # Flores lang codes: 
        # https://github.com/facebookresearch/flores/blob/main/flores200/README.md
    
    def __call__(self, predictions, _, __):
        def correct_lang(response):
            response = response.replace("\n", " ")
            return self.model.predict(response)[0][0].removeprefix('__label__') == self.target_lang
        scores = []
        correct = 0
        denom = 0
        for response in predictions:
            if len(response) > 20:
                iscorrect = correct_lang(response)
                scores.append(iscorrect)
                correct += iscorrect
                denom += 1
            else:
                scores.append(-1)
        return correct/denom, scores
