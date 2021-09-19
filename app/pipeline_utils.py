import wikipedia
import spacy
import nltk
from nltk.corpus import stopwords
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
import torch

stop_words = set(stopwords.words('english') + \
        [x.strip() for x in 
         open("assets/google-top-10000.txt").readlines()[:5000]
        ])

nlp = spacy.load("en_core_web_sm")

def format_string_with_links(orig_text, links):
    if len(links) == 0:
        return orig_text
    txt = orig_text[:links[0][1]]
    for i,link in enumerate(links):
        txt += '<a href="' + link[0] + '">' + orig_text[link[1]:link[2]] + "</a>"
        if i != len(links) - 1:
            txt += orig_text[link[2]:links[i+1][1]]
    txt += orig_text[links[-1][2]:]

    return txt

def add_links(gpt_text):
    links = []
    doc = nlp(gpt_text)
    for chunk in doc.noun_chunks:
        if len(chunk.text) < 4:
            continue
        pos_tags = [tok.pos_ for tok in nlp(chunk.text)]
        if 'PROPN' in pos_tags or 'NOUN' in pos_tags:
            wikipedia.set_lang('simple')
            search_text = " ".join([w.text for w in nlp(chunk.text)
                            if w.text.lower() not in stop_words])
            if len(search_text) < 5:
                continue 
            search_results = wikipedia.search(search_text, 1)
            if len(search_results) > 0:
                try:
                    links.append((wikipedia.page(search_results[0]).url,
                            chunk.start_char, chunk.end_char
                        ))
                except:# wikipedia.exceptions.DisambiguationError:
                    # print("----")
                    pass
            else:
                wikipedia.set_lang('en')
                search_results = wikipedia.search(search_text, 1)
                if len(search_results) > 0:
                    try:
                        links.append((wikipedia.page(search_results[0]).url,
                                chunk.start_char, chunk.end_char
                            ))
                    except:# wikipedia.exceptions.DisambiguationError:
                        print("----")
                        pass
    return links, format_string_with_links(gpt_text, links)


def print_stars(num_stars, max_stars=5):
    return '<div class="golden-stars-please-ignore" style="color:#16ff00;display: inline-block;text-align: right">'+ \
            (u'★' * num_stars) + "</div>" + u'☆'* (max_stars-num_stars)

def rank_gpt_texts(candidates, input_question=None):
    return candidates, [x/len(candidates) for x in list(reversed(range(len(candidates))))]

class Scorer:
    def __init__(self):
        config = AutoConfig.from_pretrained('bert-base-cased')
    
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=not args.use_slow_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
                'bert-base-cased',
                config=config,
            )

    def score(self, candidatas, input_question=None):
        tokens = self.tokenizer(*candidatas, padding="max_length")
        print("Model created")
        self.model.load_state_dict(torch.load('../models/cpu_model.pt'))
        self.model.eval()
        print("Model loaded")

if __name__ == "__main__":
    example_text = "Gluten intolerance remains fairly rare, and often not particularly severe. We have higher expectations for our own health now that we ever had in the past, so historically, people with a sensitivity to gluten may have just ignored it.\n\nFurther, while many people relied on wheat-based food products, it wasn't the only diet out there, and only became as dominant as it is now in the 20th century."
    op = add_links(example_text)
    print(op[0])
    print(op[1])

