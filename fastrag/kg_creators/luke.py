import nltk
import nltk.data
import spacy
from haystack.nodes.base import BaseComponent
from spacy.cli import download
from tqdm import tqdm
from transformers import LukeTokenizer
from transformers.models.luke.modeling_luke import LukeForEntityPairClassification

RELEVANT_ENTITY_LABELS = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "DATE"]


RELATION_ENTITY_TYPES = {
    "org:political/religious_affiliation": {
        "side_a": ["PERSON"],
        "side_b": ["NORP", "FAC", "ORG", "GPE"],
    },
    "per:title": {"side_a": ["PERSON"], "side_b": []},
    "org:top_members/employees": {"side_a": ["PERSON"], "side_b": ["NORP", "FAC", "ORG", "GPE"]},
    "org:country_of_headquarters": {
        "side_a": ["NORP", "FAC", "ORG", "GPE"],
        "side_b": ["LOC", "GPE"],
    },
    "org:founded": {
        "side_a": ["NORP", "FAC", "ORG", "GPE"],
        "side_b": ["NORP", "FAC", "ORG", "GPE", "LOC"],
    },
    "per:charges": {"side_a": ["PERSON"], "side_b": []},
    "per:spouse": {"side_a": ["PERSON"], "side_b": ["PERSON"]},
    "org:founded_by": {"side_a": ["NORP", "FAC", "ORG", "GPE"], "side_b": ["PERSON"]},
    "per:siblings": {"side_a": ["PERSON"], "side_b": ["PERSON"]},
    "per:country_of_birth": {"side_a": ["PERSON"], "side_b": ["LOC", "GPE"]},
    "per:cities_of_residence": {"side_a": ["PERSON"], "side_b": ["LOC", "GPE"]},
    "per:other_family": {"side_a": ["PERSON"], "side_b": ["PERSON"]},
    "per:children": {"side_a": ["PERSON"], "side_b": ["PERSON"]},
    "org:shareholders": {"side_a": ["PERSON"], "side_b": ["NORP", "FAC", "ORG", "GPE"]},
    "per:parents": {"side_a": ["PERSON"], "side_b": ["PERSON"]},
    "org:dissolved": {"side_a": ["NORP", "FAC", "ORG", "GPE"], "side_b": ["DATE"]},
    "per:city_of_birth": {"side_a": ["PERSON"], "side_b": ["LOC", "GPE"]},
    "per:age": {"side_a": ["PERSON"], "side_b": []},
    "org:city_of_headquarters": {"side_a": ["NORP", "FAC", "ORG", "GPE"], "side_b": ["LOC", "GPE"]},
    "org:subsidiaries": {
        "side_a": ["NORP", "FAC", "ORG", "GPE"],
        "side_b": ["NORP", "FAC", "ORG", "GPE"],
    },
    "org:members": {"side_a": ["PERSON"], "side_b": ["NORP", "FAC", "ORG", "GPE"]},
    "per:stateorprovince_of_death": {"side_a": ["PERSON"], "side_b": ["LOC", "GPE"]},
    "per:origin": {"side_a": ["PERSON"], "side_b": ["NORP", "FAC", "ORG", "GPE", "LOC"]},
    "per:alternate_names": {"side_a": ["PERSON"], "side_b": ["PERSON"]},
    "per:schools_attended": {"side_a": ["PERSON"], "side_b": ["NORP", "FAC", "ORG", "GPE"]},
    "per:religion": {"side_a": ["PERSON"], "side_b": ["NORP", "FAC", "ORG", "GPE"]},
    "per:cause_of_death": {"side_a": ["PERSON"], "side_b": []},
    "per:date_of_birth": {"side_a": ["PERSON"], "side_b": ["DATE"]},
    "per:stateorprovinces_of_residence": {"side_a": ["PERSON"], "side_b": ["LOC", "GPE"]},
    "per:date_of_death": {"side_a": ["PERSON"], "side_b": ["DATE"]},
    "per:stateorprovince_of_birth": {"side_a": ["PERSON"], "side_b": ["LOC", "GPE"]},
    "org:stateorprovince_of_headquarters": {
        "side_a": ["NORP", "FAC", "ORG", "GPE"],
        "side_b": ["LOC", "GPE"],
    },
    "per:country_of_death": {"side_a": ["PERSON"], "side_b": ["LOC", "GPE"]},
    "org:number_of_employees/members": {"side_a": [], "side_b": ["NORP", "FAC", "ORG", "GPE"]},
    "per:city_of_death": {"side_a": ["PERSON"], "side_b": ["LOC", "GPE"]},
    "org:member_of": {"side_a": ["PERSON"], "side_b": ["NORP", "FAC", "ORG", "GPE"]},
    "per:countries_of_residence": {"side_a": ["PERSON"], "side_b": ["LOC", "GPE"]},
    "per:employee_of": {"side_a": ["PERSON"], "side_b": ["NORP", "FAC", "ORG", "GPE"]},
    "org:alternate_names": {
        "side_a": ["NORP", "FAC", "ORG", "GPE", "LOC"],
        "side_b": ["NORP", "FAC", "ORG", "GPE", "LOC"],
    },
    "org:website": {"side_a": [], "side_b": ["NORP", "FAC", "ORG", "GPE", "LOC"]},
    "org:parents": {
        "side_a": ["NORP", "FAC", "ORG", "GPE"],
        "side_b": ["NORP", "FAC", "ORG", "GPE"],
    },
}


def init_filter_relations(pair):
    return pair[0]["label"] != "DATE"


def get_pairs(test_list):
    pairs = [[a, b] for idx, a in enumerate(test_list) for b in test_list[idx + 1 :]]
    pairs += [[x[1], x[0]] for x in pairs]
    return pairs


def ent_index_to_text(sent_chunk, ent_chunk, relations, all_ent_types):
    ent_texts = []
    for ent_pairs, cur_sentence, relation, type_pair in zip(
        ent_chunk, sent_chunk, relations, all_ent_types
    ):
        if relation == "no_relation":
            continue

        ent_1 = cur_sentence[ent_pairs[0][0] : ent_pairs[0][1]]
        ent_2 = cur_sentence[ent_pairs[1][0] : ent_pairs[1][1]]

        if ent_1 == ent_2:
            continue

        entity_const = RELATION_ENTITY_TYPES[relation]

        if len(entity_const["side_a"]) > 0 and type_pair[0] not in entity_const["side_a"]:
            continue

        if len(entity_const["side_b"]) > 0 and type_pair[1] not in entity_const["side_b"]:
            continue

        ent_texts.append((ent_1, ent_2, relation))
    return ent_texts


class LukeKGCreator(BaseComponent):
    outgoing_edges = 1

    def __init__(
        self,
        model_name: str,
        use_gpu: bool = False,
        batch_size: int = 10,
        max_length: int = 256,
        spacy_package: str = "en_core_web_sm",
    ):
        nltk.download("punkt")
        download(spacy_package)
        self.model = LukeForEntityPairClassification.from_pretrained(model_name)
        self.tokenizer = LukeTokenizer.from_pretrained(model_name)
        self.sentence_tokenizer = nltk.data.load("tokenizers/punkt/PY3/english.pickle")
        self.nlp = spacy.load(spacy_package)

        if use_gpu:
            self.model = self.model.cuda()

        self.batch_size = batch_size
        self.max_length = max_length

    def predict(self, query=None, answers=None, documents=None):
        query_dict = dict(query=query, answers=answers, documents=documents)

        all_sentences, all_ent_spans, all_ent_types = self.get_all_sentence_entities(documents)
        all_batches = self.get_batches(all_sentences, all_ent_spans)
        all_relations = []
        for inputs in tqdm(all_batches):
            relations = self.get_relations(inputs)
            all_relations += relations

        relation_triplets = ent_index_to_text(
            all_sentences, all_ent_spans, all_relations, all_ent_types
        )
        query_dict["relations"] = relation_triplets

        return query_dict

    def predict_batch(
        self,
        queries=None,
        answers=None,
        documents=None,
    ):
        query_dict = dict(query=queries, answers=answers, documents=documents, images={})
        return query_dict

    def run(self, query=None, answers=None, documents=None):
        return (
            self.predict(query=query, answers=answers, documents=documents),
            "output1",
        )

    def run_batch(
        self,
        queries=None,
        answers=None,
        documents=None,
    ):
        return (
            self.predict_batch(
                queries=queries,
                answers=answers,
                documents=documents,
            ),
            "output1",
        )

    def get_entities(self, text):
        doc = self.nlp(text)

        spans_data = []

        for ent in doc.ents:
            if ent.label_ in RELEVANT_ENTITY_LABELS:
                spans_data.append(
                    {"span": (ent.start_char, ent.end_char), "text": ent.text, "label": ent.label_}
                )

        return spans_data

    def get_all_sentence_entities(self, documents):
        all_sentences = []
        all_ent_spans = []
        all_ent_types = []

        for document_index, document in enumerate(tqdm(documents)):
            sentences = self.sentence_tokenizer.tokenize(document.content)

            for sentence in sentences:
                spans_data = self.get_entities(sentence)
                if len(spans_data) > 1:
                    all_pairs = get_pairs(spans_data)
                    for pair in all_pairs:
                        if not init_filter_relations(pair):
                            continue

                        all_sentences.append(sentence)
                        all_ent_spans.append([pair[0]["span"], pair[1]["span"]])
                        all_ent_types.append([pair[0]["label"], pair[1]["label"]])

        return all_sentences, all_ent_spans, all_ent_types

    def get_batches(self, all_sentences, all_ent_spans):
        chunks = (len(all_sentences) // self.batch_size) + 1

        all_batches = []
        for i in range(chunks):
            cur_slice = slice(i * self.batch_size, (i + 1) * self.batch_size)
            sent_chunk = all_sentences[cur_slice]
            ent_chunk = all_ent_spans[cur_slice]
            assert len(sent_chunk) == len(ent_chunk)

            if len(sent_chunk) == 0:
                continue

            inputs = self.tokenizer(
                text=sent_chunk,
                entity_spans=ent_chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            all_batches.append(inputs)
        return all_batches

    def get_relations(self, inputs):
        inputs = inputs.to(self.model.device)
        outputs = self.model(**inputs)
        logits = outputs.logits

        predicted_class_idx = logits.argmax(dim=1)
        relations_predicted = [
            self.model.config.id2label[ii] for ii in predicted_class_idx.tolist()
        ]
        return relations_predicted
