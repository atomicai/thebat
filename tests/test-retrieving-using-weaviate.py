import unittest
from thebat.storing.weaviate import IWeaviateDocStore
from transformers import AutoModel, AutoTokenizer
from more_itertools import chunked
from torch import Tensor
import torch


def tokenize(input_texts, tokenizer):
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    return batch_dict


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def process_fn_passage(x):
    return "passage:" + x


def process_fn_query(x):
    return "query:" + x


def fire_with_embeddings(docs, model, tokenizer):
    wrapped_docs = []
    for chunk_docs in chunked(docs, n=2):
        _docs = [process_fn_passage(x) for x in chunk_docs]
        batch_dict = tokenize(_docs, tokenizer=tokenizer)  # text to features
        with torch.no_grad():
            outputs = model(**batch_dict)
            _embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().numpy()
        chunk = [{"content": doc, "embedding": emb} for doc, emb in zip(_docs, _embeddings)]
        wrapped_docs.extend(chunk)
    return wrapped_docs


class IWeaviateRetrieverTest(unittest.TestCase):
    def setUp(self):
        self.store = IWeaviateDocStore()
        self.docs = [
            "I realize I detest Haymitch. No wonder the District 12 tributes never stand a chance. It isn’t just that we’ve been underfed and lack training. Some of our tributes have still been strong enough to make a go of it. But we rarely get sponsors and he’s (Haymitch) a big part of the reason why. The rich people who back tributes — either because they’re betting on them or simply for the bragging rights of picking a winner — expect someone classier than Haymitch to deal with.",
            "In late summer, I was washing up in a pond when I noticed the plants growing around me. Tall with leaves like arrowheads. Blossoms with three white petals. I knelt down in the water, my fingers digging into the soft mud, and I pulled up handfuls of the roots. Small, bluish tubers that don’t look like much but boiled or baked are as good as any potato. \"Katniss,\" I said aloud. It’s the plant I was named for. And I heard my father’s voice joking, \"As long as you can find yourself, you’ll never starve.\"\n I spent hours stirring up the pond bed with my toes and a stick, gathering the tubers that floated to the top. That night, we feasted on fish and katniss roots until we were all, for the first time in months, full.",
            "They’re funny birds and something of a slap in the face to the Capitol. During the rebellion, the Capitol bred a series of genetically altered animals as weapons. The common term for them was muttations, or sometimes mutts for short. One was a special bird called a jabberjay that had the ability to memorize and repeat whole human conversations. They were homing birds, exclusively male, that were released into regions where the Capitol’s enemies were known to be hiding. After the birds gathered words, they’d fly back to centers to be recorded. It took people awhile to realize what was going on in the districts, how private conversations were being transmitted. Then, of course, the rebels fed the Capitol endless lies, and the joke was on it. So the centers were shut down and the birds were abandoned to die off in the wild.",
            "Peeta Mellark, on the other hand, has obviously been crying and interestingly enough does not seem to be trying to cover it up. I immediately wonder if this will be his strategy in the Games. To appear weak and frightened, to reassure the other tributes that he is no competition at all, and then come out fighting. This worked very well for a girl, Johanna Mason, from District 7 a few years back. She seemed like such a sniveling, cowardly fool that no one bothered about her until there were only a handful of contestants left. It turned out she could kill viciously. Pretty clever, the way she played it. But this seems an odd strategy for Peeta Mellark because he’s a baker’s son. All those years of having enough to eat and hauling bread trays around have made him broad-shouldered and strong. It will take an awful lot of weeping to convince anyone to overlook him.",
            "Finally, Gale is here and maybe there is nothing romantic between us, but when he opens his arms I don’t hesitate to go into them. His body is familiar to me — the way it moves, the smell of wood smoke, even the sound of his heart beating I know from quiet moments on a hunt — but this is the first time I really feel it, lean and hard-muscled against my own. \"Listen,\" he says. \"Getting a knife should be pretty easy, but you’ve got to get your hands on a bow. That’s your best chance.\"\n \"They don’t always have bows,\" I say, thinking of the year there were only horrible spiked maces that the tributes had to bludgeon one another to death with.\n \"Then make one,\" says Gale. \"Even a weak bow is better than no bow at all.\" I have tried copying my father’s bows with poor results. It’s not that easy. Even he had to scrap his own work sometimes.",
        ]
        model_name_or_path = "intfloat/multilingual-e5-base"
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.store.get_document_count() <= 0:
            self.store.delete_all_documents()
        wrapped_docs = fire_with_embeddings(
            docs=[process_fn_passage(x) for x in self.docs], model=self.model, tokenizer=self.tokenizer
        )
        self.store.write_documents(documents=wrapped_docs)

    def test_searching(self):
        queries = ["Hunger games"]
        for q in queries:
            response = self.store.query(q)
            print(response)


if __name__ == "__main__":
    unittest.main()
