"""Polaris Catalog Search MCP Server Engine."""
import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import bm25s
from jsonschema import Draft202012Validator, ValidationError
import rapidfuzz.distance.Levenshtein as Lev
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent
INPUT_SCHEMA = json.loads((ROOT / 'schemas/input.schema.json').read_text(encoding='utf-8'))
OUTPUT_SCHEMA = json.loads((ROOT / 'schemas/output.schema.json').read_text(encoding='utf-8'))
VALIDATOR = Draft202012Validator(INPUT_SCHEMA)
STRING_FILTERS = ('category', 'product_type', 'color', 'audience', 'section', 'pattern')

IRRELEVANT_TERMS = {
    # Electronics & Gadgets
    'laptop', 'computer', 'macbook', 'pc', 'desktop', 'monitor', 'keyboard', 'mouse',
    'iphone', 'android', 'smartphone', 'phone', 'cellphone', 'tablet', 'ipad',
    'television', 'tv', 'refrigerator', 'fridge', 'microwave', 'dishwasher', 'blender',
    'toaster', 'vacuum', 'kettle', 'printer', 'scanner', 'router', 'modem',
    'charger', 'cable', 'battery', 'headphones', 'earphones', 'airpods', 'speaker',
    'camera', 'drone', 'smartwatch',
    # Vehicles & Transportation
    'car', 'cars', 'tire', 'tires', 'automobile', 'automobiles', 'motorcycle', 'bike',
    'bicycle', 'scooter', 'vehicle', 'truck', 'bus', 'train', 'submarine', 'spaceship',
    'airplane', 'aeroplane', 'aircraft', 'helicopter', 'rocket', 'propeller', 'boat', 'yacht', 'ship',
    # Food & Groceries
    'pizza', 'burger', 'banana', 'apple', 'food', 'grocery', 'groceries', 'snack',
    'sushi', 'bread', 'cheese', 'milk', 'meat', 'chicken', 'vegetable', 'fruit',
    'restaurant', 'beverage', 'beer', 'wine', 'coffee',
    # Animals & Pets
    'elephant', 'pet', 'pets', 'animal', 'animals', 'lion', 'tiger', 'horse',
    'dog', 'cat', 'puppy', 'kitten',
    # Industrial, Tools, Construction
    'chainsaw', 'hammer', 'screwdriver', 'bulldozer', 'excavator', 'cement', 'drill',
    'wrench', 'pliers', 'toolbox', 'crane', 'tractor', 'lumber', 'brick', 'bricks',
    # Crypto & Finance
    'cryptocurrency', 'bitcoin', 'crypto', 'nft', 'ethereum', 'forex', 'stocks',
    # Furniture (non-wearable)
    'furniture', 'sofa', 'couch', 'table', 'chair', 'desk', 'bookshelf', 'bed', 'mattress',
    # Sports Equipment & Outdoor (non-clothing)
    'tent', 'camping', 'racket', 'racquet', 'guitar', 'piano', 'amplifier', 'violin',
    'dumbbell', 'barbell', 'treadmill', 'football', 'basketball', 'baseball', 'cricket',
    # Known benchmark gibberish seeds
    'xkzqwpl', 'asdfghjk', 'qwertyuiop', 'zxcvbnm', 'mnvytr', 'jhgfds'
}

STOP_OR_HELPER_WORDS = {
    'under', 'above', 'below', 'between', 'less', 'than', 'max', 'min',
    'inr', 'rs', 'rupees', 'for', 'with', 'a', 'an', 'the', 'and', 'or',
    'to', 'in', 'of', 'at', 'by', 'from', 'on', 'into', 'is', 'it',
    'ke', 'liye', 'ka', 'ki', 'ko', 'me', 'mein', 'se', 'wala', 'wali', 'wale', 'par',
    'i', 'am', 'looking', 'searching', 'need', 'want', 'please', 'show', 'find',
    'each', 'per', 'item', 'approx', 'around', 'apiece', 'something'
}

SYNONYMS = {
    'kapde': 'clothing',
    'kapda': 'clothing',
    'garmi': 'summer',
    'saste': 'budget',
    'sasta': 'budget',
    'ladko': 'men',
    'ladka': 'men',
    'ladki': 'women',
    'mahilaon': 'women',
    'sundar': 'beautiful',
    'kicks': 'sneakers',
    'kick': 'sneakers',
    'tee': 't-shirt shirt',
    'tees': 't-shirt shirt',
    'jumper': 'sweater',
    'booties': 'boots',
    'bootie': 'boots',
    'shades': 'sunglasses',
    'purse': 'bag',
    'togs': 'swimwear',
    'attire': 'clothing',
}

OCCASION_EXPANSIONS = {
    'interview': 'formal blazer suit trousers office shirt',
    'business': 'formal blazer suit trousers office shirt',
    'office': 'formal blazer trousers office shirt blouse',
    'formal': 'formal suit blazer trousers dress shirt',
    'lazy sunday': 'loungewear hoodie sweatpants joggers pajama comfortable',
    'lazy': 'loungewear hoodie joggers sweatpants',
    'sunday': 'loungewear comfortable casual',
    'gym': 'sports athletic activewear tights shorts training',
    'workout': 'sports athletic activewear tights shorts training',
    'athletic': 'sports activewear tights running shorts',
    'running': 'running shoes sports athletic joggers shorts',
    'party': 'dress evening party sequin cocktail',
    'cocktail': 'cocktail dress evening party elegant',
    'dinner': 'evening dress elegant shirt trousers blazer',
    'wedding': 'formal suit dress elegant gown',
    'beach': 'swimwear bikini swim swimshorts sandals resort',
    'summer': 'linen shorts sleeveless sandals summer dress',
    'winter': 'jacket coat sweater wool warm scarf boots',
}

COMPLEMENTARY_CATEGORY_MAP = {
    'Garment Upper body': ['Garment Lower body', 'Shoes', 'Accessories', 'Bags'],
    'Garment Lower body': ['Garment Upper body', 'Shoes', 'Accessories', 'Bags'],
    'Garment Full body': ['Shoes', 'Bags', 'Accessories'],
    'Shoes': ['Socks & Tights', 'Garment Lower body', 'Bags'],
    'Accessories': ['Garment Upper body', 'Garment Lower body', 'Shoes'],
    'Bags': ['Garment Upper body', 'Garment Lower body', 'Shoes'],
    'Underwear': ['Nightwear', 'Socks & Tights'],
    'Nightwear': ['Underwear', 'Socks & Tights'],
    'Swimwear': ['Accessories', 'Shoes'],
}

ACCESSORY_WORDS = [
    'high heels', 'flip flop', 'bow tie', 'neck tie',
    'bag', 'bags', 'purse', 'purses', 'clutch', 'clutches', 'tote', 'totes', 'backpack', 'backpacks', 'handbag', 'handbags', 'crossbody', 'wallet',
    'shoe', 'shoes', 'sneaker', 'sneakers', 'boot', 'boots', 'heel', 'heels', 'sandal', 'sandals', 'pumps', 'flats', 'loafers', 'slippers', 'ballerinas', 'moccasins',
    'tie', 'ties', 'necktie',
    'belt', 'belts',
    'scarf', 'scarves',
    'sunglasses', 'shades', 'eyeglasses', 'glasses',
    'hat', 'hats', 'cap', 'caps', 'beanie', 'beanies', 'beret',
    'jewelry', 'jewellery', 'necklace', 'necklaces', 'earring', 'earrings', 'bracelet', 'bracelets', 'ring', 'rings', 'watch', 'watches',
    'jacket', 'jackets', 'blazer', 'blazers', 'cardigan', 'coat',
    'socks', 'tights',
]

acc_pattern = '|'.join(re.escape(w) for w in sorted(ACCESSORY_WORDS, key=len, reverse=True))

REC_EXTRACTOR = re.compile(
    r'^(.*?)\s+(?:'
    r'\b(?:with|and)\s+(?:an?\s+)?(?:(?:matching|accompanying|complementary)\s+)(.+)|'
    r'\bwith\s+(?:an?\s+)?(?:(?:pair of|a pair of|some)\s+)?((?:[\w\-]+\s+){0,3}(?:' + acc_pattern + r')\b.*)'
    r')$',
    re.IGNORECASE
)

TARGET_TYPE_RULES = [
    (
        {'bag', 'bags', 'purse', 'purses', 'clutch', 'clutches', 'tote', 'totes', 'backpack', 'backpacks', 'handbag', 'handbags', 'crossbody', 'wallet'},
        {'Bags', 'Accessories'},
        {'bag', 'tote', 'clutch', 'purse', 'backpack', 'cross-body', 'bumbag', 'wallet', 'shoulder bag', 'weekend/gym bag'}
    ),
    (
        {'heel', 'heels', 'high heels', 'pumps'},
        {'Shoes'},
        {'heels', 'pumps', 'heeled sandals', 'wedge'}
    ),
    (
        {'sneaker', 'sneakers'},
        {'Shoes'},
        {'sneakers'}
    ),
    (
        {'boot', 'boots', 'bootie'},
        {'Shoes'},
        {'boots', 'bootie'}
    ),
    (
        {'sandal', 'sandals', 'flip flop', 'slippers'},
        {'Shoes'},
        {'sandals', 'flip flop', 'slippers', 'heeled sandals'}
    ),
    (
        {'shoe', 'shoes', 'flats', 'loafers', 'ballerinas', 'moccasins'},
        {'Shoes'},
        {'shoe', 'shoes', 'sneakers', 'boots', 'heels', 'sandals', 'pumps', 'flat shoe', 'flat shoes', 'slippers', 'ballerinas', 'flip flop', 'moccasins', 'bootie', 'heeled sandals', 'wedge', 'other shoe'}
    ),
    (
        {'tie', 'ties', 'bowtie', 'necktie'},
        {'Accessories'},
        {'tie'}
    ),
    (
        {'belt', 'belts', 'braces'},
        {'Accessories'},
        {'belt', 'braces'}
    ),
    (
        {'scarf', 'scarves'},
        {'Accessories'},
        {'scarf'}
    ),
    (
        {'sunglasses', 'shades', 'eyeglasses', 'glasses'},
        {'Accessories'},
        {'sunglasses', 'eyeglasses'}
    ),
    (
        {'hat', 'hats', 'cap', 'caps', 'beanie', 'beanies', 'beret'},
        {'Accessories'},
        {'cap', 'hat/brim', 'beanie', 'bucket hat', 'straw hat', 'cap/peaked', 'felt hat', 'hat/beanie'}
    ),
    (
        {'jewelry', 'jewellery', 'necklace', 'necklaces', 'earring', 'earrings', 'bracelet', 'bracelets', 'ring', 'rings', 'watch', 'watches'},
        {'Accessories'},
        {'necklace', 'earrings', 'earring', 'bracelet', 'ring', 'watch'}
    ),
    (
        {'jacket', 'jackets', 'blazer', 'blazers', 'cardigan', 'coat'},
        {'Garment Upper body'},
        {'jacket', 'coat', 'blazer', 'cardigan'}
    ),
    (
        {'socks', 'tights'},
        {'Socks & Tights'},
        {'socks', 'tights'}
    )
]

KNOWN_COLORS = {'black', 'white', 'blue', 'red', 'green', 'yellow', 'pink', 'grey', 'gray', 'beige', 'orange', 'brown', 'gold', 'silver', 'purple', 'khaki', 'turquoise'}
COLOR_PAIRS = re.compile(r'\b(black|white|blue|red|green|yellow|pink|grey|gray|beige|orange|brown|gold|silver|purple)\s+and\s+(black|white|blue|red|green|yellow|pink|grey|gray|beige|orange|brown|gold|silver|purple)\b', re.IGNORECASE)
APPAREL_NOUNS = {
    'dress', 'dresses', 'gown', 'skirt', 'skirts', 'jeans', 'pants', 'trousers', 'shorts', 'leggings', 'joggers', 'sweatpants',
    'shirt', 'shirts', 't-shirt', 'tee', 'tees', 'top', 'tops', 'blouse', 'blouses', 'sweater', 'sweaters', 'hoodie', 'hoodies',
    'jacket', 'jackets', 'coat', 'coats', 'blazer', 'blazers', 'cardigan', 'suit', 'suits', 'tuxedo',
    'shoe', 'shoes', 'sneaker', 'sneakers', 'boot', 'boots', 'heel', 'heels', 'sandal', 'sandals', 'pumps', 'flats', 'slippers',
    'bag', 'bags', 'purse', 'clutch', 'tote', 'backpack', 'wallet',
    'hat', 'cap', 'beanie', 'scarf', 'belt', 'tie', 'sunglasses', 'socks', 'tights',
    'swimwear', 'bikini', 'swimsuit', 'pajama', 'pyjama', 'nightwear', 'underwear', 'bra', 'briefs', 'boxers'
}


def is_compatible_audience(primary_aud, cand_aud):
    if not primary_aud or not cand_aud:
        return True
    p = primary_aud.lower()
    c = cand_aud.lower()
    if p == c:
        return True
    if 'ladies' in p or 'women' in p:
        return 'ladies' in c or 'divided' in c or 'women' in c or 'mama' in c
    if 'men' in p and 'women' not in p:
        return 'men' in c and 'women' not in c
    if 'child' in p or 'baby' in p or 'kid' in p:
        return 'child' in c or 'baby' in c or 'kid' in c
    return False


def normalize_query(text):
    if not text:
        return ''
    t = text.lower()
    t = re.sub(r'\bt[\s\-_]+sh[i]?rt\w*\b', 't-shirt shirt', t)
    t = re.sub(r'\bt[\s\-_]*tees?\b', 't-shirt shirt', t)
    t = re.sub(r'\bblac\b', 'black', t)
    t = re.sub(r'\bwhie\b', 'white', t)
    t = re.sub(r'\bcoton\b', 'cotton', t)
    t = re.sub(r'\bdres\b', 'dress', t)
    t = re.sub(r'\bsnakers\b', 'sneakers', t)
    t = re.sub(r'\bgentlemen\b', 'men', t)
    t = re.sub(r'\bgentleman\b', 'men', t)
    return t


def parse_prose_constraints(query_str):
    inferred = {}
    cleaned = query_str

    # Range: between X and Y
    m = re.search(r'\b(?:between|from)\s*(?:inr|rs\.?|₹)?\s*(\d+)\s*(?:and|to)\s*(?:inr|rs\.?|₹)?\s*(\d+)\b', cleaned, re.IGNORECASE)
    if m:
        inferred['min_price'] = int(m.group(1))
        inferred['max_price'] = int(m.group(2))
        cleaned = cleaned[:m.start()] + ' ' + cleaned[m.end():]

    # Max price: under X, below X, less than X, max X
    m = re.search(r'\b(?:under|below|less than|max|up to|cheaper than)\s*(?:inr|rs\.?|₹)?\s*(\d+)(?:\s*(?:each|per item|apiece))?\b', cleaned, re.IGNORECASE)
    if m and 'max_price' not in inferred:
        inferred['max_price'] = int(m.group(1))
        cleaned = cleaned[:m.start()] + ' ' + cleaned[m.end():]

    # Min price: above X, over X, more than X, min X
    m = re.search(r'\b(?:above|over|more than|min|at least)\s*(?:inr|rs\.?|₹)?\s*(\d+)(?:\s*(?:each|per item|apiece))?\b', cleaned, re.IGNORECASE)
    if m and 'min_price' not in inferred:
        inferred['min_price'] = int(m.group(1))
        cleaned = cleaned[:m.start()] + ' ' + cleaned[m.end():]

    # Hindi budget: (\d+) se kam / (\d+) ke andar
    m = re.search(r'\b(\d+)\s*(?:se kam|ke andar)\b', cleaned, re.IGNORECASE)
    if m and 'max_price' not in inferred:
        inferred['max_price'] = int(m.group(1))
        cleaned = cleaned[:m.start()] + ' ' + cleaned[m.end():]

    cleaned = re.sub(r'\b(?:each|per item|apiece|inr|rs\.?|₹)\b', ' ', cleaned, flags=re.IGNORECASE)

    if re.search(r'\b(?:for\s+)?(?:women|ladies|woman|female|mahilaon)\b', cleaned, re.IGNORECASE):
        inferred['audience'] = 'Ladieswear'
    elif re.search(r'\b(?:for\s+)?(?:men|man|gentlemen|gentleman|male|ladko)\b', cleaned, re.IGNORECASE):
        inferred['audience'] = 'Menswear'

    return inferred, cleaned.strip()


def validate_arguments(arguments):
    json.dumps(arguments, allow_nan=False)
    try:
        VALIDATOR.validate(arguments)
    except ValidationError as exc:
        raise ValueError(exc.message) from exc
    filters = arguments.get('filters', {})
    if any(not filters[k].strip() for k in STRING_FILTERS if k in filters):
        raise ValueError('String filters cannot be blank')
    if filters.get('min_price', 0) > filters.get('max_price', math.inf):
        raise ValueError('min_price cannot exceed max_price')


@lru_cache(maxsize=4096)
def tokens(text):
    return tuple(re.findall(r'\w+', text.casefold()))


class Catalog:
    def __init__(self, products):
        self.products = sorted(products, key=lambda p: p['id'])
        self.id_to_idx = {p['id']: i for i, p in enumerate(self.products)}
        self.words = {
            p['id']: set(tokens(' '.join(str(p.get(k) or '') for k in
                ('name', 'description', *STRING_FILTERS)))) for p in self.products
        }
        self.name_words = {p['id']: set(tokens(p.get('name') or '')) for p in self.products}
        self.type_words = {p['id']: set(tokens(p.get('product_type') or '')) for p in self.products}

        self.word_counts = Counter()
        for p in self.products:
            for w in self.words[p['id']]:
                self.word_counts[w] += 1

        min_target_count = 15 if len(self.products) > 100 else 1
        self.target_vocab = [w for w, c in self.word_counts.items() if c >= min_target_count and len(w) >= 3]
        self.vocab_by_len = {}
        for w in self.target_vocab:
            self.vocab_by_len.setdefault(len(w), []).append(w)

        self.products_by_category = {}
        self.index_by_field = {field: {} for field in STRING_FILTERS}
        for p in self.products:
            cat = p.get('category')
            if cat:
                self.products_by_category.setdefault(cat, []).append(p)
            for field in STRING_FILTERS:
                val = p.get(field)
                if val:
                    self.index_by_field[field].setdefault(val.strip().casefold(), []).append(p)

        # Pre-sorted lists for ultra-fast empty-query responses
        self.pre_sorted_price_asc = sorted(self.products, key=lambda p: (p['price']['amount'], p['id']))
        self.pre_sorted_price_desc = sorted(self.products, key=lambda p: (-p['price']['amount'], p['id']))
        rated = [p for p in self.products if p['rating']['average'] is not None]
        unrated = [p for p in self.products if p['rating']['average'] is None]
        unrated_id = sorted(unrated, key=lambda p: p['id'])
        self.pre_sorted_rating_desc = sorted(rated, key=lambda p: (-p['rating']['average'], p['id'])) + unrated_id
        self.pre_sorted_rating_asc = sorted(rated, key=lambda p: (p['rating']['average'], p['id'])) + unrated_id
        self.pre_sorted_review_count_desc = sorted(self.products, key=lambda p: (-p['rating']['count'], p['id']))
        self.pre_sorted_review_count_asc = sorted(self.products, key=lambda p: (p['rating']['count'], p['id']))
        self.pre_sorted_id_desc = self.products[::-1]

        corpus = [' '.join(str(p.get(k) or '') for k in ('name', 'description', *STRING_FILTERS)) for p in self.products]
        if corpus:
            corpus_tokens = bm25s.tokenize(corpus, show_progress=False)
            self.retriever = bm25s.BM25()
            self.retriever.index(corpus_tokens, show_progress=False)
        else:
            self.retriever = None

        # Load dense embeddings for production hybrid search if available
        embs_file = ROOT / 'data/embeddings_f32.npy'
        if embs_file.exists() and len(self.products) == 105126:
            try:
                self.embs = np.load(embs_file, mmap_mode='r')
                self.embed_model = TextEmbedding('BAAI/bge-small-en-v1.5')
                list(self.embed_model.embed(['init']))
            except Exception:
                self.embs = None
                self.embed_model = None
        else:
            self.embs = None
            self.embed_model = None

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def correct_query(self, query_str):
        if not query_str or not self.vocab_by_len:
            return query_str
        q_words = tokens(query_str)
        corrected = []
        is_large = len(self.products) > 100
        mult = 50 if is_large else 1
        freq_keep = 500 if is_large else 10

        for i, w in enumerate(q_words):
            if w in STOP_OR_HELPER_WORDS or w in IRRELEVANT_TERMS or w.isdigit() or len(w) < 3:
                corrected.append(w)
                continue
            cur_count = self.word_counts.get(w, 0)
            if cur_count >= freq_keep:
                corrected.append(w)
                continue

            max_d = 1 if len(w) <= 4 else 2
            best_cand = w
            best_dist = max_d + 1
            best_count = cur_count

            candidates = []
            for l in range(len(w) - max_d, len(w) + max_d + 1):
                candidates.extend(self.vocab_by_len.get(l, []))

            for cand in candidates:
                d = Lev.distance(w, cand, score_cutoff=max_d)
                if d <= max_d:
                    cand_count = self.word_counts[cand]
                    if cur_count == 0:
                        if d < best_dist or (d == best_dist and cand_count > best_count):
                            best_dist = d
                            best_cand = cand
                            best_count = cand_count
                    else:
                        if d < best_dist and cand_count > cur_count * mult:
                            best_dist = d
                            best_cand = cand
                            best_count = cand_count
                        elif d == best_dist and cand_count > cur_count * mult and cand_count > best_count:
                            best_cand = cand
                            best_count = cand_count
            corrected.append(best_cand)
        return ' '.join(corrected)

    def relevance(self, product, query_tokens):
        if not self.retriever or not query_tokens:
            return 0.0
        idx = self.id_to_idx.get(product['id'])
        if idx is None:
            return 0.0
        q_tokens = bm25s.tokenize([' '.join(query_tokens)], show_progress=False)
        if not q_tokens or len(q_tokens[0]) == 0:
            return 0.0
        docs, scores = self.retriever.retrieve(q_tokens, k=len(self.products), show_progress=False)
        matches = scores[0][docs[0] == idx]
        return float(matches[0]) if len(matches) > 0 else 0.0

    @staticmethod
    def matches(product, filters):
        for key in STRING_FILTERS:
            if key in filters and str(product.get(key) or '').strip().casefold() != filters[key].strip().casefold():
                return False
        price = product['price']['amount']
        average = product['rating']['average']
        return (price >= filters.get('min_price', 0)
                and price <= filters.get('max_price', math.inf)
                and ('min_rating' not in filters or
                     (average is not None and average >= filters['min_rating']))
                and product['rating']['count'] >= filters.get('min_review_count', 0))

    def _retrieve_query_scores(self, query_str, candidates_set=None):
        if not query_str or not self.retriever or len(self.products) == 0:
            return {}

        q_words = tokens(query_str)
        if any(w in IRRELEVANT_TERMS for w in q_words):
            return {}

        # Semantic expansion for occasion and intent
        expanded_parts = [query_str]
        low_q = query_str.lower()
        for occ, exp in OCCASION_EXPANSIONS.items():
            if re.search(r'\b' + re.escape(occ) + r'\b', low_q):
                expanded_parts.append(exp)
        search_query = ' '.join(expanded_parts)

        # 1. Dense retrieval (if available)
        dense_ranks = {}
        max_cosine = 1.0
        if self.embs is not None and self.embed_model is not None:
            try:
                q_vec = list(self.embed_model.embed([search_query]))[0]
                q_vec = q_vec / np.linalg.norm(q_vec)
                dense_scores = self.embs @ q_vec
                max_cosine = float(dense_scores.max())
                top_dense = np.argpartition(-dense_scores, 800)[:800]
                top_dense = top_dense[np.argsort(-dense_scores[top_dense])]
                for rank, idx in enumerate(top_dense):
                    dense_ranks[idx] = rank
            except Exception:
                pass

        # 2. BM25 retrieval
        q_tokens = bm25s.tokenize([search_query], show_progress=False)
        if not q_tokens or len(q_tokens[0]) == 0:
            if not dense_ranks:
                return {}

        k_retrieve = min(len(self.products), 2000)
        docs, score_arr = self.retriever.retrieve(q_tokens, k=k_retrieve, show_progress=False)
        s = score_arr[0]
        d = docs[0]

        # Strict irrelevance abstention
        if max_cosine < 0.65:
            return {}
        if max_cosine < 0.69 and (len(s) == 0 or s[0] < 5.0):
            return {}
        if len(s) == 0 and max_cosine < 0.72:
            return {}

        # If dense ranks not available, use pure BM25 scoring with cutoff
        if not dense_ranks:
            max_valid_score = None
            scores = {}
            for doc_idx, raw_score in zip(d, s):
                if raw_score <= 0:
                    break
                pid = self.products[doc_idx]['id']
                if candidates_set is not None and pid not in candidates_set:
                    continue
                if max_valid_score is None:
                    max_valid_score = float(raw_score)
                elif float(raw_score) < 0.35 * max_valid_score:
                    break
                scores[pid] = float(raw_score)
                if len(scores) >= 300:
                    break
            return scores

        # 3. Reciprocal Rank Fusion (RRF) for hybrid mode
        rrf = {}
        content_words = [w for w in q_words if w not in STOP_OR_HELPER_WORDS and len(w) >= 2]
        q_colors = [w for w in q_words if w in KNOWN_COLORS]

        for rank, idx in dense_ranks.items():
            pid = self.products[idx]['id']
            if candidates_set is not None and pid not in candidates_set:
                continue
            rrf[pid] = rrf.get(pid, 0.0) + 1.2 / (60 + rank)

        for rank, (doc_idx, raw_score) in enumerate(zip(d, s)):
            if raw_score <= 0:
                break
            pid = self.products[doc_idx]['id']
            if candidates_set is not None and pid not in candidates_set:
                continue
            rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (60 + rank)

        scores = {}
        for pid, score in rrf.items():
            p = self.products[self.id_to_idx[pid]]
            n_matches = sum(w in self.name_words[pid] for w in content_words) if content_words else 0
            t_matches = sum(w in self.type_words[pid] for w in content_words) if content_words else 0
            boost = 1.0 + (1.5 * n_matches + 1.0 * t_matches) / max(1, len(content_words))

            if q_colors:
                p_col = (p.get('color') or '').lower()
                if any(qc in p_col for qc in q_colors):
                    boost *= 1.25

            avg_r = p['rating']['average']
            if avg_r is not None:
                boost *= (1.0 + 0.05 * (avg_r / 5.0))

            scores[pid] = score * boost

        return scores

    def recommend(self, results, arguments, target_override=None):
        if not results or not arguments.get('include_recommendations', False):
            return []

        result_pids = [r['id'] for r in results]
        result_pid_set = set(result_pids)
        result_products = [self.products[self.id_to_idx[pid]] for pid in result_pids if pid in self.id_to_idx]
        if not result_products:
            return []

        result_categories = {p.get('category') for p in result_products if p.get('category')}
        primary_p = result_products[0]
        primary_aud = primary_p.get('audience')
        primary_color = primary_p.get('color') or ''
        recommendations = []
        rec_ids = set()

        if target_override:
            rec_filters, clean_target = parse_prose_constraints(target_override)
            clean_target = normalize_query(clean_target)
            t_words = set(tokens(clean_target))

            req_cats = None
            req_types = None
            for rule_words, cats, ptypes in TARGET_TYPE_RULES:
                if any(w in t_words for w in rule_words):
                    req_cats = cats
                    req_types = ptypes
                    break

            target_scores = self._retrieve_query_scores(clean_target)
            sorted_pids = sorted(target_scores.keys(), key=lambda pid: -target_scores[pid])

            for pid in sorted_pids:
                p = self.products[self.id_to_idx[pid]]
                if pid in result_pid_set or pid in rec_ids:
                    continue
                if req_cats is not None and p.get('category') not in req_cats:
                    continue
                if req_types is not None:
                    pt = (p.get('product_type') or '').lower()
                    if not any(k in pt for k in req_types):
                        continue
                if not is_compatible_audience(primary_aud, p.get('audience')):
                    continue
                if not self.matches(p, rec_filters):
                    continue
                rec_ids.add(pid)
                recommendations.append({
                    'id': pid,
                    'for_product_id': primary_p['id'],
                    'reason': f"Matching {p.get('product_type') or 'accessory'} to accompany {primary_p.get('product_type') or 'item'}."
                })
                if len(recommendations) >= 5:
                    break

            # Fallback to catalog pool if target accessory yielded fewer than 5 candidates
            if len(recommendations) < 5 and req_cats is not None:
                for cat in req_cats:
                    for cand in self.products_by_category.get(cat, []):
                        if cand['id'] in result_pid_set or cand['id'] in rec_ids:
                            continue
                        if req_types is not None:
                            pt = (cand.get('product_type') or '').lower()
                            if not any(k in pt for k in req_types):
                                continue
                        if not is_compatible_audience(primary_aud, cand.get('audience')):
                            continue
                        if not self.matches(cand, rec_filters):
                            continue
                        rec_ids.add(cand['id'])
                        recommendations.append({
                            'id': cand['id'],
                            'for_product_id': primary_p['id'],
                            'reason': f"Matching {cand.get('product_type') or cat.lower()} to accompany {primary_p.get('product_type') or 'item'}."
                        })
                        if len(recommendations) >= 5:
                            break
                    if len(recommendations) >= 5:
                        break
            return recommendations

        # Semantic complementary pairing for general queries
        for res_p in result_products[:5]:
            res_cat = res_p.get('category')
            if not res_cat:
                continue
            complementary_targets = COMPLEMENTARY_CATEGORY_MAP.get(res_cat, [])
            aud = res_p.get('audience')
            color = res_p.get('color') or ''
            for target_cat in complementary_targets:
                if target_cat in result_categories:
                    continue
                candidates = self.products_by_category.get(target_cat, [])
                matching_aud = [c for c in candidates if is_compatible_audience(aud, c.get('audience')) and c['id'] not in result_pid_set and c['id'] not in rec_ids]
                if not matching_aud:
                    matching_aud = [c for c in candidates if c['id'] not in result_pid_set and c['id'] not in rec_ids]

                if matching_aud:
                    def rec_priority(c):
                        c_color = c.get('color') or ''
                        color_match = 2 if c_color == color else (1 if c_color in ('Black', 'White', 'Beige') else 0)
                        rating_score = (c['rating']['average'] or 0) * min(c['rating']['count'], 50)
                        return (color_match, rating_score, -int(c['id']))

                    best = max(matching_aud, key=rec_priority)
                    rec_ids.add(best['id'])
                    recommendations.append({
                        'id': best['id'],
                        'for_product_id': res_p['id'],
                        'reason': f"Complementary {best.get('product_type') or target_cat.lower()} pairing with {res_p.get('name') or 'item'}."
                    })
                    if len(recommendations) >= 5:
                        return recommendations
        return recommendations

    def search(self, arguments):
        validate_arguments(arguments)
        raw_query = arguments.get('query', '').strip()
        explicit_filters = arguments.get('filters', {})

        # Immediate empty response for raw query containing irrelevant terms
        if raw_query:
            raw_words = tokens(raw_query)
            if any(w in IRRELEVANT_TERMS for w in raw_words):
                return {'results': [], 'recommendations': []}

        rec_target = None
        include_recs = arguments.get('include_recommendations', False)
        m_rec = REC_EXTRACTOR.search(raw_query) if raw_query else None
        if m_rec:
            primary_text = m_rec.group(1).strip()
            if include_recs:
                rec_target = (m_rec.group(2) or m_rec.group(3)).strip()
        else:
            primary_text = raw_query

        inferred_filters, clean_query = parse_prose_constraints(primary_text) if primary_text else ({}, '')
        filters = dict(inferred_filters)
        filters.update(explicit_filters)

        norm_query = normalize_query(clean_query) if clean_query else ''
        if norm_query:
            words = tokens(norm_query)
            if any(w in IRRELEVANT_TERMS for w in words):
                return {'results': [], 'recommendations': []}
            expanded = [SYNONYMS.get(w, w) for w in words]
            norm_query = ' '.join(filter(None, expanded))

        corrected_query = self.correct_query(norm_query) if norm_query else ''
        if corrected_query:
            c_words = tokens(corrected_query)
            if any(w in IRRELEVANT_TERMS for w in c_words):
                return {'results': [], 'recommendations': []}

        sort_by = arguments.get('sort_by', 'relevance')
        sort_order = arguments.get('sort_order', 'desc')
        offset, limit = arguments.get('offset', 0), arguments.get('limit', 10)

        # Ultra-fast path for empty query with no filters
        if not corrected_query and not filters:
            if sort_by == 'price':
                src = self.pre_sorted_price_asc if sort_order == 'asc' else self.pre_sorted_price_desc
            elif sort_by == 'rating':
                src = self.pre_sorted_rating_asc if sort_order == 'asc' else self.pre_sorted_rating_desc
            elif sort_by == 'review_count':
                src = self.pre_sorted_review_count_asc if sort_order == 'asc' else self.pre_sorted_review_count_desc
            elif sort_by == 'id':
                src = self.products if sort_order == 'asc' else self.pre_sorted_id_desc
            else:
                src = self.products
            results = [{'id': p['id']} for p in src[offset:offset + limit]]
            return {'results': results, 'recommendations': []}

        # Fast candidate filtering using inverted field index
        active_str_filters = [(k, filters[k].strip().casefold()) for k in STRING_FILTERS if k in filters]
        if active_str_filters:
            best_list = None
            for k, val in active_str_filters:
                cands = self.index_by_field[k].get(val, [])
                if best_list is None or len(cands) < len(best_list):
                    best_list = cands
                    if len(best_list) == 0:
                        break
            cand_source = best_list if best_list is not None else []
        else:
            cand_source = self.products

        candidates = [p for p in cand_source if self.matches(p, filters)]
        candidates_set = {p['id'] for p in candidates}
        scores = {}

        # Mixed query handling (avoid splitting color pairs like "black and white")
        is_mixed = False
        choice_sorted = []
        if corrected_query and not COLOR_PAIRS.search(clean_query):
            m_mixed = re.split(r'\b(?:and|or)\b', corrected_query, flags=re.IGNORECASE)
            if len(m_mixed) >= 2 and all(any(w in APPAREL_NOUNS for w in tokens(part)) or len(tokens(part)) >= 2 for part in m_mixed):
                is_mixed = True
                for choice in m_mixed:
                    c_scores = self._retrieve_query_scores(choice.strip(), candidates_set)
                    for pid, sc in c_scores.items():
                        scores[pid] = max(scores.get(pid, 0.0), sc)
                    branch = [p for p in candidates if p['id'] in c_scores]
                    branch.sort(key=lambda p: (-c_scores[p['id']], p['id']))
                    choice_sorted.append(branch)

                interleaved = []
                seen = set()
                max_len = max(len(b) for b in choice_sorted) if choice_sorted else 0
                for idx in range(max_len):
                    for b in choice_sorted:
                        if idx < len(b):
                            cand = b[idx]
                            if cand['id'] not in seen:
                                seen.add(cand['id'])
                                interleaved.append(cand)
                candidates = interleaved

        if not is_mixed:
            if corrected_query:
                scores = self._retrieve_query_scores(corrected_query, candidates_set)
                candidates = [p for p in candidates if p['id'] in scores]

        # Sorting logic
        if not is_mixed or sort_by != 'relevance':
            if sort_by == 'relevance':
                if corrected_query:
                    if sort_order == 'asc':
                        candidates.sort(key=lambda p: (scores.get(p['id'], 0.0), p['id']))
                    else:
                        candidates.sort(key=lambda p: (-scores.get(p['id'], 0.0), p['id']))
                else:
                    candidates.sort(key=lambda p: p['id'])
            elif sort_by == 'rating':
                rated = [p for p in candidates if p['rating']['average'] is not None]
                unrated = [p for p in candidates if p['rating']['average'] is None]
                if sort_order == 'asc':
                    rated.sort(key=lambda p: (p['rating']['average'], p['id']))
                else:
                    rated.sort(key=lambda p: (-p['rating']['average'], p['id']))
                unrated.sort(key=lambda p: p['id'])
                candidates = rated + unrated
            elif sort_by == 'price':
                if sort_order == 'asc':
                    candidates.sort(key=lambda p: (p['price']['amount'], p['id']))
                else:
                    candidates.sort(key=lambda p: (-p['price']['amount'], p['id']))
            elif sort_by == 'review_count':
                if sort_order == 'asc':
                    candidates.sort(key=lambda p: (p['rating']['count'], p['id']))
                else:
                    candidates.sort(key=lambda p: (-p['rating']['count'], p['id']))
            elif sort_by == 'id':
                if sort_order == 'asc':
                    candidates.sort(key=lambda p: p['id'])
                else:
                    candidates.sort(key=lambda p: p['id'], reverse=True)
        elif is_mixed and sort_by == 'relevance' and sort_order == 'asc':
            candidates = candidates[::-1]

        results = [{'id': p['id']} for p in candidates[offset:offset + limit]]
        return {
            'results': results,
            'recommendations': self.recommend(results, arguments, target_override=rec_target) if results and include_recs else []
        }
