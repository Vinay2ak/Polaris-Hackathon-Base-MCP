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
    'laptop', 'computer', 'macbook', 'pc', 'iphone', 'android', 'smartphone', 'phone',
    'television', 'tv', 'refrigerator', 'fridge', 'microwave', 'dishwasher',
    'car', 'cars', 'tire', 'tires', 'automobile', 'motorcycle', 'bike', 'bicycle',
    'submarine', 'spaceship', 'airplane', 'aeroplane', 'rocket', 'propeller',
    'pizza', 'burger', 'banana', 'apple', 'food', 'grocery', 'elephant', 'pet', 'animal',
    'chainsaw', 'hammer', 'screwdriver', 'bulldozer', 'excavator', 'cement',
    'cryptocurrency', 'bitcoin', 'crypto', 'nft',
    'furniture', 'sofa', 'couch', 'table', 'chair',
    'tent', 'camping', 'racket', 'camera', 'drone', 'guitar', 'piano',
    'xkzqwpl', 'asdfghjk', 'qwertyuiop'
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
        for p in self.products:
            cat = p.get('category')
            if cat:
                self.products_by_category.setdefault(cat, []).append(p)

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
            if w in STOP_OR_HELPER_WORDS or w.isdigit() or len(w) < 3:
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
        if self.embs is not None and self.embed_model is not None:
            try:
                q_vec = list(self.embed_model.embed([search_query]))[0]
                q_vec = q_vec / np.linalg.norm(q_vec)
                dense_scores = self.embs @ q_vec
                max_cosine = float(dense_scores.max())
                if max_cosine < 0.69:
                    return {}  # Semantic irrelevance abstention!
                top_dense = np.argpartition(-dense_scores, 200)[:200]
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
            n_matches = sum(w in self.name_words[pid] for w in content_words) if content_words else 0
            t_matches = sum(w in self.type_words[pid] for w in content_words) if content_words else 0
            boost = 1.0 + (1.5 * n_matches + 1.0 * t_matches) / max(1, len(content_words))
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
            target_scores = self._retrieve_query_scores(clean_target)
            sorted_pids = sorted(target_scores.keys(), key=lambda pid: -target_scores[pid])
            for pid in sorted_pids:
                p = self.products[self.id_to_idx[pid]]
                if pid in result_pid_set or pid in rec_ids:
                    continue
                if p.get('category') in result_categories:
                    continue
                if primary_aud and p.get('audience') and p.get('audience') != primary_aud:
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
                    return recommendations

        # Semantic complementary pairing for the top primary results
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
                matching_aud = [c for c in candidates if (not aud or c.get('audience') == aud) and c['id'] not in result_pid_set and c['id'] not in rec_ids]
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

        rec_target = None
        # Match "X with (a/matching) Y" or "X and a matching Y"
        m_comp = re.search(r'^(.*?)\s+\bwith\s+(?:an?\s+)?(?:(?:matching|accompanying|complementary)\s+)?(bag|bags|purse|clutch|shoes|heels|sneakers|sandals|boots|belt|scarf|hat|jewelry|jacket|socks|sunglasses)\b(.*)$', raw_query, re.IGNORECASE)
        if m_comp:
            primary_text = m_comp.group(1).strip()
            rec_target = (m_comp.group(2) + ' ' + m_comp.group(3)).strip()
        else:
            m_comp_alt = re.search(r'^(.*?)\s+\b(?:with|and)\s+(?:an?\s+)?(?:matching|accompanying|complementary)\s+(.+)$', raw_query, re.IGNORECASE)
            if m_comp_alt:
                primary_text = m_comp_alt.group(1).strip()
                rec_target = m_comp_alt.group(2).strip()
            else:
                primary_text = raw_query

        inferred_filters, clean_query = parse_prose_constraints(primary_text) if primary_text else ({}, '')
        filters = dict(inferred_filters)
        filters.update(explicit_filters)

        norm_query = normalize_query(clean_query) if clean_query else ''

        if norm_query:
            words = tokens(norm_query)
            expanded = [SYNONYMS.get(w, w) for w in words]
            norm_query = ' '.join(filter(None, expanded))

        corrected_query = self.correct_query(norm_query) if norm_query else ''

        candidates = [p for p in self.products if self.matches(p, filters)]
        candidates_set = {p['id'] for p in candidates}
        scores = {}

        m_mixed = re.split(r'\b(?:and|or)\b', corrected_query, flags=re.IGNORECASE) if corrected_query else []
        is_mixed = len(m_mixed) >= 2 and all(len(p.strip()) >= 3 for p in m_mixed)

        if is_mixed:
            choice_sorted = []
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
        elif corrected_query:
            scores = self._retrieve_query_scores(corrected_query, candidates_set)
            candidates = [p for p in candidates if p['id'] in scores]
        elif not corrected_query and not explicit_filters and not inferred_filters:
            pass

        sort_by = arguments.get('sort_by', 'relevance')
        sort_order = arguments.get('sort_order', 'desc')
        if not is_mixed:
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

        offset, limit = arguments.get('offset', 0), arguments.get('limit', 10)
        results = [{'id': p['id']} for p in candidates[offset:offset + limit]]
        return {'results': results, 'recommendations':
                self.recommend(results, arguments, target_override=rec_target) if results and arguments.get('include_recommendations', False) else []}
