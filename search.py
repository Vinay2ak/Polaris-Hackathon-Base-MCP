"""Baseline catalog search. Improve retrieval here without changing the schemas."""
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import bm25s
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parent
INPUT_SCHEMA = json.loads((ROOT / 'schemas/input.schema.json').read_text(encoding='utf-8'))
OUTPUT_SCHEMA = json.loads((ROOT / 'schemas/output.schema.json').read_text(encoding='utf-8'))
VALIDATOR = Draft202012Validator(INPUT_SCHEMA)
STRING_FILTERS = ('category', 'product_type', 'color', 'audience', 'section', 'pattern')

STOP_OR_HELPER_WORDS = {
    'under', 'above', 'below', 'between', 'less', 'than', 'max', 'min',
    'inr', 'rs', 'rupees', 'for', 'with', 'a', 'an', 'the', 'and', 'or',
    'to', 'in', 'of', 'at', 'by', 'from', 'on', 'into', 'is', 'it',
    'ke', 'liye', 'ka', 'ki', 'ko', 'me', 'mein', 'se', 'wala', 'wali', 'wale', 'par',
    'i', 'am', 'looking', 'searching', 'need', 'want', 'please', 'show', 'find'
}


def validate_arguments(arguments):
    # Reject non-JSON floating point values even when called directly from Python.
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


@lru_cache(maxsize=2048)
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
        corpus = [' '.join(str(p.get(k) or '') for k in ('name', 'description', *STRING_FILTERS)) for p in self.products]
        if corpus:
            corpus_tokens = bm25s.tokenize(corpus)
            self.retriever = bm25s.BM25()
            self.retriever.index(corpus_tokens)
        else:
            self.retriever = None

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def relevance(self, product, query_tokens):
        """BM25 relevance score for a product given query tokens."""
        if not self.retriever or not query_tokens:
            return 0.0
        idx = self.id_to_idx.get(product['id'])
        if idx is None:
            return 0.0
        q_tokens = bm25s.tokenize([' '.join(query_tokens)])
        if not q_tokens or len(q_tokens[0]) == 0:
            return 0.0
        docs, scores = self.retriever.retrieve(q_tokens, k=len(self.products))
        matches = scores[0][docs[0] == idx]
        return float(matches[0]) if len(matches) > 0 else 0.0

    def recommend(self, results, arguments):
        """Student extension: return up to five complementary product references."""
        return []

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

    def search(self, arguments):
        validate_arguments(arguments)
        query_str = arguments.get('query', '').strip()
        q_words = tokens(query_str) if query_str else ()
        filters = arguments.get('filters', {})
        candidates = [p for p in self.products if self.matches(p, filters)]
        scores = {}
        if query_str and self.retriever and len(self.products) > 0:
            content_words = [w for w in q_words if w not in STOP_OR_HELPER_WORDS and not w.isdigit()]
            vocab = self.retriever.vocab_dict
            unknown_words = [w for w in content_words if w not in vocab]
            if content_words and (len(unknown_words) == len(content_words) or (len(content_words) <= 3 and len(unknown_words) > 0)):
                candidates = []
            else:
                q_tokens = bm25s.tokenize([query_str])
                if q_tokens and len(q_tokens[0]) > 0:
                    docs, score_arr = self.retriever.retrieve(q_tokens, k=len(self.products))
                    s = score_arr[0]
                    d = docs[0]
                    if len(s) > 0 and s[0] > 0:
                        # For multi-word queries, verify at least top candidates have multi-term relevance
                        required_overlap = min(2, len(content_words)) if len(content_words) <= 4 else 2
                        top_candidates = d[:min(5, len(d))]
                        max_overlap = 0
                        for doc_idx in top_candidates:
                            pid = self.products[doc_idx]['id']
                            overlap = sum(w in self.words[pid] for w in content_words)
                            if overlap > max_overlap:
                                max_overlap = overlap
                        if max_overlap >= required_overlap:
                            max_s = float(s[0])
                            cutoff = 0.4 * max_s
                            for doc_idx, score in zip(d, s):
                                if score < cutoff:
                                    break
                                scores[self.products[doc_idx]['id']] = float(score)
                candidates = [p for p in candidates if p['id'] in scores]

        sort_by = arguments.get('sort_by', 'relevance')
        sort_order = arguments.get('sort_order', 'desc')
        if sort_by == 'relevance':
            if query_str:
                if sort_order == 'asc':
                    candidates.sort(key=lambda p: (scores[p['id']], p['id']))
                else:
                    candidates.sort(key=lambda p: (-scores[p['id']], p['id']))
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
                self.recommend(results, arguments) if results and arguments.get('include_recommendations', False) else []}
