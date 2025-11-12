# import re
# from typing import List
# from rapidfuzz import process, fuzz

# EMAIL_TOKEN_PATTERNS = [
#     (r'\b\(?(at|@)\)?\b', '@'),
#     (r'\b(dot)\b', '.'),
#     (r'\s*@\s*', '@'),
#     (r'\s*\.\s*', '.')
# ]

# def collapse_spelled_letters(s: str) -> str:
#     # Collapse sequences like 'g m a i l' -> 'gmail'
#     tokens = s.split()
#     out = []
#     i = 0
#     while i < len(tokens):
#         # lookahead for sequences of single letters
#         if all(len(t)==1 for t in tokens[i:i+5]) and i+4 <= len(tokens):
#             out.append(''.join(tokens[i:i+5]))
#             i += 5
#         else:
#             out.append(tokens[i])
#             i += 1
#     return ' '.join(out)

# def normalize_email_tokens(s: str) -> str:
#     s2 = s
#     s2 = collapse_spelled_letters(s2)
#     for pat, rep in EMAIL_TOKEN_PATTERNS:
#         s2 = re.sub(pat, rep, s2, flags=re.IGNORECASE)
#     # remove spaces around @ and . inside emails
#     s2 = re.sub(r'\s*([@\.])\s*', r'\1', s2)
#     return s2

# # Numbers: handle 'double nine', 'triple zero', 'oh' for zero
# NUM_WORD = {
#     'zero':'0','oh':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
#     'six':'6','seven':'7','eight':'8','nine':'9'
# }

# def words_to_digits(seq: List[str]) -> str:
#     out = []
#     i = 0
#     while i < len(seq):
#         tok = seq[i].lower()
#         if tok in ('double','triple') and i+1 < len(seq):
#             times = 2 if tok=='double' else 3
#             nxt = seq[i+1].lower()
#             if nxt in NUM_WORD:
#                 out.append(NUM_WORD[nxt]*times)
#                 i += 2
#                 continue
#         if tok in NUM_WORD:
#             out.append(NUM_WORD[tok])
#             i += 1
#         else:
#             # stop on first non-number word in a numeric phrase
#             i += 1
#     return ''.join(out)

# def normalize_numbers_spoken(s: str) -> str:
#     # Replace simple spoken digit sequences with digits
#     tokens = s.split()
#     out = []
#     i = 0
#     while i < len(tokens):
#         # greedy take up to 8 tokens for number
#         window = tokens[i:i+8]
#         wd = words_to_digits(window)
#         if len(wd) >= 2:  # treat as a number
#             out.append(wd)
#             i += len(window)  # move past window
#         else:
#             out.append(tokens[i])
#             i += 1
#     return ' '.join(out)

# def normalize_currency(s: str) -> str:
#     # Replace 'rupees ...' preceding digits with ₹
#     s = re.sub(r'\brupees\s+', '₹', s, flags=re.IGNORECASE)
#     # Insert commas in Indian grouping (basic version) for 5+ digits
#     def indian_group(num):
#         # last 3, then every 2
#         x = str(num)
#         if len(x) <= 3: return x
#         last3 = x[-3:]
#         rest = x[:-3]
#         parts = []
#         while len(rest) > 2:
#             parts.insert(0, rest[-2:])
#             rest = rest[:-2]
#         if rest: parts.insert(0, rest)
#         return ','.join(parts + [last3])
#     def repl(m):
#         raw = re.sub('[^0-9]', '', m.group(0))
#         if not raw: return m.group(0)
#         return '₹' + indian_group(int(raw))
#     s = re.sub(r'₹\s*[0-9][0-9,\.]*', repl, s)
#     return s

# def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = 90) -> str:
#     tokens = s.split()
#     out = []
#     for t in tokens:
#         best = process.extractOne(t, names_lex, scorer=fuzz.ratio)
#         if best and best[1] >= threshold:
#             out.append(best[0])
#         else:
#             out.append(t)
#     return ' '.join(out)

# def generate_candidates(text: str, names_lex: List[str]) -> List[str]:
#     cands = set()
#     t = text
#     # Base clean-ups
#     t1 = normalize_email_tokens(t)
#     t1 = normalize_numbers_spoken(t1)
#     t1 = normalize_currency(t1)
#     t1 = correct_names_with_lexicon(t1, names_lex)
#     cands.add(t1)

#     # Variants
#     # Variant 2: only email normalization
#     t2 = correct_names_with_lexicon(normalize_email_tokens(text), names_lex)
#     cands.add(t2)

#     # Variant 3: currency + numbers only
#     t3 = normalize_currency(normalize_numbers_spoken(text))
#     cands.add(t3)

#     # Variant 4: names only
#     t4 = correct_names_with_lexicon(text, names_lex)
#     cands.add(t4)

#     # ensure original too
#     cands.add(text)

#     # Deduplicate and limit
#     out = list(cands)
#     out = sorted(out, key=lambda x: len(x))[:5]  # simple cap
#     return out


from typing import List, Tuple
import numpy as np
import re

# Optional imports guarded to allow partial environments
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForMaskedLM = None


class PseudoLikelihoodRanker:
    def __init__(self, model_name: str = "distilbert-base-uncased", onnx_path: str = None, device: str = "cpu", max_length: int = 48):
        self.max_length = max_length  # Reduced from 64 to 48 for speed
        self.model_name = model_name
        self.onnx = None
        self.torch_model = None
        self.device = device
        self.tokenizer = None
        
        if onnx_path and ort is not None:
            self._init_onnx(onnx_path)
        elif AutoTokenizer is not None and AutoModelForMaskedLM is not None:
            self._init_torch()
        else:
            raise RuntimeError("Neither onnxruntime nor transformers/torch are available. Please install requirements.")

    def _init_onnx(self, onnx_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.onnx = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

    def _init_torch(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.torch_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.torch_model.eval()
        self.torch_model.to(self.device)

    def _has_valid_email(self, text: str) -> bool:
        """Check if text contains a valid-looking email."""
        email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        return bool(re.search(email_pattern, text))

    def _has_valid_number(self, text: str) -> bool:
        """Check if text contains valid formatted numbers (phone, currency, etc.)."""
        # Check for Indian currency
        if re.search(r'₹\s*[\d,]+', text):
            return True
        # Check for long digit sequences (like phone numbers)
        if re.search(r'\b\d{10,}\b', text):
            return True
        # Check for formatted numbers with commas
        if re.search(r'\b\d{1,3}(,\d{2,3})+\b', text):
            return True
        return False

    def _should_skip_scoring(self, candidate: str) -> bool:
        """
        Short-circuit: if candidate already has valid email/number formatting,
        it's likely the best option - skip expensive model scoring.
        """
        return self._has_valid_email(candidate) or self._has_valid_number(candidate)

    def _score_with_onnx_fast(self, text: str) -> float:
        """
        Fast single-pass scoring using perplexity approximation.
        Instead of masking each token individually, we compute a single forward pass
        and estimate the pseudo-likelihood from the logits.
        """
        toks = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = toks["input_ids"]
        attn = toks["attention_mask"]

        ort_inputs = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attn.astype(np.int64),
        }

        # Single forward pass
        logits = self.onnx.run(None, ort_inputs)[0]  # [1, L, V]
        
        # Compute perplexity-based score (faster approximation)
        L = int(attn[0].sum())
        positions = list(range(1, L - 1))  # Skip [CLS] and [SEP]
        
        if not positions:
            return 0.0
        
        seq = input_ids[0]
        total = 0.0
        
        # For each position, get the logit for the actual token
        for pos in positions:
            token_id = int(seq[pos])
            logits_pos = logits[0, pos, :]
            
            # Log-softmax
            m = logits_pos.max()
            log_probs = logits_pos - m - np.log(np.exp(logits_pos - m).sum())
            
            total += float(log_probs[token_id])
        
        # Average log probability (higher = better)
        return total / len(positions) if positions else 0.0

    def _score_with_onnx_original(self, text: str) -> float:
        """Original per-token masking approach - kept for fallback."""
        toks = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = toks["input_ids"]
        attn = toks["attention_mask"]
        L = int(attn[0].sum())
        positions = list(range(1, L - 1))
        
        if not positions:
            return 0.0

        mask_id = self.tokenizer.mask_token_id
        seq = input_ids[0]
        total = 0.0

        for pos in positions:
            masked = seq.copy()
            orig_token_id = int(masked[pos])
            masked[pos] = mask_id

            ort_inputs = {
                "input_ids": masked[None, :].astype(np.int64),
                "attention_mask": attn.astype(np.int64),
            }

            logits = self.onnx.run(None, ort_inputs)[0]
            logits_pos = logits[0, pos, :]

            m = logits_pos.max()
            log_probs = logits_pos - m - np.log(np.exp(logits_pos - m).sum())
            total += float(log_probs[orig_token_id])

        return total

    def _score_with_torch_fast(self, text: str) -> float:
        """Fast single-pass scoring for PyTorch."""
        import torch
        
        toks = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        input_ids = toks["input_ids"]
        attn = toks["attention_mask"]
        
        with torch.no_grad():
            logits = self.torch_model(**toks).logits  # [1, L, V]
            
            L = int(attn.sum())
            positions = list(range(1, L - 1))
            
            if not positions:
                return 0.0
            
            seq = input_ids[0]
            total = 0.0
            
            for pos in positions:
                token_id = seq[pos].item()
                log_probs = logits[0, pos, :].log_softmax(dim=-1)
                total += log_probs[token_id].item()
            
            return total / len(positions) if positions else 0.0

    def score(self, sentences: List[str]) -> List[float]:
        """Score a list of candidate sentences."""
        scores = []
        
        for s in sentences:
            # Short-circuit: if sentence has valid formatting, give it high score
            if self._should_skip_scoring(s):
                scores.append(999.0)  # Very high score to prioritize it
            else:
                # Use fast scoring
                if self.onnx is not None:
                    scores.append(self._score_with_onnx_fast(s))
                else:
                    scores.append(self._score_with_torch_fast(s))
        
        return scores

    def choose_best(self, candidates: List[str]) -> str:
        """Choose the best candidate from a list."""
        if len(candidates) == 1:
            return candidates[0]
        
        # Quick check: if only one has valid email/number, return it immediately
        valid_candidates = [c for c in candidates if self._should_skip_scoring(c)]
        if len(valid_candidates) == 1:
            return valid_candidates[0]
        
        # Otherwise score all candidates
        scores = self.score(candidates)
        i = int(np.argmax(scores))
        return candidates[i]
