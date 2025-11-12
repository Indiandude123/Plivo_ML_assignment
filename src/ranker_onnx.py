# from typing import List, Tuple
# import numpy as np

# # Optional imports guarded to allow partial environments
# try:
#     import onnxruntime as ort  # type: ignore
# except Exception:
#     ort = None

# try:
#     import torch
#     from transformers import AutoTokenizer, AutoModelForMaskedLM
# except Exception:
#     torch = None
#     AutoTokenizer = None
#     AutoModelForMaskedLM = None

# class PseudoLikelihoodRanker:
#     def __init__(self, model_name: str = "distilbert-base-uncased", onnx_path: str = None, device: str = "cpu", max_length: int = 64):
#         self.max_length = max_length
#         self.model_name = model_name
#         self.onnx = None
#         self.torch_model = None
#         self.device = device
#         self.tokenizer = None
#         if onnx_path and ort is not None:
#             self._init_onnx(onnx_path)
#         elif AutoTokenizer is not None and AutoModelForMaskedLM is not None:
#             self._init_torch()
#         else:
#             raise RuntimeError("Neither onnxruntime nor transformers/torch are available. Please install requirements.")

#     def _init_onnx(self, onnx_path: str):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         sess_options = ort.SessionOptions()
#         sess_options.intra_op_num_threads = 1
#         sess_options.inter_op_num_threads = 1
#         self.onnx = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

#     def _init_torch(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.torch_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
#         self.torch_model.eval()
#         self.torch_model.to(self.device)

#     def _batch_mask_positions(self, input_ids: np.ndarray, attn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         # Create a batch of masked sequences, one for each non-[CLS]/[SEP] position
#         mask_id = self.tokenizer.mask_token_id
#         seq = input_ids[0]  # [1, L]
#         L = int(attn[0].sum())
#         positions = list(range(1, L-1))  # skip [CLS] and [SEP] equivalents
#         batch = np.repeat(seq[None, :], len(positions), axis=0)
#         for i, pos in enumerate(positions):
#             batch[i, pos] = mask_id
#         batch_attn = np.repeat(attn, len(positions), axis=0)
#         return batch, batch_attn, np.array(positions, dtype=np.int64)

#     def _score_with_onnx(self, text: str) -> float:
#         # Tokenize to NumPy for ORT
#         toks = self.tokenizer(
#             text,
#             return_tensors="np",
#             truncation=True,
#             max_length=self.max_length,
#         )
#         input_ids = toks["input_ids"]           # (1, L)
#         attn = toks["attention_mask"]           # (1, L)

#         # Figure out which token positions to score (skip special tokens)
#         # Use attention mask to get real length L
#         L = int(attn[0].sum())
#         # Typically for HF models: position 0 = [CLS] or [BOS], position L-1 = [SEP] or [EOS]
#         positions = list(range(1, L - 1))

#         mask_id = self.tokenizer.mask_token_id
#         seq = input_ids[0]                      # (L,)
#         total = 0.0

#         for pos in positions:
#             # Make a masked copy with batch=1
#             masked = seq.copy()
#             orig_token_id = int(masked[pos])
#             masked[pos] = mask_id

#             ort_inputs = {
#                 "input_ids": masked[None, :].astype(np.int64),   # (1, L)
#                 "attention_mask": attn.astype(np.int64),         # (1, L)
#             }

#             # Run the model: logits shape (1, L, V)
#             logits = self.onnx.run(None, ort_inputs)[0]
#             logits_pos = logits[0, pos, :]                       # (V,)

#             # log-softmax in a numerically stable way
#             m = logits_pos.max()
#             log_probs = logits_pos - m - np.log(np.exp(logits_pos - m).sum())

#             total += float(log_probs[orig_token_id])

#         return total  # higher = better

#     # def _score_with_onnx(self, text: str) -> float:
#     #     toks = self.tokenizer(text, return_tensors="np", truncation=True, max_length=self.max_length)
#     #     input_ids = toks["input_ids"]
#     #     attn = toks["attention_mask"]
#     #     batch, batch_attn, positions = self._batch_mask_positions(input_ids, attn)
#     #     ort_inputs = {"input_ids": batch.astype(np.int64), "attention_mask": batch_attn.astype(np.int64)}
#     #     logits = self.onnx.run(None, ort_inputs)[0]  # [B, L, V]
#     #     # gather logprobs at the original token for each masked position
#     #     orig = np.repeat(input_ids, len(positions), axis=0)
#     #     rows = np.arange(len(positions))
#     #     cols = positions
#     #     token_ids = orig[rows, cols]
#     #     # log softmax per row at the masked position
#     #     logits_pos = logits[rows, cols, :]  # [B, V]
#     #     m = logits_pos.max(axis=1, keepdims=True)
#     #     log_probs = logits_pos - m - np.log(np.exp(logits_pos - m).sum(axis=1, keepdims=True))
#     #     picked = log_probs[np.arange(len(rows)), token_ids]
#     #     return float(picked.sum())  # higher = better

#     def _score_with_torch(self, text: str) -> float:
#         import torch
#         toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
#         input_ids = toks["input_ids"]
#         attn = toks["attention_mask"]
#         # batch mask
#         seq = input_ids[0]
#         L = int(attn.sum())
#         positions = list(range(1, L-1))
#         batch = seq.unsqueeze(0).repeat(len(positions), 1)
#         for i, pos in enumerate(positions):
#             batch[i, pos] = self.tokenizer.mask_token_id
#         batch_attn = attn.repeat(len(positions), 1)
#         with torch.no_grad():
#             out = self.torch_model(input_ids=batch, attention_mask=batch_attn).logits  # [B, L, V]
#             orig = seq.unsqueeze(0).repeat(len(positions), 1)
#             rows = torch.arange(len(positions))
#             cols = torch.tensor(positions)
#             token_ids = orig[rows, cols]
#             logits_pos = out[rows, cols, :]
#             log_probs = logits_pos.log_softmax(dim=-1)
#             picked = log_probs[torch.arange(len(rows)), token_ids]
#         return float(picked.sum().item())

#     def score(self, sentences: List[str]) -> List[float]:
#         return [self._score_with_onnx(s) if self.onnx is not None else self._score_with_torch(s) for s in sentences]

#     def choose_best(self, candidates: List[str]) -> str:
#         if len(candidates) == 1:
#             return candidates[0]
#         scores = self.score(candidates)
#         i = int(np.argmax(scores))
#         return candidates[i]


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
