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




import re
from typing import List
from rapidfuzz import process, fuzz

# --- EMAIL NORMALIZATION ---

def collapse_spelled_letters(s: str) -> str:
    """
    Collapse sequences of single letters like:
      'r a m e s h' -> 'ramesh'
    """
    tokens = s.split()
    result = []
    buffer = []

    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            buffer.append(tok)
        else:
            if buffer:
                result.append(''.join(buffer))
                buffer = []
            result.append(tok)
    
    if buffer:
        result.append(''.join(buffer))

    return ' '.join(result)


def normalize_email_tokens(s: str) -> str:
    """Normalize email-related tokens in text."""
    # First collapse spelled letters
    s2 = collapse_spelled_letters(s)
    
    # Fix malformed emails like "rameshcgmail.com" -> "ramesh@gmail.com"
    s2 = re.sub(r'(\w+)c(gmail|yahoo|outlook|hotmail)\.com', r'\1@\2.com', s2, flags=re.IGNORECASE)
    
    # Pattern to detect email-like sequences
    # Look for: word + (at|@) + word + (dot|.) + word
    # But only after "email" or "mail" keywords, or if it contains common domains
    email_pattern = r'(\b\w+)\s*\b(at|@)\b\s*(\w+)\s*\b(dot|\.)\b\s*(\w+\b)'
    
    def email_replacer(match):
        full_match = match.group(0)
        parts = match.groups()
        
        # Check if this is actually an email context
        # Look for common email domains or "email/mail" keyword before this
        domain = parts[2].lower()
        tld = parts[4].lower()
        
        # Common email patterns
        if domain in ['gmail', 'yahoo', 'outlook', 'hotmail', 'icloud'] or tld in ['com', 'org', 'net', 'in', 'co']:
            return f"{parts[0]}@{parts[2]}.{parts[4]}"
        
        # Otherwise, leave as is
        return full_match
    
    s2 = re.sub(email_pattern, email_replacer, s2, flags=re.IGNORECASE)
    
    return s2.strip()


# --- NUMBER NORMALIZATION ---

NUM_WORD = {
    'zero': '0', 'oh': '0', 'one': '1', 'two': '2', 'three': '3', 
    'four': '4', 'five': '5', 'six': '6', 'seven': '7', 
    'eight': '8', 'nine': '9'
}

# Large number words for currency
MAGNITUDE_WORDS = {
    'thousand': 1000,
    'lakh': 100000,
    'lakhs': 100000,
    'crore': 10000000,
    'crores': 10000000,
    'hundred': 100,
    'ten': 10,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,
}

ONES_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
}


def words_to_number(text: str) -> str:
    """Convert words like 'fifty thousand' to '50000'."""
    words = text.lower().split()
    
    total = 0
    current = 0
    
    for word in words:
        if word in ONES_WORDS:
            current += ONES_WORDS[word]
        elif word == 'hundred':
            current = (current or 1) * 100
        elif word in ['thousand', 'lakh', 'lakhs', 'crore', 'crores']:
            multiplier = MAGNITUDE_WORDS[word]
            current = (current or 1) * multiplier
            total += current
            current = 0
    
    total += current
    return str(total) if total > 0 else text


def words_to_digits(seq: List[str]) -> str:
    """Convert a sequence of number words to digits."""
    out, i = [], 0
    
    while i < len(seq):
        tok = seq[i].lower()
        
        # Handle 'double' and 'triple'
        if tok in ('double', 'triple') and i + 1 < len(seq):
            times = 2 if tok == 'double' else 3
            nxt = seq[i + 1].lower()
            if nxt in NUM_WORD:
                out.append(NUM_WORD[nxt] * times)
                i += 2
                continue
        
        # Handle regular number words
        if tok in NUM_WORD:
            out.append(NUM_WORD[tok])
        
        i += 1
    
    return ''.join(out)


def normalize_numbers_spoken(s: str) -> str:
    """
    Convert spoken numbers like:
    - 'double nine eight seven six' -> '99876'
    - 'nine eight seven' -> '987'
    """
    tokens = s.split()
    result, buffer = [], []

    for tok in tokens:
        low = tok.lower()
        
        if low in NUM_WORD or low in ('double', 'triple'):
            buffer.append(low)
        else:
            if buffer:
                digits = words_to_digits(buffer)
                if digits:
                    result.append(digits)
                buffer = []
            result.append(tok)
    
    # Don't forget remaining buffer
    if buffer:
        digits = words_to_digits(buffer)
        if digits:
            result.append(digits)

    # Merge adjacent digit blocks
    merged = []
    for tok in result:
        if merged and merged[-1].isdigit() and tok.isdigit():
            merged[-1] += tok
        else:
            merged.append(tok)
    
    return ' '.join(merged)


# --- CURRENCY NORMALIZATION ---

def normalize_currency(s: str) -> str:
    """
    Normalize spoken rupee amounts like:
    'rupees fifty thousand' -> '₹50,000'
    'rupee five lakh' -> '₹5,00,000'
    """
    # Pattern: rupees/rupee followed by number words
    pattern = r'\b(rupees?)\s+([a-z\s]+?)(?=\s+(?:and|[,.]|$)|$)'
    
    def currency_replacer(match):
        amount_text = match.group(2).strip()
        # Try to convert to number
        amount_num = words_to_number(amount_text)
        
        if amount_num.isdigit():
            # Format with Indian grouping
            return f"₹{indian_group(amount_num)}"
        else:
            # Couldn't convert, return original
            return match.group(0)
    
    s = re.sub(pattern, currency_replacer, s, flags=re.IGNORECASE)
    
    # Also handle existing ₹ symbols with numbers
    def indian_format(m):
        raw = re.sub(r'[^\d]', '', m.group(0))
        if not raw:
            return m.group(0)
        return '₹' + indian_group(raw)
    
    s = re.sub(r'₹\s*\d[\d,\.]*', indian_format, s)
    
    return s


def indian_group(num: str) -> str:
    """Format number with Indian comma grouping."""
    num = str(num).replace(',', '')
    if len(num) <= 3:
        return num
    
    last3 = num[-3:]
    rest = num[:-3]
    groups = []
    
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    
    if rest:
        groups.insert(0, rest)
    
    return ','.join(groups + [last3])


# --- NAME CORRECTION ---

def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = 90) -> str:
    """
    Correct name spelling mistakes using fuzzy matching.
    """
    if not names_lex:
        return s
    
    tokens = s.split()
    result = []
    
    for t in tokens:
        # Only attempt correction on capitalized words or all-lowercase words
        if t and len(t) > 2 and (t[0].isupper() or t.islower()):
            best = process.extractOne(t, names_lex, scorer=fuzz.ratio)
            if best and best[1] >= threshold:
                result.append(best[0])
            else:
                result.append(t)
        else:
            result.append(t)
    
    return ' '.join(result)


# --- CANDIDATE GENERATOR ---

def generate_candidates(text: str, names_lex: List[str] = None) -> List[str]:
    """
    Generate multiple cleaned versions (candidates) of the ASR text.
    Returns up to 5 unique normalized candidates.
    """
    if names_lex is None:
        names_lex = []
    
    candidates = set()

    # 1️⃣ Full normalization chain
    t1 = text
    t1 = normalize_email_tokens(t1)
    t1 = normalize_numbers_spoken(t1)
    t1 = normalize_currency(t1)
    t1 = correct_names_with_lexicon(t1, names_lex)
    candidates.add(t1)

    # 2️⃣ Email + Names
    t2 = normalize_email_tokens(text)
    t2 = correct_names_with_lexicon(t2, names_lex)
    candidates.add(t2)

    # 3️⃣ Numbers + Currency only
    t3 = normalize_numbers_spoken(text)
    t3 = normalize_currency(t3)
    candidates.add(t3)

    # 4️⃣ Currency only
    t4 = normalize_currency(text)
    candidates.add(t4)

    # 5️⃣ Original text
    candidates.add(text)

    # Sort by length and return top 5
    final = sorted(list(candidates), key=len)[:5]
    return final


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "my name is ramesh and email is rameshcgmail.com rupees fifty thousand",
        "my name is ramesh and email is r a m e s h at gmail dot com rupees fifty thousand",
        "contact me at john dot doe at company dot com",
        "phone number is double nine eight seven six",
    ]
    
    names_lexicon = ["Ramesh", "John", "Doe"]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        candidates = generate_candidates(text, names_lexicon)
        print("Candidates:")
        for i, cand in enumerate(candidates, 1):
            print(f"  {i}. {cand}")
