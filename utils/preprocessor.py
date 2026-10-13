"""
Text preprocessing and linguistic analysis utilities for Fake News Detection.
Ensures identical text normalization during both training and real-time inference.
"""
import re
import html

# Common English stop words
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any',
    'are', 'aren', 'arent', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', 'did', 'didn', 'do',
    'does', 'doesn', 'doing', 'don', 'dont', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'hadn', 'has', 'hasn', 'have', 'haven', 'having', 'he', 'hed', 'hell',
    'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows',
    'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'isn', 'isnt', 'it', 'its',
    'itself', 'lets', 'me', 'more', 'most', 'mustn', 'my', 'myself', 'no', 'nor', 'not',
    'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves',
    'out', 'over', 'own', 'same', 'shan', 'she', 'shed', 'shell', 'shes', 'should', 'shouldn',
    'so', 'some', 'such', 'than', 'that', 'thats', 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre',
    'theyve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
    'was', 'wasn', 'we', 'wed', 'well', 'were', 'weren', 'weve', 'what', 'whats', 'when',
    'whens', 'where', 'wheres', 'which', 'while', 'who', 'whos', 'whom', 'why', 'whys',
    'with', 'won', 'wont', 'would', 'wouldn', 'you', 'youd', 'youll', 'youre', 'youve',
    'your', 'yours', 'yourself', 'yourselves'
}

# Source-specific shortcut patterns to prevent data leakage
SOURCE_SHORTCUT_REGEX = re.compile(
    r'^[A-Z\s,]+(?:\([A-Za-z\s]+\))?\s*[-–—]\s*|'  # Datelines like "WASHINGTON (Reuters) - "
    r'\b(?:reuters|associated\s+press|\bap\b|afp|bloomberg)\b',
    re.IGNORECASE
)

# HTML tags and URL patterns
HTML_TAG_REGEX = re.compile(r'<[^>]+>')
URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
SPECIAL_CHAR_REGEX = re.compile(r'[^a-zA-Z\s]')
WHITESPACE_REGEX = re.compile(r'\s+')


def clean_text(text: str, strip_shortcuts: bool = True) -> str:
    """
    Cleans raw text for NLP model training and inference.
    
    Steps:
    1. Unescape HTML entities
    2. Remove HTML tags
    3. Remove URLs and email addresses
    4. Strip publisher datelines/shortcuts (mitigating data leakage)
    5. Remove non-alphabetic symbols (retaining single spaces)
    6. Convert to lowercase
    7. Normalize whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unescape HTML
    text = html.unescape(text)
    
    # Strip HTML tags
    text = HTML_TAG_REGEX.sub(' ', text)
    
    # Strip URLs and Emails
    text = URL_REGEX.sub(' ', text)
    text = EMAIL_REGEX.sub(' ', text)
    
    # Strip source shortcuts (critical to eliminate publisher leakage)
    if strip_shortcuts:
        text = SOURCE_SHORTCUT_REGEX.sub(' ', text)
    
    # Keep only alphabetical characters
    text = SPECIAL_CHAR_REGEX.sub(' ', text)
    
    # Lowercase
    text = text.lower()
    
    # Collapse multiple whitespaces
    text = WHITESPACE_REGEX.sub(' ', text).strip()
    
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Splits text into words and optionally removes stopwords.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    
    words = cleaned.split()
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return words


def get_linguistic_stats(raw_text: str) -> dict:
    """
    Extracts descriptive linguistic statistics for display in UI.
    """
    if not raw_text or not isinstance(raw_text, str):
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'lexical_diversity': 0.0,
            'reading_time_seconds': 0,
            'processed_tokens': 0
        }
    
    char_count = len(raw_text.strip())
    
    # Count sentences based on punctuation
    sentences = [s.strip() for s in re.split(r'[.!?]+', raw_text) if len(s.strip()) > 0]
    sentence_count = max(1, len(sentences))
    
    # Raw word tokens
    raw_words = [w for w in re.findall(r'\b\w+\b', raw_text.lower()) if len(w) > 0]
    word_count = len(raw_words)
    
    # Clean tokens for ML
    tokens = tokenize(raw_text, remove_stopwords=True)
    processed_tokens = len(tokens)
    
    if word_count > 0:
        avg_word_length = round(sum(len(w) for w in raw_words) / word_count, 1)
        unique_words = len(set(raw_words))
        lexical_diversity = round((unique_words / word_count) * 100, 1)
    else:
        avg_word_length = 0.0
        lexical_diversity = 0.0
    
    # Avg adult reading speed: ~200 words per minute
    reading_time_seconds = max(1, round((word_count / 200) * 60)) if word_count > 0 else 0
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'lexical_diversity': lexical_diversity,
        'reading_time_seconds': reading_time_seconds,
        'processed_tokens': processed_tokens
    }
