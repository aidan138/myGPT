from collections import defaultdict
from uuid import uuid4
from collections import Counter
import heapq
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import cpu_count
import concurrent.futures
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Pretoken():
  def __init__(self, tokens, freq = 1):
    self.tokens = tokens # list of token ids
    self.freq = freq
    self.id = uuid4()

  def __hash__(self):
    return hash(self.id)

  def __eq__(self, x):
    if isinstance(x, Pretoken):
      return self.id == x.id
    raise NotImplementedError
  
  def _get_pairs(self):
    for pair in zip(self.tokens[:-1], self.tokens[1:]):
      yield pair

  def merge(self, merge_pair, new_token):
    tokens = []
    i = 0

    # Track the pairs modified and the number of times its modified
    prev_pair_counts = Counter(self._get_pairs())

    while i < len(self.tokens):
      if i + 1 < len(self.tokens) and (self.tokens[i], self.tokens[i+1]) == merge_pair:
        tokens.append(new_token)
        i+=2

      else:
        tokens.append(self.tokens[i])
        i+=1
    
    reduced_pairs = {}
    new_pairs = {}

    self.tokens = tuple(tokens)
    new_pair_counts = Counter(self._get_pairs())
    for pair in new_pair_counts | prev_pair_counts:
      diff = new_pair_counts.get(pair,0) - prev_pair_counts.get(pair, 0) if pair != merge_pair else 0
      if diff > 0:
        new_pairs[pair] = diff * self.freq
      elif diff < 0:
        reduced_pairs[pair] = -diff * self.freq

    return reduced_pairs, new_pairs

def get_stats(parent: Pretoken, stats: dict[tuple, int] = {}, bp_to_pt: dict[tuple, set] = {}):
    ids = parent.tokens
    increment = parent.freq
    for id_pair in zip(ids[:-1], ids[1:]):
        stats[id_pair] = stats.get(id_pair, 0) + increment # For sure works
        bp_to_pt[id_pair].add(parent)
    return stats

def handle_reduction(pair, freq, byte_freqs, bp_to_pt, parent):
  byte_freqs[pair] -= freq


def handle_addition(pair, freq, byte_freqs, bp_to_pt, parent):
  byte_freqs[pair] = byte_freqs.get(pair, 0) + freq
  bp_to_pt[pair].add(parent)

def count_chunk(args) -> Counter:
  input_path, start, end, spec_token_pattern = args
  print(f"working on chunk {start} - {end}")
  with open(input_path, 'rb') as f:
    # Read the chunk from the file
    f.seek(start)
    chunk = f.read(end-start).decode('utf-8', errors='ignore')

  # Removes the last item if ends on a special token
  pretoken_counts = Counter()
  documents = re.split(spec_token_pattern, chunk)
  for document in documents:
    split_pretokens = re.finditer(PAT, document)
    pretoken_counts += Counter(tuple(pretoken.group().encode('utf-8')) for pretoken in split_pretokens)
    #print(pretoken_counts)
  print(f"Finished counting chunk {start} - {end}")

  return pretoken_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
  assert vocab_size >= (len(special_tokens) + 256)
  special_tok_pattern = '|'.join(re.escape(tok) for tok in special_tokens)
  special_start = vocab_size - len(special_tokens)
  special_token_dict = {special_start: special_tokens[i].encode('utf-8') for i in range(len(special_tokens))}

  with open(input_path, 'rb') as f:
    num_processes = cpu_count()
    chunk_boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode('utf-8'))

  pretoken_counts = Counter()
  print(f"Starting ")
  with concurrent.futures.ProcessPoolExecutor() as executor:
    arguments = [
      (input_path, chunk_boundaries[i], chunk_boundaries[i+1], special_tok_pattern) for i in range(len(chunk_boundaries)-1)
    ]

    results = [executor.submit(count_chunk, arg) for arg in arguments]
    for f in concurrent.futures.as_completed(results):
      pretoken_counts += f.result()
    
  pretokens = [Pretoken(tok, freq) for tok, freq in pretoken_counts.items()]
  byte_freqs = {} # byte_pair : freq
  bp_to_pt = defaultdict(set)# byte_pair : set(parent Pretokens)

  for pretoken in pretokens:
    get_stats(pretoken, byte_freqs, bp_to_pt)

  heap = [(-freq, tuple(-x for x in bp)) for bp, freq in byte_freqs.items()] # Inverse values for min priority queue
  heapq.heapify(heap)  

  # Non parrallelizable merging
  num_merges = vocab_size - len(special_tokens) - 256
  current_token = 256

  # Establish return values
  vocab = {i: bytes([i]) for i in range(256)} | special_token_dict  # Token: bytes representation
  merges = {} # (token, token) -> token

  # Main merge loop
  for _ in range(num_merges):

    while heap:
      neg_freq, neg_pair = heapq.heappop(heap)
      pair = tuple(-x for x in neg_pair)

      # Check for a valid merge
      if neg_freq != 0 and pair in byte_freqs and -neg_freq == byte_freqs[pair]:
        
        vocab[current_token] = vocab[pair[0]] + vocab[pair[1]]
        merges[pair] = current_token

        byte_freqs.pop(pair)

        parents = bp_to_pt.pop(pair)

        for parent in parents:
          reduced_pairs, new_pairs = parent.merge(pair, current_token)

          for mod_pair, mod_freq in reduced_pairs.items():
            handle_reduction(mod_pair, mod_freq, byte_freqs, bp_to_pt, parent)
            heapq.heappush(heap, (-byte_freqs[mod_pair], tuple(-x for x in mod_pair)))

          for mod_pair, mod_freq in new_pairs.items():
            handle_addition(mod_pair, mod_freq, byte_freqs, bp_to_pt, parent)
            heapq.heappush(heap, (-byte_freqs[mod_pair], tuple(-x for x in mod_pair)))

        current_token += 1
        break
  
  for k, v in merges.items():
    try:
        a = vocab[k[0]].decode('utf-8')
    except UnicodeDecodeError:
        a = str(vocab[k[0]])
    try:
        b = vocab[k[1]].decode('utf-8')
    except UnicodeDecodeError:
        b = str(vocab[k[1]])
    try:
        ab = vocab[v].decode('utf-8')
    except UnicodeDecodeError:
        ab = str(vocab[v])
    print(f"{a}<|merged|>{b} -> {ab} ({v})")

  return vocab, merges 



if __name__ == "__main__":
  train_bpe("data\TinyStoriesV2-GPT4-valid.txt", 500 , ['<|endoftext|>'])