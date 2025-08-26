
from dataclasses import dataclass
from operator import le
import os
import copy
import heapq
import regex as re
from tqdm.auto import tqdm
from typing import BinaryIO
from collections import defaultdict

from functools import partial
from multiprocessing import Pool


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def remove_special_tokens(file_chunk: bytes, special_tokens: list[bytes]) -> list[bytes]:
    """
    Remove special tokens in the file chunk.
    """
    escaped_special_tokens = [re.escape(tok) for tok in special_tokens]
    to_match = b"|".join(escaped_special_tokens)

    split_file = re.split(pattern=to_match, string=file_chunk)
    return split_file



RE_PATTERN = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_worker(args) -> dict[tuple[bytes], int]:
    start, end, file_path, special_tokens = args
    with open(file_path, "rb") as f:
        f.seek(start)
        file_chunk = f.read(end - start)
        print(f"Size of fine chunk: {len(file_chunk)}")

    freq_table: dict[tuple[bytes], int] = defaultdict(int)
    split_file = remove_special_tokens(file_chunk, special_tokens)

    for split_chunk in split_file:
        for match in re.finditer(RE_PATTERN, split_chunk):
            key = match.group()
            key_tokens = tuple(bytes([b]) for b in key)
            freq_table[key_tokens] += 1

    return freq_table


def pretokenize(
    input_path: str, 
    special_tokens: list[str], 
    num_processes: int=8
) -> dict[tuple[bytes], int]:
    print("Starting pretokenization")
    special_tokens_bytes = [s.encode("utf-8") for s in special_tokens]

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # print(f"Found boundaries {boundaries}")

    # Create arguments list with file path
    worker_args = [(start, end, input_path, special_tokens_bytes) 
                   for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    with Pool(processes=num_processes) as pool:
        freq_tables = pool.map(pretokenize_worker, worker_args)

    final_dict: dict[tuple[bytes], int] = {}
    for freq_table in tqdm(freq_tables, desc="Merging dicts..."):
        for k, v in freq_table.items():
            final_dict[k] = final_dict.get(k, 0) + v

    print(f"Pretokenization done. Got {len(final_dict)} groups.")
    return final_dict


@dataclass
class MaxHeapObj():
    def __init__(self, val): self.val = val
    def __lt__(self, other): return self.val > other.val
    def __eq__(self, other): return self.val == other.val
    def __str__(self): return str(self.val)
        

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretok_dict = pretokenize(input_path, special_tokens)

    vocab: dict[int, bytes] = {
        i: bytes([i])
        for i in range(256)
    }
    num_vocab: int = 256
    cached_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    heap = []  # duplicate storage for fast get max
    merges: list[tuple[bytes, bytes]] = []

    def heap_push_new(k: tuple[bytes, bytes]):
        if cached_counts.get(k, 0):
            heapq.heappush(heap, MaxHeapObj((cached_counts[k], k)))

    print("Initializing cached counts...")

    for k, v in pretok_dict.items():
        for i in range(len(k) - 1):
            cached_counts[(k[i], k[i+1])] += v
    
    for k, v in cached_counts.items():
        heapq.heappush(heap, MaxHeapObj((v, k)))

    while num_vocab < vocab_size - len(special_tokens):
        assert num_vocab == len(vocab)

        # Break if there is no more to merge
        if all(len(k) == 1 for k in pretok_dict.keys()):
            break

        # Get new merge
        max_key = None
        while heap:
            heap_obj = heapq.heappop(heap)
            val, key = heap_obj.val

            # only use if 1) value is not stale, and 2) not already merged
            if cached_counts[key] == val and key not in merges:
                max_key = key
                break
        if max_key is None:
            break

        new_word = max_key[0] + max_key[1]
        vocab[num_vocab] = new_word
        num_vocab += 1
        merges.append(max_key)

        # Update pretok_dict and cached_counts
        replacements: dict[tuple[bytes], tuple[bytes]] = {}
        for k, v in pretok_dict.items():
            if max_key in [(k[i], k[i+1]) for i in range(len(k) - 1)]:

                # replace key name
                k_hat_list = []
                i = 0
                while i < len(k):
                    if i == len(k) - 1:
                        k_hat_list.append(k[i])
                        i += 1
                    elif (k[i], k[i+1]) == max_key:
                        k_hat_list.append(new_word)
                        i += 2
                    else:
                        k_hat_list.append(k[i])
                        i += 1

                replacements[k] = tuple(k_hat_list)

                # update cached_counts
                for i in range(len(k) - 1):
                    cached_counts[(k[i], k[i+1])] -= v
                for i in range(len(k_hat_list) - 1):
                    cached_counts[(k_hat_list[i], k_hat_list[i+1])] += v


        for k, k_hat in replacements.items():
            val = pretok_dict[k]
            pretok_dict.pop(k)
            pretok_dict[k_hat] = pretok_dict.get(k_hat, 0) + val

        cached_counts.pop(max_key)

        # update heap
        for word in vocab.values():
            heap_push_new((word, max_key[0]))
            heap_push_new((max_key[1], word))
            if word == new_word:
                heap_push_new((word, word))
            else:
                heap_push_new((word, new_word))
                heap_push_new((new_word, word))

        print(f"Cached count now has {len(cached_counts)} items")
        print(f"Heap now has {len(heap)} items")

    # Fill in the special tokens
    for tok in special_tokens:
        vocab[num_vocab] = tok.encode("utf-8")
        num_vocab += 1

    return vocab, merges


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = copy.deepcopy(vocab)
        for new_special_tok in special_tokens:
            self.vocab[len(vocab)] = new_special_tok

        self.merges = merges
        self.new_special_tokens = special_tokens
        self.all_special_tokens = [v for k, v in self.vocab if k >= len(self.merges)]

        print(f"Tokenizer has {len(self.all_special_tokens)} special tokens in total, {len(self.new_special_tokens)} new special tokens")


    def encode_pretoken(self, pretoken: list[bytes]) -> list[int]:
        """
        Encode a split pre-token containing a list of single bytes.
        """
        




    def encode(self, input: str) -> list[int]:
        escaped_special_tokens = [re.escape(tok) for tok in self.all_special_tokens]
        to_match = b"(" + b"|".join(escaped_special_tokens) + b")"

        split_file = re.split(pattern=to_match, string=input.encode("utf-8"))

        pretokenized_list: list[list[bytes]] = []
        for split_chunk in split_file:
            # pre-tokenize
            for match in re.finditer(RE_PATTERN, split_chunk):
                key = match.group()
                pretokenized_list.append([bytes([b]) for b in key])

        
        





if __name__ == "__main__":
    import pickle

    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=8192,
        special_tokens=["<|endoftext|>"],
    )

    with open("bpe_tokenizer_tsval.pkl", "wb") as f:
        pickle.dump((vocab, merges), f)

