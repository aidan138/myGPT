import torch
from cs336_basics.nn.utils import softmax
from cs336_basics.tokenizers.pretrained_tokenizer import PretrainedTokenizer
from cs336_basics.nn.layers import TransformerLM
from cs336_basics.train_utils import load_checkpoint

tiny_stories_files = [
        r"cs336_basics\tokenizers\trained_bpes\tiny_stories_vocab.pkl",
        r"cs336_basics\tokenizers\trained_bpes\tiny_stories_merges.pkl",
        ['<|endoftext|>']
    ]



def decode(model: TransformerLM, tokenizer: PretrainedTokenizer, prompt: str, max_tokens: int, temperature: float = 0, p_samples = .90):

    tokens = torch.Tensor(tokenizer.encode(prompt)).unsqueeze(0) # 1, N
    model.eval()
    for _ in range(max_tokens):
        logits = model(prompt) # 1, V
        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            tmp_scaled = logits / temperature
            pred_probs = softmax(tmp_scaled)
            



def main():

    load_checkpoint()
    return

if __name__ == '__main__':
    tokenizer = PretrainedTokenizer.from_files(*tiny_stories_files)
    main()