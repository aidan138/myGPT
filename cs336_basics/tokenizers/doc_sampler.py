from cs336_basics.tokenizers.pretrained_tokenizer import PretrainedTokenizer
import regex as re

def main():
    tokenizer = PretrainedTokenizer.from_files(
        r"cs336_basics\tokenizers\trained_bpes\tiny_stories_vocab.pkl",
        r"cs336_basics\tokenizers\trained_bpes\tiny_stories_merges.pkl",
        ['<|endoftext|>']
    )

    datasets = [r'data\TinyStoriesV2-GPT4-train.txt', r'data\owt_train.txt']
    for dataset_path in datasets:
    
        with open(dataset_path, 'r', encoding='utf-8') as f:
            txt = f.read(1000000) # Read the 1st MB
            documents = re.split(tokenizer.spec_pat, txt)
            for i in range(10):
                document = documents[i]
                #print(document)
                print(f"Document {i} has compression ratio: {((len(list([b for b in document.encode('utf-8')])) / len(tokenizer.encode(document))) * 100):.3f}%")
    

if __name__ == '__main__':
    main()