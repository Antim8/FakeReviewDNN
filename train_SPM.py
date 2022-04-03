import sentencepiece as spm

spm.SentencePieceTrainer.train('--input=rev_data.txt --model_prefix=amazon --vocab_size=4000')

sp = spm.SentencePieceProcessor()
sp.load('amazon.model')


