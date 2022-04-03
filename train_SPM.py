import sentencepiece as spm

spm.SentencePieceTrainer.train('--input=rev_data.txt --model_prefix=amazon --vocab_size=2000')

sp = spm.SentencePieceProcessor()
sp.load('amazon.model')


