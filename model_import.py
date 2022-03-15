# Ich denke, dass beste ist mit dem ULMfit model von tensorflow_hub zu arbeiten 
# documentation: https://bitbucket.org/edroneteam/tf2_ulmfit/src/master/
# vorher halt noch sich die dateien ziehen und die requirements installieren -> habs mal noch nich commited wegen upload


import tensorflow as tf
from tf2_ulmfit.ulmfit_tf2 import tf2_ulmfit_encoder

spm_args = {'spm_model_file': 'tf2_ulmfit/enwiki100-toks-sp35k-cased.model',
            'add_bos': True,
            'add_eos': True,
            'fixed_seq_len': 70}
lm_num, encoder_num, mask_num, spm_encoder_model = tf2_ulmfit_encoder(spm_args=spm_args,
                                                                       fixed_seq_len=70)
encoder_num.summary()