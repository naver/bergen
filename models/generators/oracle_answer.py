'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
from models.generator.oracle_provenance import OracleProvenance


class OracleAnswer(OracleProvenance):
    """
    Generator whose generation is the label
    Subclass OracleProvenance since it's mostly identical:
    both return the instruction as generation, and
    pre-emptively modify the instruction to include doc or gt label.
    """
    def __init__(self, 
                 model_name: str = None,
                 batch_size: int = 1, 
                 **kwargs
                 ):
        OracleProvenance.__init__(self, model_name=model_name, batch_size=batch_size, **kwargs)

    def format_instruction(self, sample):
        return f"""{sample['label'][0]}"""
