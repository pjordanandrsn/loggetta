### Price small frozen weights retained during dense streaming

Count frozen weight bytes that the unchanged offload engine keeps on the device
below its streaming threshold. Keep their quantization statistics and trainable
LoRA accounting separate. This corrects structural pricing without changing
runtime behavior, reserve/context hypotheses or observation licensing (#39).
