### Plans: a model with no MoE layers is refused by the MoE backend at admission

experts4bit-qlora describes some dense families (gemma4 among them) as a MoE topology with zero MoE layers. The MoE
backend admitted that description, so it owned the plan, and the plan was refused later at pricing with "no fused expert
stack at the loader's expert path", plus a note that experts still train. The dense backend's own refusal never showed.
The MoE backend now refuses a topology without expert stacks ("no MoE layers (0/N): not a Mixture-of-Experts model").
The dense backend owns the model when it can describe it. When it cannot, as with gemma4 today, the plan lists each
backend's own reason (#78).
