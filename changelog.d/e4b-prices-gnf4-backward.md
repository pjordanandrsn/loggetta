### The grouped_nf4 backward line defers to experts4bit-qlora when it prices the branch itself

experts4bit-qlora#1532 (for #1526) prices the grouped_nf4 MoE backward inside its own `activations` item. Loggetta now
checks whether the installed experts4bit-qlora has that branch (`e4b_prices_gnf4_backward`), and if so adds no line of
its own. For an older release it keeps its line, as before. Totals are never counted twice either way: against the new
estimate, the old line's excess over `activations` is not positive, and `tests/test_gnf4_training_terms.py` checks that.
