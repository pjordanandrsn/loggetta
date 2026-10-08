### Fix legacy observation licensing with nullable plan metadata

Receipts with a null plan or null constraints retain their existing `licensed_for`
scope instead of raising an exception. Raw DQ7/DQ9/DQ10 records and explicit DQ10
policy receipts remain excluded from observation import (#37).
