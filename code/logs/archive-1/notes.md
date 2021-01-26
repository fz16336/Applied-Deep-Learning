LMC
run 0 - adamw optimiser dropout on fcout
run 1 - adamw without dropout on fcout
run 2 - SGD without dropout on fcout (2950272)
run 3 - adam with dropout on fcout (2951056)
run 4 - adam with dropout on fcout and initialised conv3 & conv4 layers (2951085) - timed-out
run 5 - " " with more walltime (2951222)
run 6 - SGD "" (2951571)
run 7 - AdamW with new and improved code (2951995)

MC
run 0 - adamw with dropout on fcout (2950277)
run 1 - adam init conv3 and conv4 (2951566)

-------------------------------
*New and improved code*
run 0 - LMC, SGD (2951572)
run 1 - LMC, Adam (2951769)
run 2 - LMC, Adam, check if switch back form eval mode to train mode code line position mattered (2951823) - it didn't
run 3 - LMC, Adam, testing if sigmoid function was the issue (2951824) - it wasn't
