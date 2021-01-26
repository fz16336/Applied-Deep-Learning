LMC
run 0 - fresh run with adam and adding dropout to conv2 and removing bnorm after fc1 (2952181)
run 1 - now with max pooling instead of stride on conv4 (2952878)
run 2 - using loads of padding with the strides in the paper (2953195) - failed run
run 3 - reverting to normal code but with main function at bottom (2953196)
run 4 - switching height and width and using strides + padding... (2953228) - slightly recuded accuracy and overfitting. Worked better than expected
run 5 - Data:(C,H,W) and normal padding with no strides (2953230) - set as new baseline config
run 6 - "" and removed bias params (2953276) - did not really make much of a difference, but new baseline config
run 7 - "" but with AdamW (2953336)
run 8 - "" but with SGD (2953408)
run 9 - baseline with Adam and Dropout2d (2953418)
run 10 - "" with L2 reg (2953425) -meh 
run 11 - Adam, Dropout1D and L2reg (2953453) - meh
run 12 - Adam, dropout1D, l2reg and last dropout removed (2953571) -meh 
run 13 - "" with l2reg smaller (2953574) - meh
run 14 - "" without l2 reg (2953582)
run 15 - "" and with smaller beta2 for adam (2953583)
run 16 - "" normal beta but l2reg = 0.0005 (2953608) - seems similar to the best config so far
run 17 - "" repeat (2953682)
run 18 - weight_decay(l2reg)=0.0006 (2953684)
run 19 - '' repeat (2953685)
run 20 - l2=0.0004 (2953686)
run 21 - "" repeat (2953687)
run 22 - l2=0.0003 (2953688)
run 23 - "" (2953689)
run 24 - l2=0.0007 (2953691)
run 25 - "" (2953692)
run 26 - l2=0.1 (2953699)
run 27 - l2=0.004 (2953789)
run 28 - "" (2953790)
run 29 - failed (2954995)
run 30 - adding weight for unbalanced data l2=0.004 (2955006)
run 31 - adding weight for unballanced data l2=.0005 (2955013)
run 32 - adding weights (inversed now) l2=0.0005 (2955169)
run 33 - "" and using proper testing evaluation (2955307)
run 34 - "" repeat (2955308)
run 35 - "" (2955311)
run 36 - "" SGD l2=0.004 (2956619)
run 37 - '' SGD l2=0.004 (2956701)

MC
run 0 - fresh run with adam and adiding dropout to conv2 and removing bnorm after fc1 (2952123)
run 1 - run 1 - now with max pooling instead of stride on conv4 (2952877)
run 2 - same as LMC run 6 (2953277)
run 3 - baseline with Adam and Dropout2d (2953419)
run 4 - "" with L2 reg (2953424)
run 5 - Adam, Dropout1D and L2reg (2953454)
run 6 - equiv to run 33 (2955330)
run 7 - "" (2955331)
run 8 - "" (2955332)

MLMC
run 0 - fresh run with adam and adding dropout to conv2 and removing bnorm after fc1 (2952829)
run 1 - now with max pooling instead of stride on conv4 (2952876)
run 2 - same as LMC run 6 (2953278)
run 3 - baseline with Adam and Dropout2d (2953420)
run 4 - "" with L2 reg (2953423)
run 5 - Adam, Dropout1D and L2reg (2953455)
run 6 - equiv to run 33 (2955333)
run 7 - "" (2955334)
run 8 - "" (2955335)
run 9 - "" SGD l2=0.0005 (2956363)
run 10 - "" SGD l2=0.004 (2956610)

