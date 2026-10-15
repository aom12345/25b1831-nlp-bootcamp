Q1
Better: val_hg_01 went from 0.452 to 0.032, all three English words (office, internet, slow) come out right. val_hg_05 went from 0.283 to 0.000. Worse: val_hi_02 (0.000 to 0.078) and val_lg_01 (0.024 to 0.047). Running val_hi_02 with seeds 0, 1, 2 gave 0.078, 0.000, 0.020. It got slightly worse due to randomness.

Q2
The model could say English words once they were written in the script it was trained on, it didn't get better for some because we wern't focusing on those issues. I also learned small CER differences are mostly randomness based on processing, and two recognizers can disagree on the same audio. I would test every idea with a few seeds and a second recognizer from the start, and hold back some words when writing rules.

Q3
One of the things i wanted to try is editing the audio speed maybe this can be done by processing it in chuncks then joining them. I would also try a stronger guidance setting (cfg_strength 3.0 instead of 2.0), which pushes the model to follow the text more. A small test lowered CER on both recognizers, but it needs a full 24 item check.
