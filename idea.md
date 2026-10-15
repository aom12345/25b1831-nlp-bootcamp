(1) Failure targeted was english words being mispronounced. (english_word_error, hinglish), e.g. val_hg_01 (CER 0.452) and val_hg_06 (CER 0.318).

(2) The main idea was to break down english words and then reconstruct them in devanagri script because the speech model was trained in that. A additional hypothesis was that english  words might take a different amount of time to say then hindi words hence we should add some time scaling of sorts.

(3) On the 40 eval items I expect Hinglish CER to drop around 60-80% compared to the Task 2 control, less than validation since most eval words are new.
Risks: English/hindi exceptions, new spelling patterns.

(4) Using self made rules to convert to hindi script worked(cer=0.039) which came close to what we get by using the provided hindi spellings(cer=0.026). Isolating the word to make it sound clear did not improve this. I could not get the speed change to work with in the competition rules since it required speed change per syllable. Hinglish CER went from 0.309 to 0.039, overall CER 0.242 to 0.061, SIM 0.841.
