import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')
def synonym_aug(text):
    augmented_text = aug.augment(text)
    augmented_text = ' '.join(augmented_text)  # Join the list of strings into a single string
    return augmented_text