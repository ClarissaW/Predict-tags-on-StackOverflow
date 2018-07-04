from collections import Counter

""" A FASTER WAY """
################# ONE WAY TO COUNT TAGS (DEFAULT SORT)###################
def count_words(words):
    counts = Counter(words)
    return counts

####################### ANOTHER WAY TO COUNT TAGS ########################

def count(all_tags):
    tags_counts = {}
    unique_tag = set(all_tags)
    for tag in unique_tag:
        tags_counts[tag] = all_tags.count(tag)
    return tags_counts



