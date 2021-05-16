# Open file
read_file = open(r"./english-words-master/words_alpha.txt", 'r')
write_file = open(r"valid_words_A_to_F.txt", 'w+')

char_set = {"a", "b", "c", "d", "e", "f"}
intermediate_list = [word.strip('\n') for word in read_file if set(word.strip('\n')).issubset(char_set)]

final_list = []

for word in intermediate_list:
    valid = True
    for char in char_set:
        if word.count(char) > 1:
            valid = False
    if valid:
        final_list.append(word)

for word in final_list:
    write_file.write(word + "\n")