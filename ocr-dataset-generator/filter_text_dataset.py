text = open('data/gothicliterature.txt', 'r', encoding='utf-8').read()

text = text.replace('&c.', 'etc.')
text = text.replace('&', 'and')

text = text.replace('*', '')
text = text.replace('_', '')
text = text.replace('£', '')

text = text.replace('[', '(')
text = text.replace(']', ')')
text = text.replace('{', '(')
text = text.replace('}', ')')

text = text.replace('“', '"')
text = text.replace('”', '"')
text = text.replace('‘', '\'')
text = text.replace('’', '\'')

text = text.replace('—', '--')

text = text.replace('à', 'a')
text = text.replace('á', 'a')
text = text.replace('â', 'a')

text = text.replace('æ', 'ae')

text = text.replace('è', 'e')
text = text.replace('é', 'e')
text = text.replace('ê', 'e')
text = text.replace('ë', 'e')

text = text.replace('ï', 'i')

text = text.replace('ô', 'o')
text = text.replace('ö', 'o')

i = 0
while i < len(text) - 3:

    if text[i+1] == text[i+2] == '-':
        if text[i] != ' ' and text[i] != '-' and text[i+3] != ' ' and text[i+3] != '-':
            text = text[:i+1] + ' -- ' + text[i+3:]
            i += 3

    i += 1

with open('data/gothicliterature_filtered.txt', 'w') as f:
    f.write(text)
