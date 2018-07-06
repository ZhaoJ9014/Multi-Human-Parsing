import os

destdir_1 = 'WEB_4980'
destdir_2 = 'OTHER_20848'

files_1 = [ f for f in os.listdir(destdir_1) if os.path.isfile(os.path.join(destdir_1, f))]
files_2 = [ f for f in os.listdir(destdir_2) if os.path.isfile(os.path.join(destdir_2, f))]

text_file_1 = open('list_IMG.txt', 'w') # IMG list
text_file_2 = open('list_PNG.txt', 'w') # PNG list

for lines in files_1:
	text_file_1.write('"' + 'data/images/' + destdir_1 + '/' + lines + '",' + '\n')
	text_file_2.write('"' + 'data/annotations/' + destdir_1 + '/' + lines.split('.')[0] + '.png' + '",' + '\n')

for lines in files_2:
	text_file_1.write('"' + 'data/images/' + destdir_2 + '/' + lines + '",' + '\n')
	text_file_2.write('"' + 'data/annotations/' + destdir_2 + '/' + lines.split('.')[0] + '.png' + '",' + '\n')

text_file_1.close()
text_file_2.close()