import os

def merge_txt_files(folder_path, output_file):
    # Create a list to hold all the text contents
    all_texts = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt.bio'):
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                all_texts.append(file.read())

    # Write all the contents to the output file with two blank lines in between
    with open(output_file, 'w', encoding='utf-8') as file_out:
        for i, text in enumerate(all_texts):
            file_out.write(text)
            # Ensure we don't add extra newlines at the end of the file
            if i < len(all_texts) - 1:
                file_out.write('\n\n')



train_path = 'D:/BIT/毕设/data/txt_docx/txt/train'
train_output = 'D:/BIT/毕设/data/txt_docx/txt/train.txt'
test_path = 'D:/BIT/毕设/data/txt_docx/txt/test'
test_output = 'D:/BIT/毕设/data/txt_docx/txt/test.txt'

merge_txt_files(train_path, train_output)
merge_txt_files(test_path, test_output)