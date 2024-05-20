import docx2txt
import os

def convert_docx_to_txt(fileDir):
    # 获取文件夹中所有的.docx文件
    docx_files = [f for f in os.listdir(fileDir) if f.endswith('.docx')]
    
    # 创建一个存储txt文件的文件夹
    txt_folder_path = os.path.join(fileDir, 'txt_docx')
    if not os.path.exists(txt_folder_path):
        os.makedirs(txt_folder_path)

    # 对每个.docx文件进行处理
    for docx_file in docx_files:
        file_path = os.path.join(fileDir, docx_file)
        txt_file_name = os.path.splitext(docx_file)[0] + '.txt'
        txt_file_path = os.path.join(txt_folder_path, txt_file_name)
        
        # 将.docx内容转换为文本
        text = docx2txt.process(file_path)

        # 将文本写入.txt文件中
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        
        print(f"Converted {docx_file} to txt and saved in {txt_file_path}")
