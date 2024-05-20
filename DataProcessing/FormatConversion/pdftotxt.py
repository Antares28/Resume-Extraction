import os
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator

# 解析文本用到的类：
# PDFParser（文档分析器）：从文件中获取数据
# PDFDocument（文档对象）：保存文件数据
# PDFPageInterpreter（解释器）：处理页面内容
# PDFResourceManager（资源管理器）：存储共享资源
# PDFDevice:将解释器处理好的内容转换为我们所需要的
# PDFPageAggregator（聚合器）:读取文档对象
# LAParams（参数分析器）

# convert one PDF file to TXT file
def onePdfToTxt(filepath, outpath):
    try:
        #rb以二进制读模式打开本地pdf文件
        fp = open(filepath, 'rb')
        outfp = open(outpath, 'w', encoding='utf-8')
        #创建一个pdf文档分析器
        parser = PDFParser(fp)
        #创建一个PDF文档
        doc= PDFDocument(parser)
        
        # 检测文档是否提供txt转换，不提供就忽略
        if not doc.is_extractable:
            pass
        else:
            #创建PDf资源管理器
            resource = PDFResourceManager()
            #创建一个PDF参数分析器
            laparams = LAParams()
            #创建聚合器,用于读取文档的对象
            device = PDFPageAggregator(resource,laparams=laparams)
            #创建解释器，对文档编码，解释成Python能够识别的格式
            interpreter = PDFPageInterpreter(resource,device)
            # 循环遍历列表，每次处理一页的内容
            for page in enumerate(PDFPage.create_pages(doc)):
                interpreter.process_page(page[1])
                layout = device.get_result()
                
                #这里layout是一个LTPage对象,里面存放着这个page解析出的各种对象
                for out in layout:
                    #判断是否含有get_text()方法，获取我们想要的文字
                    if hasattr(out,"get_text"):
                        text=out.get_text()
                        outfp.write(text+'\n')
            fp.close()
            outfp.close()
    except Exception as e:
         print (e)



# convert all PDF files in a folder to TXT files
def convert_pdf_to_txt(fileDir):
    # 获取文件夹中所有的.pdf文件
    pdf_files = [f for f in os.listdir(fileDir) if f.endswith('.pdf')]
    
    # 创建一个存储txt文件的文件夹
    txt_folder_path = os.path.join(fileDir, 'txt_pdf')
    if not os.path.exists(txt_folder_path):
        os.makedirs(txt_folder_path)
        
    # 对每个.pdf文件进行处理
    for pdf_file in pdf_files:
        file_Path = os.path.join(fileDir, pdf_file)
        txt_file_name = os.path.splitext(pdf_file)[0] + '.txt'
        txt_file_path = os.path.join(txt_folder_path, txt_file_name)
        
        onePdfToTxt(file_Path, txt_file_path)
        print(f"Converted {pdf_file} to txt and saved in {txt_file_path}")