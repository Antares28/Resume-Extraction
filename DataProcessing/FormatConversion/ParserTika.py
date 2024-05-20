import tika
tika.initVM()
from tika import parser
parsed = parser.from_file('D:\BIT\毕设\数据\简历数据-个人简历网\jianli_moban_018417.docx')
print(parsed["metadata"])
print(parsed["content"])