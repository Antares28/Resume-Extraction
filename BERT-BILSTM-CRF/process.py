import json

class IDGenerator:
    def __init__(self, start_id=2000):
        self.current_id = start_id
    
    def next_id(self):
        id_value = self.current_id
        self.current_id += 1
        return id_value


def convert_to_dict_format(text, id_generator, output_file):
    # Split the text into sentences based on empty lines
    sentences = text.strip().split("\n\n")
    
    with open(output_file, 'w', encoding="utf-8") as file:
        for sentence in sentences:
            # Split each sentence into lines
            lines = sentence.strip().split("\n")
            
            # Initialize the text and labels lists for each sentence
            text_list = []
            labels_list = []

            # Iterate over each line and split by space to get character and label
            for line in lines:
                parts = line.split()
                if len(parts) == 2:  # Ensure the line has exactly two parts (char and label)
                    char, label = parts
                    text_list.append(char.strip())
                    labels_list.append(label.strip())
            
            # Create the dictionary in the desired format if text_list is not empty
            if text_list:
                result = {
                    "id": f"AT{id_generator.next_id():04d}",  # Generate id with prefix 'AT' and zero-padded to 4 digits
                    "text": text_list,
                    "labels": labels_list
                }
                # Write the result as a JSON object to the output file
                file.write(json.dumps(result, ensure_ascii=False) + '\n')


def read_input_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def save_output_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

# Main function to process the input and save the output
def main(input_file, output_file):
    # Initialize the ID generator
    id_generator = IDGenerator(start_id=1)
    
    # Read the input file
    input_text = read_input_file(input_file)
    
    # Convert the text to dictionary format
    convert_to_dict_format(input_text, id_generator, output_file)
    

# train dataset
input_train_path = "D:/BIT/毕设/data/txt_docx/txt/train.txt"
output_train_path = "D:/BIT/毕设/代码/BERT-BILSTM-CRF/data/resume/train1.txt"
# test dataset
input_test_path = "D:/BIT/毕设/data/txt_docx/txt/test.txt"
output_test_path = "D:/BIT/毕设/代码/BERT-BILSTM-CRF/data/resume/dev1.txt"

# main(input_train_path, output_train_path)
main(input_test_path, output_test_path)
