import os, json
import pandas as pd

path_to_json = '/Users/defnecirci/Desktop/Duke_PhD/Courses/NLP/new_training_data/json'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]


files_that_contain_experimental_section = []

for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        print("json file number", index)
        data = json.load(json_file)
        titles_in_body =[]

        for text in data['body']:
            
            #titles[0] -> names of the sections
            #titles[1][0]-> first sentences
            #print(titles)
            #print(titles[0])
            titles_in_body.append(text[0])

        if  "Experimental Section" in titles_in_body:
            
            print("found it")
            index_of_methods = titles_in_body.index("Experimental Section")
            print(index_of_methods)
            files_that_contain_experimental_section.append(js)

            
            with open("new_sentences_acs.txt", 'a') as csv_file:
                
                for i in range(len(data['body'][index_of_methods][1])):
                    print("sentence",i)
                    print(data['body'][index_of_methods][1][i])
                    sentence= data['body'][index_of_methods][1][i]
                    csv_file.write(sentence)
                    csv_file.write('\n')

        else:
            print("not found it")

print(files_that_contain_experimental_section)


