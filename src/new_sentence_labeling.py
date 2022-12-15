import os, json
import pandas as pd

path_to_json = '/Users/defnecirci/Desktop/Duke_PhD/Courses/NLP/new_training_data/new_sentences_acs.json'



constituent_sentences = 0
property_sentences = 0
action_sentences = 0
non_relevant_sentences = 0

cons =["Co." , "Ltd." ,"USA" ,"China" ,"India" ,"purchase" ,"procure","France","Belgium"]
prop=["weight" ,"volume" , "thickness" ,"melt flow index" , "density" ,  "aspect ratio" , "glass transition ratio" , "diameter" , "viscocity" , "elongation" , "yield stress" , "conductivity" , "boiling point" , "size","properties","dielectric"]

act=["add" , "stir" ,"dry" , "dried" , "anneal" , "disperse" , "wash","prepare","mix" ,"heated","rinsed"]

with open(path_to_json) as json_file:
    data = json.load(json_file)


    for text in data:
        print(text)
            #titles[0] -> names of the sections
            #titles[1][0]-> first sentences
            #print(titles)
            #print(titles[0])
        #eliminatig short texts as they are likely to be not sentences
        if(len(text) >= 3):
            if any(x in  text for x in cons ) :
                    
                print("constituent found")

                    #print(len(text[index_of_methods]))
                    
                with open("constituent_sentences.txt", 'a') as csv_file:
                
                    csv_file.write(text)
                    csv_file.write('\n')
                    constituent_sentences = constituent_sentences + 1
                    
            elif  any(x in text for x in prop ):
                    
                print("property found")
                property_sentences = property_sentences + 1

                    #print(len(text[index_of_methods]))
                    
                with open("property_sentences.txt", 'a') as csv_file:
                
                    csv_file.write(text)
                    csv_file.write('\n')
     
            elif  any(x in  text for x in act ):
                    
                print("action found")

                    #print(len(text[index_of_methods]))
                    
                with open("action_sentences.txt", 'a') as csv_file:
                
                    csv_file.write(text)
                    csv_file.write('\n')
                    action_sentences = action_sentences + 1

            else:
                print("non relevant found")
                with open("non_relevant_sentences.txt", 'a') as csv_file:
                
                    csv_file.write(text)
                    csv_file.write('\n')
                    non_relevant_sentences = non_relevant_sentences + 1
                
print("Number of constituent sentences is: ",constituent_sentences)
print("Number of property sentences is: ",property_sentences)
print("Number of non relevant sentences is: ",non_relevant_sentences)
print("Number of action sentences is: ",action_sentences)
